# /// script
# requires-python = "==3.12"
# dependencies = [
#     "clu",
#     "datasets",
#     "einops",
#     "flax",
#     "jax",
#     "optax",
#     "tensorflow",
# ]
# ///
import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax.lax import stop_gradient as sg
from flax import nnx

import optax
from optax import sigmoid_binary_cross_entropy as binary_ce
from optax import softmax_cross_entropy_with_integer_labels as softmax_ce

from einops import rearrange, reduce

from functools import partial
from collections import deque
import argparse
import random

from datasets import load_dataset, Dataset
from clu import metric_writers, periodic_actions

from absl import logging

logging.set_verbosity(logging.INFO)


class TRM(nnx.Module):
    def __init__(
        self,
        net,
        output_head,
        Q_head,
        input_embedding,
        init_y,
        init_z,
    ):
        self.net = net
        self.output_head = output_head
        self.Q_head = Q_head
        self.input_embedding = input_embedding
        self.init_y = init_y
        self.init_z = init_z

    def latent_recursion(self, *, x, y, z, n=6):
        # refine the latent (z) n times
        def refine_latent(_, carry):
            y, z = carry
            z = self.net(x=x, y=y, z=z)
            return y, z

        y, z = lax.fori_loop(0, n, refine_latent, (y, z))
        y = self.net(x=jnp.zeros_like(x), y=y, z=z)  # refine output (y) once
        return y, z

    def __call__(self, *, x, y, z, n=6, T=3):  # deep recursion
        # run T steps; stop grads for steps < T-1
        def body(i, carry):
            y, z = carry
            y, z = self.latent_recursion(x=x, y=y, z=z, n=n)
            is_last = i == (T - 1)
            y = lax.select(is_last, y, sg(y))
            z = lax.select(is_last, z, sg(z))
            return y, z

        y, z = lax.fori_loop(0, T, body, (y, z))
        return (sg(y), sg(z)), self.output_head(y), self.Q_head(y)

    def predict(self, x_input, N_supervision=16, n=6, T=3, rngs=None):
        x = self.input_embedding(x_input)
        y, z = self.init_y(rngs), self.init_z(rngs)

        def supervision_step(carry, _):
            y, z = carry
            (y, z), y_hat, _ = self(x=x, y=y, z=z, n=n, T=T)
            return (y, z), y_hat

        _, y_hats = lax.scan(supervision_step, (y, z), None, length=N_supervision)
        return y_hats  # (N, B, L, C)


def _find_multiple(a, b):
    return (-(a // -b)) * b


# TODO swap out with less elementwise ops?
class SwiGLU(nnx.Module):
    """SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) * W3x)"""

    def __init__(self, d_model, expansion, linear, rngs):
        d_inter = _find_multiple(round(d_model * expansion * 2 / 3), 256)
        self.W1 = linear(d_model, d_inter, use_bias=False, rngs=rngs)
        self.W3 = linear(d_model, d_inter, use_bias=False, rngs=rngs)
        self.W2 = linear(d_inter, d_model, use_bias=False, rngs=rngs)

    def __call__(self, x):
        return self.W2(nnx.silu(self.W1(x)) * self.W3(x))


class MixerBlock(nnx.Module):
    def __init__(self, seq_len, h_dim, expansion, linear, rngs):
        self.l_mixer = SwiGLU(seq_len, expansion, linear, rngs=rngs)
        self.d_mixer = SwiGLU(h_dim, expansion, linear, rngs=rngs)
        self.l_norm = nnx.RMSNorm(h_dim, use_scale=True, rngs=rngs, dtype=jnp.float32)
        self.d_norm = nnx.RMSNorm(h_dim, use_scale=True, rngs=rngs, dtype=jnp.float32)

    def __call__(self, h):
        o = self.l_norm(h)
        o = rearrange(o, "b l d -> b d l")
        o = self.l_mixer(o)
        o = rearrange(o, "b d l -> b l d")

        h = o + h

        o = self.d_norm(h)
        o = self.d_mixer(o)
        return o + h


def Net(seq_len, h_dim, expansion, n_layers, linear, rngs):
    return nnx.Sequential(
        lambda x, y, z: x + y + z,
        *[
            MixerBlock(
                seq_len=seq_len,
                h_dim=h_dim,
                expansion=expansion,
                linear=linear,
                rngs=rngs,
            )
            for _ in range(n_layers)
        ],
        nnx.RMSNorm(h_dim, use_scale=True, rngs=rngs, dtype=jnp.float32),
    )


class InitState(nnx.Module):
    def __init__(self, mode, batch_size, seq_len, h_dim, rngs):
        self.scale = jnp.sqrt(1 / h_dim)  # match input emb scale
        self.gen_state = partial(jax.random.normal, shape=(batch_size, seq_len, h_dim))
        if mode == "static":
            self.state = self.gen_state(rngs.next()) * self.scale
        else:
            self.state = None

    def __call__(self, rngs=None):
        if self.state is None:
            return self.gen_state(rngs.next()) * self.scale
        return self.state


def loss_fn(model, x, z, y, y_true, n=6, T=3, halt_loss_weight=0.5):
    (y, z), y_hat, q_hat = model(x=x, y=y, z=z, n=n, T=T)
    y_hat, q_hat = y_hat.astype(jnp.float32), q_hat.astype(jnp.float32)

    rec_loss = softmax_ce(
        logits=rearrange(y_hat, "b l c -> (b l) c"),
        labels=rearrange(y_true, "b l -> (b l)"),
    ).mean()

    halt_targets = (y_hat.argmax(axis=-1) == y_true).all(axis=-1, keepdims=True)
    halt_loss = binary_ce(logits=q_hat, labels=halt_targets).mean()

    loss = rec_loss + halt_loss_weight * halt_loss
    return loss, (y, z)


@nnx.jit(static_argnames=("grad_fn", "ema_beta", "N_supervision"))
def train_step(grad_fn, model, ema_model, opt, batch, ema_beta, N_supervision, rngs):
    model.train()

    x_input, y_true = batch["inputs"], batch["labels"]
    x = model.input_embedding(x_input)
    y, z = model.init_y(rngs), model.init_z(rngs)

    def sup_step(carry, _):
        model, opt, y, z = carry
        (loss, (y, z)), grads = grad_fn(model, x, z, y, y_true)
        opt.update(model, grads)
        # IDEA -- stay on policy here
        return (model, opt, y, z), (loss, optax.global_norm(grads))

    (model, opt, y, z), (losses, norms) = jax.lax.scan(
        sup_step, (model, opt, y, z), None, length=N_supervision
    )

    new_ema_model = optax.incremental_update(model, ema_model, step_size=1 - ema_beta)

    return (
        model,
        opt,
        new_ema_model,
        {
            "train/loss": losses[-1],
            "train/first_loss": losses[0],
            "train/halfway_loss": losses[N_supervision // 2],
            "train/grad_norm": norms[-1],
        },
    )


@nnx.jit(static_argnames=("n", "T", "N_supervision"))
def eval_step(model, batch, n, T, N_supervision, rngs):
    model.eval()

    x_input, y_true = batch["inputs"], batch["labels"]
    y_hats = model.predict(x_input, N_supervision=N_supervision, n=n, T=T, rngs=rngs)
    y_hat = y_hats[-1]  # just final prediction -- IDEA: could ensemble, etc.

    preds = y_hat.argmax(axis=-1)
    solved_acc = (preds == y_true).all(axis=-1).mean()
    cell_acc = (preds == y_true).mean()
    return solved_acc, cell_acc


def evaluate(model, data_iter, eval_step):
    total_solved_acc, total_cell_acc = 0.0, 0.0
    total_samples = 0
    for batch in data_iter:
        solved_acc, cell_acc = eval_step(model, batch)

        bs = batch["inputs"].shape[0]
        total_solved_acc += solved_acc * bs
        total_cell_acc += cell_acc * bs
        total_samples += bs

    return total_solved_acc / total_samples, total_cell_acc / total_samples


class Loader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size,
        epochs=None,
        shuffle_seed=None,
        prefetch_size=2,
    ):
        if shuffle_seed is not None:
            dataset = dataset.shuffle(seed=shuffle_seed)
        dataset = dataset.with_format("numpy")

        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.prefetch_size = prefetch_size

    def iter_batch(self):
        N, bs = len(self.dataset), self.batch_size
        epochs = self.epochs
        while epochs is None or epochs > 0:
            for i in range(0, N, bs):
                j = min(i + bs, N)
                if (j - i) < bs:  # always drop incomplete batch
                    break
                batch = self.dataset[i:j]
                yield {
                    "inputs": np.asarray(batch["inputs"], dtype=np.int32),
                    "labels": np.asarray(batch["labels"], dtype=np.int32),
                }
            if epochs is not None:
                epochs -= 1

    def prefetch_to_device(self):  # TODO benchmark to see if necessary
        size = self.prefetch_size
        it = iter(self.iter_batch())
        put_in_device = jax.device_put

        if size <= 0:
            yield from self.iter_batch()
            return

        # prime
        q = deque(
            (put_in_device(batch) for _, batch in zip(range(size), it)), maxlen=size
        )

        for batch in it:
            if q:
                yield q.popleft()
            q.append(put_in_device(batch))

        while q:
            yield q.popleft()

    def __iter__(self):
        return self.prefetch_to_device()


def model_factory(args, param_dtype, compute_dtype, rngs):
    Linear = partial(
        nnx.Linear, dtype=compute_dtype, param_dtype=param_dtype, rngs=rngs
    )

    init_state = partial(
        InitState,
        args.init_state,
        args.batch_size,
        args.seq_len,
        args.h_dim,
        rngs=rngs,
    )

    model = TRM(
        net=Net(
            args.seq_len,
            args.h_dim,
            expansion=args.mlp_factor,
            n_layers=args.n_layers,
            linear=Linear,
            rngs=rngs,
        ),
        output_head=Linear(args.h_dim, args.vocab_size),
        Q_head=nnx.Sequential(
            partial(reduce, pattern="b l h -> b h", reduction="mean"),
            Linear(args.h_dim, 1),
        ),
        input_embedding=nnx.Embed(
            args.vocab_size, args.h_dim, param_dtype=param_dtype, rngs=rngs
        ),
        init_y=init_state(),
        init_z=init_state(),
    )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--h_dim", type=int, default=512)
    parser.add_argument("--mlp_factor", type=int, default=4)
    parser.add_argument(
        "--init_state", type=str, default="random", choices=["static", "random"]
    )

    parser.add_argument("--N_supervision", type=int, default=16)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--T", type=int, default=3)
    parser.add_argument("--halt_loss_weight", type=float, default=0.5)

    parser.add_argument("--batch_size", type=int, default=768)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=2000 // 16)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--ema_beta", type=float, default=0.999**16)
    parser.add_argument("--epochs", type=int, default=60_000 // 16)
    parser.add_argument("--steps", type=int, default=None)

    # parser.add_argument("--debug", action="store_true")
    parser.add_argument("--half_precision", action="store_true")
    # parser.add_argument("--k_passes", type=int, default=1)
    parser.add_argument("--val_every", type=int, default=100)
    # parser.add_argument("--eval_only", action="store_true")
    # parser.add_argument("--skip_eval", action="store_true")
    # parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    args.seq_len, args.vocab_size = 81, 10

    tpu = jax.default_backend() == "tpu"
    param_dtype = jnp.float32
    compute_dtype = jnp.bfloat16 if tpu and args.half_precision else jnp.float32

    args.seed = args.seed or random.randint(0, 2**32 - 1)
    rngs = nnx.Rngs(args.seed)

    ds_path = "emiliocantuc/sudoku-extreme-1k-aug-1000"
    train_ds = load_dataset(ds_path, split="train")
    val_ds = load_dataset(ds_path, split="test[:1024]")
    test_ds = load_dataset(ds_path, split="test")

    train_loader = Loader(
        train_ds, batch_size=args.batch_size, epochs=args.epochs, shuffle_seed=args.seed
    )
    val_loader = Loader(val_ds, batch_size=args.batch_size, epochs=1)
    test_loader = Loader(test_ds, batch_size=args.batch_size, epochs=1)

    args.steps = args.steps or args.epochs * (len(train_ds) // args.batch_size)

    grad_fn = nnx.value_and_grad(
        partial(loss_fn, n=args.n, T=args.T, halt_loss_weight=args.halt_loss_weight),
        has_aux=True,
    )

    _eval_step = partial(
        eval_step, n=args.n, T=args.T, N_supervision=args.N_supervision, rngs=rngs
    )

    model = model_factory(args, param_dtype, compute_dtype, rngs)
    n_params = sum(jax.tree.map(jnp.size, jax.tree.leaves(nnx.state(model, nnx.Param))))
    print(f"No. of parameters: {n_params}")

    lr_schedule = optax.warmup_constant_schedule(
        init_value=0.0, peak_value=args.lr, warmup_steps=args.lr_warmup_steps
    )

    opt = nnx.Optimizer(
        model=model,
        tx=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=lr_schedule,
                weight_decay=args.weight_decay,
                b1=0.9,
                b2=0.95,
            ),
        ),
        wrt=nnx.Param,
    )

    ema_model = nnx.clone(model)

    # logging
    writer = metric_writers.create_default_writer(
        args.workdir, just_logging=jax.process_index() > 0
    )
    writer.write_hparams(vars(args))
    writer.write_scalars(0, {"hparams/n_params": n_params})

    def _val_callback(step, t):
        solved_acc, cell_acc = evaluate(ema_model, val_loader, _eval_step)
        writer.write_scalars(
            step, {"eval/solved_acc": solved_acc, "eval/cell_acc": cell_acc}
        )

    hooks = [
        periodic_actions.ReportProgress(num_train_steps=args.steps, writer=writer),
        periodic_actions.PeriodicCallback(
            every_steps=args.val_every, on_steps=[args.steps], callback_fn=_val_callback
        ),
    ]
    if args.workdir is not None and jax.process_index() == 0:
        hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=args.workdir))

    with metric_writers.ensure_flushes(writer):
        for step, batch in enumerate(train_loader, start=1):
            model, opt, ema_model, train_metrics = train_step(
                grad_fn,
                model,
                ema_model,
                opt,
                batch,
                args.ema_beta,
                args.N_supervision,
                rngs,
            )
            train_metrics["train/lr"] = lr_schedule(step)
            writer.write_scalars(step, train_metrics)

            for h in hooks:
                h(step)

            if step >= args.steps:
                break

    # TODO:
    # - add checkpointing
    # - ignore grad of halted tokens
    # - run full experiments
