import jax
import jax.numpy as jnp
from jax import lax
from jax.lax import stop_gradient as sg
from jax.sharding import PartitionSpec as P
from flax import nnx
from einops import rearrange, reduce

import optax
from optax import sigmoid_binary_cross_entropy as binary_ce
from optax import softmax_cross_entropy_with_integer_labels as softmax_ce

import random
import contextlib
from dataclasses import dataclass
from functools import partial
from collections import defaultdict
from absl import logging
from clu import metric_writers, periodic_actions
import simple_parsing

from datasets import load_dataset
from utils import Loader

logging.set_verbosity(logging.INFO)

jax.config.update("jax_debug_nans", True)


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

        # stop gradients for T-1 steps
        def body(_, carry):
            y, z = carry
            y, z = self.latent_recursion(x=x, y=y, z=z, n=n)
            return y, z

        y, z = lax.fori_loop(0, T - 1, body, (y, z))
        y, z = sg(y), sg(z)

        # final step with gradients
        y, z = self.latent_recursion(x=x, y=y, z=z, n=n)
        return (y, z), self.output_head(y), self.Q_head(y)

    def predict(self, x_input, N_supervision=16, n=6, T=3, rngs=None):
        x = self.input_embedding(x_input)
        batch_size, seq_len, _ = x.shape
        y, z = (
            self.init_y(batch_size, seq_len, rngs),
            self.init_z(batch_size, seq_len, rngs),
        )

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

    def __init__(self, h_dim, expansion, linear, rngs):
        inter_dim = _find_multiple(round(h_dim * expansion * 2 / 3), 256)
        self.W1 = linear(h_dim, inter_dim, use_bias=False, rngs=rngs)
        self.W3 = linear(h_dim, inter_dim, use_bias=False, rngs=rngs)
        self.W2 = linear(inter_dim, h_dim, use_bias=False, rngs=rngs)

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


# TODO flax.nnx.initializers.truncated_normal?
class InitState(nnx.Module):
    def __init__(self, mode, h_dim, rngs):
        self.scale = jnp.sqrt(1 / h_dim)  # match input emb scale
        self.gen_state = partial(jax.random.normal, shape=(1, 1, h_dim))
        if mode == "static":
            self.state = self.gen_state(rngs.next()) * self.scale
        else:
            self.state = None

    def __call__(self, batch_size, seq_len, rngs=None):
        if self.state is None:
            base = self.gen_state(rngs.next()) * self.scale
        else:
            base = self.state
        return jnp.broadcast_to(base, (batch_size, seq_len, base.shape[-1]))


def loss_fn(model, x, y, z, y_true, config):
    (y, z), y_hat, q_hat = model(x=x, y=y, z=z, n=config.n, T=config.T)
    y_hat, q_hat = y_hat.astype(jnp.float32), q_hat.astype(jnp.float32)

    rec_loss = softmax_ce(
        logits=rearrange(y_hat, "b l c -> (b l) c"),
        labels=rearrange(y_true, "b l -> (b l)"),
    ).mean()

    should_halt = (y_hat.argmax(axis=-1) == y_true).all(axis=-1, keepdims=True)
    halt_loss = binary_ce(logits=q_hat, labels=should_halt).mean()

    loss = rec_loss + config.halt_loss_weight * halt_loss
    return loss, (y, z, y_hat, q_hat)


grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)


def pred_metrics(y_hat, y_true, prefix):
    N, *_ = y_hat.shape
    preds = y_hat.argmax(axis=-1)
    cell_acc = (preds == y_true).mean(axis=(-1, -2))
    solved_acc = (preds == y_true).all(axis=-1).mean(axis=-1)
    return {
        f"{prefix}/cell_acc": cell_acc[-1],
        f"{prefix}/cell_acc_first_delta": cell_acc[-1] - cell_acc[0],
        f"{prefix}/cell_acc_halfway_delta": cell_acc[-1] - cell_acc[N // 2],
        f"{prefix}/solved_acc": solved_acc[-1],
        f"{prefix}/solved_acc_first_delta": solved_acc[-1] - solved_acc[0],
        f"{prefix}/solved_acc_halfway_delta": solved_acc[-1] - solved_acc[N // 2],
    }


@nnx.jit(static_argnames=("config"))
def train_step(model, ema_model, opt, batch, config, rngs):
    model.train()

    x_input, y_true = batch["inputs"], batch["labels"]
    x = model.input_embedding(x_input)
    y, z = (
        model.init_y(config.batch_size, config.seq_len, rngs),
        model.init_z(config.batch_size, config.seq_len, rngs),
    )

    def sup_step(carry, _):
        model, opt, y, z = carry
        (loss, (y, z, y_hat, q_hat)), grads = grad_fn(model, x, y, z, y_true, config)
        opt.update(model, grads)
        # IDEA -- stay on policy here
        return (model, opt, y, z), (loss, y_hat, q_hat, optax.global_norm(grads))

    (model, opt, y, z), (losses, y_hats, q_hats, norms) = jax.lax.scan(
        sup_step, (model, opt, y, z), None, length=config.N_supervision
    )
    new_ema_model = optax.incremental_update(
        model, ema_model, step_size=1 - config.ema_beta
    )

    return (
        model,
        opt,
        new_ema_model,
        {
            "train/loss": losses[-1],
            "train/loss_first_delta": losses[-1] - losses[0],
            "train/loss_halfway_delta": losses[-1] - losses[config.N_supervision // 2],
            "train/loss_std": jnp.std(losses),
            "train/grad_norm": norms[-1],
            **pred_metrics(y_hats, y_true, prefix="train"),
        },
    )


@nnx.jit(static_argnames=("config"))
def eval_step(model, batch, config, rngs):
    model.eval()

    x_input, y_true = batch["inputs"], batch["labels"]
    y_hats = model.predict(
        x_input, N_supervision=config.N_supervision, n=config.n, T=config.T, rngs=rngs
    )
    return {
        **pred_metrics(y_hats, y_true, prefix="eval"),
        "batch_size": x_input.shape[0],
    }


def evaluate_epoch(model, data_iter, config, rngs, mesh=None):
    totals = defaultdict(float)
    total_weight = 0.0

    for batch in data_iter:
        if mesh is not None:
            batch = shard_batch(batch)
        metrics = eval_step(model, batch, config, rngs)
        bs = float(metrics.pop("batch_size"))
        for k, v in metrics.items():
            totals[k] += v * bs
        total_weight += bs
    return {k: v / total_weight for k, v in totals.items()}


def model_factory(config, param_dtype, compute_dtype, rngs):
    Linear = partial(
        nnx.Linear, dtype=compute_dtype, param_dtype=param_dtype, rngs=rngs
    )

    model = TRM(
        net=Net(
            config.seq_len,
            config.h_dim,
            expansion=config.mlp_factor,
            n_layers=config.n_layers,
            linear=Linear,
            rngs=rngs,
        ),
        output_head=Linear(config.h_dim, config.vocab_size),
        Q_head=nnx.Sequential(
            partial(reduce, pattern="b l h -> b h", reduction="mean"),
            Linear(config.h_dim, 1),
        ),
        input_embedding=nnx.Embed(
            config.vocab_size, config.h_dim, param_dtype=param_dtype, rngs=rngs
        ),
        init_y=InitState(config.init_state, config.h_dim, rngs=rngs),
        init_z=InitState(config.init_state, config.h_dim, rngs=rngs),
    )

    return model


def shard_batch(batch):
    # data parallel sharding
    return {
        "inputs": jax.device_put(batch["inputs"], P("data", None)),
        "labels": jax.device_put(batch["labels"], P("data", None)),
    }


@dataclass(frozen=True)
class Config:
    dataset: str = "emiliocantuc/sudoku-extreme-1k-aug-1000"
    seq_len: int = 81
    vocab_size: int = 10

    n_layers: int = 2
    h_dim: int = 512
    mlp_factor: int = 4
    init_state: str = "random"

    N_supervision: int = 16
    n: int = 6
    T: int = 3
    halt_loss_weight: float = 0.0

    batch_size: int = 768
    lr: float = 1e-4
    lr_warmup_steps: int = 2000 // 16
    weight_decay: float = 1.0
    ema_beta: float = 0.999**16
    steps: int = None

    half_precision: bool = False
    val_every: int = 250
    workdir: str = None
    seed: int = None


if __name__ == "__main__":
    config = simple_parsing.parse(Config)

    tpu = jax.default_backend() == "tpu"
    param_dtype = jnp.float32
    compute_dtype = jnp.bfloat16 if tpu and config.half_precision else jnp.float32
    seed = config.seed or random.randint(0, 2**32 - 1)
    rngs = nnx.Rngs(seed)

    num_devices = jax.device_count()
    if num_devices > 1:
        mesh = jax.make_mesh((num_devices,), ("data",))
        nnx.use_eager_sharding(True)
    else:
        mesh = None
    print(f"Using mesh: {mesh}")

    train_ds = load_dataset(config.dataset, split="train")
    # train_ds = load_dataset(config.dataset, split=f"train[:{config.batch_size}]") # debug by overfitting to single batch
    val_ds = load_dataset(config.dataset, split="test[:1024]")
    test_ds = load_dataset(config.dataset, split="test")

    train_loader = Loader(train_ds, batch_size=config.batch_size, shuffle_seed=seed)
    val_loader = Loader(val_ds, batch_size=config.batch_size, epochs=1)
    test_loader = Loader(test_ds, batch_size=config.batch_size, epochs=1)

    mesh_ctx = jax.set_mesh(mesh) if mesh is not None else contextlib.nullcontext()

    with mesh_ctx:
        model = model_factory(config, param_dtype, compute_dtype, rngs)
        n_params = sum(
            jax.tree.map(jnp.size, jax.tree.leaves(nnx.state(model, nnx.Param)))
        )
        print(f"No. of parameters: {n_params}")

        lr_schedule = optax.warmup_constant_schedule(
            init_value=0.0, peak_value=config.lr, warmup_steps=config.lr_warmup_steps
        )

        opt = nnx.Optimizer(
            model=model,
            tx=optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(
                    learning_rate=lr_schedule,
                    b1=0.9,
                    b2=0.95,
                    eps=1e-4 if config.half_precision else 1e-8,
                    weight_decay=config.weight_decay,
                    # don't apply weight decay to biases or norm params
                    # mask=lambda params: jax.tree.map(
                    #     lambda p: getattr(p, "ndim", 0) > 1, params
                    # ),
                ),
            ),
            wrt=nnx.Param,
        )

        ema_model = nnx.clone(model)

        # logging
        writer = metric_writers.create_default_writer(
            config.workdir, just_logging=jax.process_index() > 0
        )
        writer.write_hparams(vars(config))
        writer.write_scalars(0, {"hparams/n_params": n_params})

        def _val_callback(step, t):
            writer.write_scalars(
                step, evaluate_epoch(ema_model, val_loader, config, rngs, mesh)
            )

        hooks = [
            periodic_actions.ReportProgress(
                num_train_steps=config.steps, writer=writer
            ),
            periodic_actions.PeriodicCallback(
                every_steps=config.val_every,
                on_steps=[config.steps],
                callback_fn=_val_callback,
            ),
        ]
        if config.workdir is not None and jax.process_index() == 0:
            hooks.append(
                periodic_actions.Profile(num_profile_steps=5, logdir=config.workdir)
            )

        with metric_writers.ensure_flushes(writer):
            for step, batch in enumerate(train_loader, start=1):
                if mesh is not None:
                    batch = shard_batch(batch)
                model, opt, ema_model, train_metrics = train_step(
                    model, ema_model, opt, batch, config, rngs
                )
                train_metrics["train/lr"] = lr_schedule(step)
                writer.write_scalars(step, train_metrics)

                for h in hooks:
                    h(step)

                if step >= config.steps:
                    break

        # TODO:
        # - run full experiments
        # - add checkpointing?
