import jax
import jax.numpy as jnp
from jax import lax
from jax.lax import stop_gradient as sg
from flax import nnx
import orbax.checkpoint as ocp
from einops import rearrange, reduce

import optax
from optax import sigmoid_binary_cross_entropy as binary_ce
from optax import softmax_cross_entropy_with_integer_labels as softmax_ce

import os
import random
import contextlib
from dataclasses import dataclass
from functools import partial
from collections import defaultdict
from absl import logging
from clu import metric_writers, periodic_actions
import simple_parsing

from datasets import load_dataset
from utils import (
    Loader,
    restore_checkpoint,
    save_checkpoint,
    calc_metric_over_batches,
    shard_batch,
)

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

    def predict(self, x_input, y=None, z=None, N_sup=16, n=6, T=3, rngs=None):
        x = self.input_embedding(x_input)
        batch_size, seq_len, _ = x.shape

        if y is None or z is None:
            y, z = (
                self.init_y(batch_size, seq_len, rngs),
                self.init_z(batch_size, seq_len, rngs),
            )

        def supervision_step(carry, _):
            y, z = carry
            (y, z), y_hat, q_hat = self(x=x, y=y, z=z, n=n, T=T)
            return (y, z), (y_hat, q_hat)

        (ys, zs), (y_hats, q_hats) = lax.scan(
            supervision_step, (y, z), None, length=N_sup
        )
        return y_hats, q_hats, (ys, zs)


def _find_multiple(a, b):
    return (-(a // -b)) * b


class SwiGLU(nnx.Module):
    """SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) * W3x)"""

    def __init__(self, h_dim, expansion, linear):
        inter_dim = _find_multiple(round(h_dim * expansion * 2 / 3), 256)
        self.W1 = linear(h_dim, inter_dim, use_bias=False)
        self.W3 = linear(h_dim, inter_dim, use_bias=False)
        self.W2 = linear(inter_dim, h_dim, use_bias=False)

    def __call__(self, x):
        return self.W2(nnx.silu(self.W1(x)) * self.W3(x))


class MixerBlock(nnx.Module):
    def __init__(self, seq_len, h_dim, expansion, linear, rngs):
        self.l_mixer = SwiGLU(seq_len, expansion, linear)
        self.d_mixer = SwiGLU(h_dim, expansion, linear)
        self.l_norm = nnx.RMSNorm(h_dim, use_scale=False, rngs=rngs, dtype=jnp.float32)
        self.d_norm = nnx.RMSNorm(h_dim, use_scale=False, rngs=rngs, dtype=jnp.float32)

    def __call__(self, h):
        o = self.l_norm(h)
        o = rearrange(o, "b l d -> b d l")
        o = self.l_mixer(o)
        o = rearrange(o, "b d l -> b l d")

        h = o + h

        o = self.d_norm(h)
        o = self.d_mixer(o)
        return o + h


class Net(nnx.Module):
    def __init__(self, seq_len, h_dim, expansion, n_layers, linear, rngs):
        # normalize x, y, z separately before adding
        norm = partial(nnx.RMSNorm, num_features=h_dim, dtype=jnp.float32, rngs=rngs)
        self.x_norm, self.y_norm, self.z_norm = (norm(), norm(), norm())

        self.net = nnx.Sequential(
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
            norm(rngs=rngs),
        )

    def __call__(self, *, x, y, z):
        return self.net(self.x_norm(x) + self.y_norm(y) + self.z_norm(z))


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


def loss_fn(model, x, y, z, y_true, alive, cfg, T):
    (y, z), y_hat, q_hat = model(x=x, y=y, z=z, n=cfg.n, T=T)

    y_hat, q_hat = y_hat.astype(jnp.float32), q_hat.astype(jnp.float32)
    alive = alive.astype(jnp.float32)
    total_alive = alive.sum().clip(min=1.0)
    bs, seq_len, _ = x.shape

    rec_loss = (
        rearrange(
            softmax_ce(
                logits=rearrange(y_hat, "b l c -> (b l) c"),
                labels=rearrange(y_true, "b l -> (b l)"),
            ),
            "(b l) -> b l",
            b=bs,
        )
        * alive
    ).sum() / (total_alive * seq_len)

    should_halt = (y_hat.argmax(axis=-1) == y_true).all(axis=-1, keepdims=True)
    halt_loss = (
        rearrange(binary_ce(logits=q_hat, labels=should_halt), "b 1 -> b 1") * alive
    ).sum() / total_alive

    loss = rec_loss + cfg.halt_loss_weight * halt_loss
    return loss, (y, z, y_hat, q_hat)


grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)


def pred_metrics(preds, y_true, prefix, train_N=None, return_curves=False):
    N, *_ = preds.shape
    cell_acc = (preds == y_true).mean(axis=(-1, -2))
    solved_acc = (preds == y_true).all(axis=-1).mean(axis=-1)
    metrics = {
        f"{prefix}/cell_acc": cell_acc[-1],
        f"{prefix}/cell_acc_first_delta": cell_acc[-1] - cell_acc[0],
        f"{prefix}/cell_acc_halfway_delta": cell_acc[-1] - cell_acc[N // 2],
        f"{prefix}/solved_acc": solved_acc[-1],
        f"{prefix}/solved_acc_first_delta": solved_acc[-1] - solved_acc[0],
        f"{prefix}/solved_acc_halfway_delta": solved_acc[-1] - solved_acc[N // 2],
    }
    if train_N is not None and train_N < N:
        metrics[f"{prefix}/cell_acc_train_length_delta"] = (
            cell_acc[-1] - cell_acc[train_N - 1]
        )
        metrics[f"{prefix}/solved_acc_train_length_delta"] = (
            solved_acc[-1] - solved_acc[train_N - 1]
        )
    if return_curves:
        return metrics, cell_acc, solved_acc
    else:
        return metrics


@nnx.jit(static_argnames=("cfg",))
def train_step(model, ema_model, opt, batch, cfg, rngs):
    model.train()

    x_input, y_true = batch["inputs"], batch["labels"]
    x = model.input_embedding(x_input)
    bs, seq_len, _ = x.shape

    y, z = (
        model.init_y(bs, seq_len, rngs),
        model.init_z(bs, seq_len, rngs),
    )

    if cfg.rand_T:
        T = jax.random.randint(rngs(), shape=(), minval=1, maxval=cfg.T + 1)
    else:
        T = cfg.T

    min_steps = (
        jax.random.uniform(rngs(), (bs, 1)) <= cfg.halt_exploration_prob
    ) * jax.random.randint(rngs(), (bs, 1), 2, cfg.N_sup + 1)

    def sup_step(carry, _):
        step, model, opt, y_in, z_in, alive, rngs = carry

        # update step
        (loss, (y, z, y_hat, q_hat)), grads = grad_fn(
            model, x, y_in, z_in, y_true, alive, cfg, T
        )
        opt.update(model, grads)

        if cfg.stay_on_policy:
            (y, z), _, _ = model(x=x, y=y_in, z=z_in, n=cfg.n, T=T)

        # add noise to latents (new) TODO mess with std shape
        corr_std = (
            (jax.random.uniform(rngs(), (bs, 1, 1)) >= cfg.corruption_clean_prop)
            * jax.random.uniform(rngs(), (bs, 1, 1))
            * cfg.max_corruption_std
        )
        y = (
            y
            + jax.random.normal(rngs(), y.shape) * y.std((-1), keepdims=True) * corr_std
        )
        z = (
            z
            + jax.random.normal(rngs(), z.shape) * z.std((-1), keepdims=True) * corr_std
        )

        keep_alive = q_hat < 0.0
        alive = alive & (keep_alive | (step < min_steps))

        return (step + 1, model, opt, y, z, alive, rngs), (
            loss,
            y_hat,
            q_hat,
            optax.global_norm(grads),
            alive.mean(),
        )

    alive = jnp.ones((bs, 1), dtype=jnp.bool_)
    (_, model, opt, y, z, alive, _), (losses, y_hats, q_hats, norms, props_alive) = (
        jax.lax.scan(
            sup_step,
            (1, model, opt, y, z, alive, rngs),
            None,
            length=cfg.N_sup,
        )
    )
    new_ema_model = optax.incremental_update(
        model, ema_model, step_size=1 - cfg.ema_beta
    )

    return (
        model,
        opt,
        new_ema_model,
        {
            "train/loss": losses[-1],
            "train/prop_alive": props_alive[-1],
            "train/grad_norm": norms[-1],
            "train/logit_mean": jnp.abs(y_hats[-1]).mean(),
            "train/logit_max": jnp.max(y_hats[-1]),
            "train/x_prenorm_scale": model.net.x_norm.scale.mean(),
            "train/y_prenorm_scale": model.net.y_norm.scale.mean(),
            "train/z_prenorm_scale": model.net.z_norm.scale.mean(),
            **pred_metrics(y_hats.argmax(axis=-1), y_true, prefix="train"),
        },
    )


@nnx.jit(static_argnames=("cfg",))
def eval_step(model, batch, cfg, rngs):
    model.eval()
    x_input, y_true = batch["inputs"], batch["labels"]
    keys = jax.random.split(rngs(), cfg.test_k)

    def one_pred(key):
        run_rngs = nnx.Rngs(key)
        y_hats, q_hats, _ = model.predict(
            x_input,
            N_sup=cfg.N_sup_test,
            n=cfg.n,
            T=cfg.T,
            rngs=run_rngs,
        )
        return y_hats.argmax(axis=-1), q_hats

    train_N = cfg.N_sup if cfg.N_sup_test > cfg.N_sup else None
    k_preds, q_hats = jax.vmap(one_pred)(keys)

    preds = k_preds[0]
    metrics, cell_acc, solved_acc = pred_metrics(
        preds, y_true, prefix="eval", train_N=train_N, return_curves=True
    )

    if cfg.test_k > 1:
        conf_preds = rearrange(
            jnp.take_along_axis(k_preds, q_hats.argmax(axis=0, keepdims=True), axis=0),
            "1 n b l -> n b l",
        )
        mode_preds = jnp.argmax(
            jax.nn.one_hot(k_preds, cfg.vocab_size).sum(axis=0), axis=-1
        )
        conf_metrics, conf_cell_acc, conf_solved_acc = pred_metrics(
            conf_preds, y_true, prefix="eval_conf", train_N=train_N, return_curves=True
        )
        mode_metrics, mode_cell_acc, mode_solved_acc = pred_metrics(
            mode_preds, y_true, prefix="eval_mode", train_N=train_N, return_curves=True
        )
        metrics = {
            **metrics,
            **conf_metrics,
            **mode_metrics,
            "cell_acc_per_step_conf": conf_cell_acc,
            "solved_acc_per_step_conf": conf_solved_acc,
            "cell_acc_per_step_mode": mode_cell_acc,
            "solved_acc_per_step_mode": mode_solved_acc,
        }

    return {
        **metrics,
        "cell_acc_per_step": cell_acc,
        "solved_acc_per_step": solved_acc,
        "batch_size": x_input.shape[0],
    }


def evaluate_epoch(model, data_iter, cfg, rngs, mesh=None, log_curves=False):
    totals = defaultdict(float)
    all_curve_steps = defaultdict(list)
    total_weight = 0.0

    for batch in data_iter:
        if mesh is not None:
            batch = shard_batch(batch)
        metrics = eval_step(model, batch, cfg, rngs)
        bs = float(metrics.pop("batch_size"))

        for k in [k for k in metrics if "_per_step" in k]:
            if log_curves:
                all_curve_steps[k].append(metrics.pop(k) * bs)
            else:
                metrics.pop(k)

        for k, v in metrics.items():
            totals[k] += v * bs
        total_weight += bs

    results = {k: v / total_weight for k, v in totals.items()}

    if log_curves:
        for k, arrays in all_curve_steps.items():
            results[f"_curve_{k.replace('_per_step', '')}"] = sum(arrays) / total_weight

    return results


@nnx.jit(static_argnames=("cfg",))
def asymptotic_alignment_score(model, batch, cfg, rngs):
    """arxiv:2211.09961"""

    def cos_sim(a, b):
        a, b = rearrange(a, "b ... -> b (...)"), rearrange(b, "b ... -> b (...)")
        return (a * b).sum(-1) / (
            jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1)
        ).clip(min=1e-8)

    model.eval()
    y_hats1, _, (y1, z1) = model.predict(
        batch["inputs"], N_sup=cfg.N_sup, n=cfg.n, T=cfg.T, rngs=rngs
    )

    y1_s, z1_s = jnp.roll(y1, shift=1, axis=0), jnp.roll(z1, shift=1, axis=0)

    y_hats2, _, (y2, z2) = model.predict(
        batch["inputs"], y=y1_s, z=z1_s, N_sup=cfg.N_sup, n=cfg.n, T=cfg.T, rngs=rngs
    )

    pred_match = (y_hats1[-1].argmax(axis=-1) == y_hats2[-1].argmax(axis=-1)).mean()

    return {
        "asymp_align/pred_match": pred_match,
        "asymp_align/y_cos_sim": cos_sim(y1, y2).mean(),
        "asymp_align/z_cos_sim": cos_sim(z1, z2).mean(),
    }


def model_factory(cfg, param_dtype, compute_dtype, rngs):
    Linear = partial(
        nnx.Linear, dtype=compute_dtype, param_dtype=param_dtype, rngs=rngs
    )

    model = TRM(
        net=Net(
            cfg.seq_len,
            cfg.h_dim,
            expansion=cfg.mlp_factor,
            n_layers=cfg.n_layers,
            linear=Linear,
            rngs=rngs,
        ),
        output_head=Linear(cfg.h_dim, cfg.vocab_size),
        Q_head=nnx.Sequential(
            partial(reduce, pattern="b l h -> b h", reduction="mean"),
            Linear(cfg.h_dim, 1),
        ),
        input_embedding=nnx.Embed(
            cfg.vocab_size, cfg.h_dim, param_dtype=param_dtype, rngs=rngs
        ),
        init_y=InitState(cfg.init_state, cfg.h_dim, rngs=rngs),
        init_z=InitState(cfg.init_state, cfg.h_dim, rngs=rngs),
    )

    # TODO get rid of this? test if actually helps
    decay_mask = nnx.state(model, nnx.Param).map(
        lambda path, p: (p.ndim >= 2) and ("embedding" not in path)
    )

    return model, decay_mask


@dataclass(frozen=True)
class cfg:
    dataset: str = "emiliocantuc/sudoku-extreme-1k-aug-1000"
    seq_len: int = 81
    vocab_size: int = 10

    n_layers: int = 2
    h_dim: int = 512
    mlp_factor: int = 4
    init_state: str = "static"

    N_sup: int = 16
    n: int = 6
    T: int = 3
    rand_n: bool = False  # TODO
    rand_T: bool = False

    halt_loss_weight: float = 0.5
    halt_exploration_prob: float = 0.1
    max_corruption_std: float = 0.0
    corruption_clean_prop: float = 0.5
    stay_on_policy: bool = False

    batch_size: int = 768
    lr: float = 1e-4
    lr_warmup_steps: int = 2000 // 16
    weight_decay: float = 1.0
    ema_beta: float = 0.999**16
    half_precision: bool = False
    steps: int = 15_000
    val_every: int = 500
    val_size: int = 2048

    test_only: bool = False
    skip_test: bool = False
    test_size: int | None = None
    N_sup_test: int = 16 * 4
    test_k: int = 1

    seed: int = None
    data_seed: int = 42
    workdir: str = None
    logdir: str = None
    checkpoint_every: int = 500
    max_checkpoints: int = 1
    use_parallel: bool = True


if __name__ == "__main__":
    cfg = simple_parsing.parse(cfg)
    tpu = jax.default_backend() == "tpu"
    param_dtype = jnp.float32
    compute_dtype = (
        jnp.bfloat16 if tpu and cfg.half_precision else jnp.float32
    )  # TODO test if ever stable, else delete

    seed = cfg.seed or random.randint(0, 2**32 - 1)
    if cfg.seed is not None:  # only when seed explicitly provided
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + " --xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
        )
    rngs = nnx.Rngs(seed)

    num_devices = jax.device_count()
    mesh = (
        jax.make_mesh((num_devices,), ("data",))
        if cfg.use_parallel and num_devices > 1
        else None
    )
    nnx.use_eager_sharding(mesh is not None)
    logging.info(f"Using mesh: {mesh}")

    ds = load_dataset(cfg.dataset)
    val_test = ds["test"].train_test_split(train_size=cfg.val_size, seed=cfg.data_seed)
    train_ds, val_ds, test_ds = ds["train"], val_test["train"], val_test["test"]
    if cfg.test_size is not None:
        test_ds = test_ds.shuffle(seed=cfg.data_seed).select(range(cfg.test_size))

    train_loader = Loader(train_ds, batch_size=cfg.batch_size, shuffle_seed=seed)
    val_loader = Loader(val_ds, batch_size=cfg.batch_size, epochs=1)
    test_loader = Loader(test_ds, batch_size=cfg.batch_size, epochs=1)

    with jax.set_mesh(mesh) if mesh is not None else contextlib.nullcontext():
        model, decay_mask = model_factory(cfg, param_dtype, compute_dtype, rngs)
        n_params = sum(
            jax.tree.map(jnp.size, jax.tree.leaves(nnx.state(model, nnx.Param)))
        )
        logging.info(f"No. of parameters: {n_params}")

        lr_schedule = optax.warmup_constant_schedule(
            init_value=0.0, peak_value=cfg.lr, warmup_steps=cfg.lr_warmup_steps
        )

        opt = nnx.Optimizer(
            model=model,
            tx=optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(
                    learning_rate=lr_schedule,
                    b1=0.9,
                    b2=0.95,
                    eps=1e-4 if cfg.half_precision else 1e-8,
                    weight_decay=cfg.weight_decay,
                    mask=decay_mask,
                ),
            ),
            wrt=nnx.Param,
        )

        ema_model = nnx.clone(model)

        checkpoint_manager = None
        if cfg.workdir is not None and (cfg.max_checkpoints > 0 or cfg.test_only):
            checkpoint_manager = ocp.CheckpointManager(
                cfg.workdir
                if cfg.workdir.startswith("gs://")
                else os.path.abspath(cfg.workdir),
                options=ocp.CheckpointManagerOptions(
                    best_mode="max",
                    best_fn=lambda m: m["eval/solved_acc"],
                    max_to_keep=cfg.max_checkpoints,
                ),
            )

        writer = metric_writers.create_default_writer(
            cfg.logdir or cfg.workdir, just_logging=jax.process_index() > 0
        )
        writer.write_hparams(vars(cfg))
        writer.write_scalars(0, {"hparams/n_params": n_params})

        restore_items = (
            ("ema_model",) if cfg.test_only else ("model", "opt", "ema_model")
        )
        start_step = restore_checkpoint(
            checkpoint_manager, model, opt, ema_model, items=restore_items
        )

        def _run_test():
            logging.info("Testing ...")
            test_metrics = evaluate_epoch(
                ema_model, test_loader, cfg, rngs, mesh, log_curves=True
            )

            curves = {k: test_metrics.pop(k) for k in list(test_metrics) if k.startswith("_curve_")}

            test_metrics = {
                k.replace("eval/", "test/"): v for k, v in test_metrics.items()
            }
            writer.write_scalars(0, test_metrics)

            for curve_key, curve in curves.items():
                name = curve_key.removeprefix("_curve_")
                for i, v in enumerate(curve):
                    writer.write_scalars(i + 1, {f"test_curve/{name}": float(v)})
            logging.info(f"Test metrics: {test_metrics}")

        if cfg.test_only and start_step <= 1:
            logging.error("No checkpoint found for test_only run.")
            exit(1)
        elif start_step > cfg.steps:
            logging.info(
                f"Loaded step {start_step - 1} already exceeds total {cfg.steps}."
            )
            if not cfg.skip_test:
                _run_test()
            exit(0)
        elif cfg.test_only:
            _run_test()
            exit(0)

        last_eval_metrics = {"metrics": None}

        def _run_val(step, t):
            m = evaluate_epoch(ema_model, val_loader, cfg, rngs, mesh)
            last_eval_metrics["metrics"] = m
            writer.write_scalars(step, m)

        def _save_checkpoint(step, t):
            if last_eval_metrics["metrics"] is None:
                return
            save_checkpoint(
                checkpoint_manager,
                step,
                model,
                opt,
                ema_model,
                last_eval_metrics["metrics"],
            )

        hooks = [
            periodic_actions.ReportProgress(num_train_steps=cfg.steps, writer=writer),
            periodic_actions.PeriodicCallback(
                every_steps=cfg.val_every,
                on_steps=[cfg.steps],
                callback_fn=_run_val,
            ),
            periodic_actions.PeriodicCallback(
                every_steps=cfg.val_every,
                callback_fn=lambda step, t: writer.write_scalars(
                    step,
                    calc_metric_over_batches(
                        lambda batch: asymptotic_alignment_score(
                            ema_model, batch, cfg, rngs
                        ),
                        iter(val_loader),
                        mesh,
                    ),
                ),
            ),
        ]
        if checkpoint_manager is not None:
            hooks.append(
                periodic_actions.PeriodicCallback(
                    every_steps=cfg.checkpoint_every,
                    on_steps=[cfg.steps],
                    callback_fn=_save_checkpoint,
                )
            )
        if cfg.workdir is not None and jax.process_index() == 0:
            hooks.append(
                periodic_actions.Profile(num_profile_steps=5, logdir=cfg.workdir)
            )

        with metric_writers.ensure_flushes(writer):
            for step, batch in enumerate(train_loader, start=start_step):
                if mesh is not None:
                    batch = shard_batch(batch)
                model, opt, ema_model, train_metrics = train_step(
                    model, ema_model, opt, batch, cfg, rngs
                )
                train_metrics["train/lr"] = lr_schedule(step)
                writer.write_scalars(step, train_metrics)

                for h in hooks:
                    h(step)

                if step >= cfg.steps:
                    break

            if not cfg.skip_test:
                _run_test()
            if checkpoint_manager is not None:
                checkpoint_manager.wait_until_finished()
                checkpoint_manager.close()  # important: joins any internal workers




