# trm-jax

Working small model run:

```bash
uv run --with jax\[tpu\] main.py \
--batch_size 128 --h_dim 256  --weight_decay 1e-3 --lr 1e-3 \
--steps 25_000 --init_state static --half_precision
```

Trying big run in a v4-8 with:
```bash
uv run --with jax\[tpu\] main.py \
--batch_size 768 --h_dim 512 --weight_decay 1.0 --lr 1e-4 \
--N_supervision 8 --steps 25_000 --init_state static
```