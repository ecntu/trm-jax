# trm-jax

Working small model run:

```bash
uv run --with jax\[tpu\] main.py \
--batch_size 128 --h_dim 256  --weight_decay 1e-3 --lr 1e-3 \
--steps 25_000 --init_state static --half_precision
```