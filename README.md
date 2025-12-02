# trm-jax

```bash
uv run --with jax\[tpu\] main.py \
--batch_size 512 --h_dim 512 \
--N_supervision 8 --weight_decay 1e-3 --lr 1e-3 \
--steps 10_000 --workdir logs/tmp2
```