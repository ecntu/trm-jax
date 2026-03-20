#!/usr/bin/env bash

for seed in {1..3}; do
    for n in 6 4 2 1; do
        for T in 1 2 3 6 9; do
            echo "Running with T=${T}, n=${n}, seed=${seed}"
            uv run --with jax[tpu] main.py \
                --h_dim 256 --batch_size 128 --N_sup 8 --steps 30_000 \
                --T $T --n $n --seed $seed \
                --skip_test --max_checkpoints 0 --workdir logs/n-T-ablations/T${T}_n${n}_seed${seed}
        done
    done
done