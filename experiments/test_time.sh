#!/usr/bin/env bash

# Test out test-time inference methods
# Large model only.

STEPS=10_000
N_SUP_TEST=768
LOG_PREFIX="logs/test-time"

for seed in {1..3}; do
    WORKDIR="${LOG_PREFIX}/seed${seed}"
    [[ -f "${WORKDIR}/TESTED" ]] && { echo "Skipping (already tested): ${WORKDIR}"; continue; }

    if [[ ! -f "${WORKDIR}/TRAINED" ]]; then
        echo "Running with seed=${seed}, size=large"
        uv run --with jax[tpu] main.py \
            --seed "$seed" --skip_test --steps "$STEPS" \
            --max_checkpoints 1 --workdir "$WORKDIR"
    fi

    ks=(3 5 7)
    z_noise_stds=(0.0 0.1 0.5 1.0)
    for k in "${ks[@]}"; do
        for z_noise_std in "${z_noise_stds[@]}"; do
            echo "Testing with seed=${seed}, k=${k}, z_noise_std=${z_noise_std}, size=large"
            uv run --with jax[tpu] main.py \
                --test_k "$k" --seed "$seed" \
                --z_noise_std "$z_noise_std" \
                --test_only --test_size 10_000 --N_sup_test "$N_SUP_TEST" \
                --workdir "$WORKDIR" \
                --logdir "${WORKDIR}/test_k${k}_znoise${z_noise_std//./_}"
        done
    done
    touch "${WORKDIR}/TESTED"
done
