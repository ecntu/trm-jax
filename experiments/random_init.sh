#!/usr/bin/env bash

# Test out init_state buffer vs. random
# And, for random inits,
# Note: hardcoded test_k=3 (for val curves) during training
# Sizes: "big" (default), "small" (--h_dim 256 --batch_size 128 --N_sup 8 --steps 30_000)

SIZE=${1:-big}

if [[ "$SIZE" == "small" ]]; then
    EXTRA_ARGS="--h_dim 256 --batch_size 128 --N_sup 8"
    STEPS=30_000
    N_SUP_TEST=512
    LOG_PREFIX="logs/random-init-small"
else
    EXTRA_ARGS=""
    STEPS=10_000
    N_SUP_TEST=768
    LOG_PREFIX="logs/random-init"
fi

for seed in {1..5}; do
    for init_state in random static; do

        WORKDIR="${LOG_PREFIX}/init-state${init_state}_seed${seed}"
        [[ -f "${WORKDIR}/TESTED" ]] && { echo "Skipping (already tested): ${WORKDIR}"; continue; }

        if [[ ! -f "${WORKDIR}/TRAINED" ]]; then
            echo "Running with init_state=${init_state}, seed=${seed}, size=${SIZE}"
            uv run --with jax[tpu] main.py \
                --init_state $init_state --test_k 3 \
                --seed $seed --skip_test --steps $STEPS \
                --max_checkpoints 1 --workdir $WORKDIR $EXTRA_ARGS
        fi

        ks=(1)
        [[ "$init_state" == "random" ]] && ks=(1 3 5 7)
        z_noise_stds=(0.0 0.1 0.5 1.0)
        for k in "${ks[@]}"; do
            for z_noise_std in "${z_noise_stds[@]}"; do
                [[ "$k" == "1" && "$z_noise_std" != "0.0" ]] && continue
                echo "Testing with init_state=${init_state}, seed=${seed}, k=${k}, z_noise_std=${z_noise_std}, size=${SIZE}"
                uv run --with jax[tpu] main.py \
                    --init_state $init_state --test_k $k --seed $seed \
                    --z_noise_std $z_noise_std \
                    --test_only --test_size 10_000 --N_sup_test $N_SUP_TEST \
                    --workdir $WORKDIR \
                    --logdir ${WORKDIR}/test_k${k}_znoise${z_noise_std//./_} $EXTRA_ARGS
            done
        done
        touch "${WORKDIR}/TESTED"

    done
done
