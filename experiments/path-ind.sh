#!/usr/bin/env bash

# Test out diff ideas to increase path-independence
# Sizes: "big" (default), "small" (--h_dim 256 --batch_size 128 --N_sup 8 --steps 30_000)

SIZE=${1:-big}

if [[ "$SIZE" == "small" ]]; then
    EXTRA_ARGS="--h_dim 256 --batch_size 128 --N_sup 8"
    STEPS=30_000
    N_SUP_TEST=512
    LOG_PREFIX="logs/path-ind-small"
else
    EXTRA_ARGS=""
    STEPS=10_000
    N_SUP_TEST=768
    LOG_PREFIX="logs/path-ind"
fi

for seed in {1..5}; do
    for variant in baseline random_init rand_T halt_exploration_prob rand_N_sup warmup_T; do

        VARIANT_ARGS=""
        [[ "$variant" == "rand_T" ]] && VARIANT_ARGS="--rand_T"
        [[ "$variant" == "halt_exploration_prob" ]] && VARIANT_ARGS="--halt_exploration_prob 0.5"
        [[ "$variant" == "rand_N_sup" ]] && VARIANT_ARGS="--rand_N_sup"
        [[ "$variant" == "warmup_T" ]] && VARIANT_ARGS="--warmup_T"

        INIT_ARGS=""
        [[ "$variant" != "baseline" ]] && INIT_ARGS="--init_state random"

        echo "Running with variant=${variant}, seed=${seed}, size=${SIZE}"
        uv run --with jax[tpu] main.py \
            $VARIANT_ARGS $INIT_ARGS --seed $seed \
            --steps $STEPS --test_size 10_000 --N_sup_test $N_SUP_TEST \
            --max_checkpoints 0 --workdir ${LOG_PREFIX}/${variant}/seed${seed} $EXTRA_ARGS

    done
done