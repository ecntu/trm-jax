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
    for variant in baseline random_init rand_T rand_N_sup rand_T_and_N_sup; do

        INIT_ARGS=""
        [[ "$variant" != "baseline" ]] && INIT_ARGS="--init_state random"

        # Determine width sweeps per variant (T widths limited to 1,2 since T=3)
        if [[ "$variant" == "rand_T" ]]; then
            T_WIDTHS=(1 2)
            N_WIDTHS=(0)
        elif [[ "$variant" == "rand_N_sup" ]]; then
            T_WIDTHS=(0)
            N_WIDTHS=(2 4 8)
        elif [[ "$variant" == "rand_T_and_N_sup" ]]; then
            T_WIDTHS=(1 2)
            N_WIDTHS=(2 4 8)
        else
            T_WIDTHS=(0)
            N_WIDTHS=(0)
        fi

        for t_width in "${T_WIDTHS[@]}"; do
        for n_width in "${N_WIDTHS[@]}"; do

            VARIANT_ARGS=""
            WIDTH_ARGS=""
            WORKDIR_SUFFIX=""
            if [[ "$variant" == "rand_T" ]]; then
                VARIANT_ARGS="--rand_T"
                WIDTH_ARGS="--rand_T_width $t_width"
                WORKDIR_SUFFIX="/wT${t_width}"
            elif [[ "$variant" == "rand_N_sup" ]]; then
                VARIANT_ARGS="--rand_N_sup"
                WIDTH_ARGS="--rand_N_sup_width $n_width"
                WORKDIR_SUFFIX="/wN${n_width}"
            elif [[ "$variant" == "rand_T_and_N_sup" ]]; then
                VARIANT_ARGS="--rand_T --rand_N_sup"
                WIDTH_ARGS="--rand_T_width $t_width --rand_N_sup_width $n_width"
                WORKDIR_SUFFIX="/wT${t_width}_wN${n_width}"
            fi

            WORKDIR="${LOG_PREFIX}/${variant}${WORKDIR_SUFFIX}/seed${seed}"
            [[ -f "${WORKDIR}/TESTED" ]] && { echo "Skipping (already tested): ${WORKDIR}"; continue; }

            echo "Running with variant=${variant}, t_width=${t_width}, n_width=${n_width}, seed=${seed}, size=${SIZE}"
            uv run --with jax[tpu] main.py \
                $VARIANT_ARGS $WIDTH_ARGS $INIT_ARGS --seed $seed \
                --steps $STEPS --test_size 10_000 --N_sup_test $N_SUP_TEST \
                --max_checkpoints 0 --workdir "${WORKDIR}" $EXTRA_ARGS

        done
        done
    done
done