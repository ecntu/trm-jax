#!/usr/bin/env bash

# Compare online confidence mode (conf_online_k) against post-hoc conf/mode (test_k).
# Online: at each supervision step, branch into k noisy candidates and keep the most confident.
# Post-hoc: run k independent chains, aggregate predictions after the full N_sup chain.
# Sizes: "big" (default), "small" (--h_dim 256 --batch_size 128 --N_sup 8 --steps 30_000)

SIZE=${1:-big}

if [[ "$SIZE" == "small" ]]; then
    EXTRA_ARGS="--h_dim 256 --batch_size 128 --N_sup 8"
    STEPS=30_000
    N_SUP_TEST=512
    LOG_PREFIX="logs/conf-online-small"
else
    EXTRA_ARGS=""
    STEPS=10_000
    N_SUP_TEST=768
    LOG_PREFIX="logs/conf-online"
fi

for seed in {1..3}; do
    for init_state in random static; do

        WORKDIR="${LOG_PREFIX}/init${init_state}_seed${seed}"

        echo "Training: init_state=${init_state}, seed=${seed}"
        uv run main.py \
            --init_state $init_state \
            --seed $seed --skip_test --steps $STEPS \
            --max_checkpoints 1 --workdir $WORKDIR $EXTRA_ARGS

        for k in 1 3 5 7; do
            echo "Testing post-hoc: init_state=${init_state}, seed=${seed}, test_k=${k}"
            uv run main.py \
                --init_state $init_state --test_k $k \
                --seed $seed --test_only --test_size 5_000 --N_sup_test $N_SUP_TEST \
                --workdir $WORKDIR --logdir ${WORKDIR}/test_k${k} $EXTRA_ARGS

            if [[ $k -gt 1 ]]; then
                echo "Testing online: init_state=${init_state}, seed=${seed}, conf_online_k=${k}"
                uv run main.py \
                    --init_state $init_state --conf_online_k $k \
                    --seed $seed --test_only --test_size 5_000 --N_sup_test $N_SUP_TEST \
                    --workdir $WORKDIR --logdir ${WORKDIR}/conf_online_k${k} $EXTRA_ARGS
            fi
        done

    done
done
