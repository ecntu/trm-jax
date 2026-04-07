#!/usr/bin/env bash

# Test out on-policy vs. off-policy training

for seed in {1..5}; do
    for on_policy in 0 1; do

        on_policy_flag=""
        [[ "$on_policy" == "1" ]] && on_policy_flag="--stay_on_policy"

        echo "Running with on_policy=${on_policy}, seed=${seed}"
        uv run --with jax[tpu] main.py \
            $on_policy_flag --seed $seed \
            --steps 10_000 --test_size 10_000 --N_sup_test 768 \
            --workdir logs/on-policy/onpolicy${on_policy}_seed${seed} \
            --max_checkpoints 0
    done
done
