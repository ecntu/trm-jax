#!/usr/bin/env bash

# Test out init_state buffer vs. random
# And, for random inits,
# Note: hardcoded test_k=3 (for val curves) during training

for seed in {1..3}; do
    for init_state in random static; do

        echo "Running with init_state=${init_state}, seed=${seed}"
        uv run --with jax[tpu] main.py \
            --init_state $init_state --test_k 3 \
            --seed $seed --skip_test \
            --max_checkpoints 1 --workdir logs/random-init/init-state${init_state}_seed${seed}

        ks=(1)
        [[ "$init_state" == "random" ]] && ks=(1 3 5 7)
        for k in "${ks[@]}"; do
            echo "Testing with init_state=${init_state}, seed=${seed}, k=${k}"
            uv run --with jax[tpu] main.py \
                --init_state $init_state --test_k $k --seed $seed \
                --test_only --test_size 76800 --N_sup_test 768 \
                --workdir logs/random-init/init-state${init_state}_seed${seed} \
                --logdir logs/random-init/init-state${init_state}_seed${seed}/test_k${k}
        done

    done
done
