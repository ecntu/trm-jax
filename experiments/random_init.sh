#!/usr/bin/env bash

# Test out init_state buffer vs. random
# And, for random inits,
# Note: hardcoded test_k=3 (for val curves) during training

for seed in {1..3}; do
    for init_state in random static; do

        echo "Running with init_state=${init_state}, seed=${seed}"
        # uv run --with jax[tpu] main.py \
        #     --h_dim 256 --batch_size 128 --N_sup 8 --steps 30_000 \
        #     --init_state $init_state --test_k 3 \
        #     --seed $seed --skip_test \
        #     --max_checkpoints 1 --workdir logs/random-init-small/init-state${init_state}_seed${seed}

        echo "Testing with init_state=${init_state}, seed=${seed}, k=1"
        uv run --with jax[tpu] main.py \
            --h_dim 256 --batch_size 128 \
            --init_state $init_state --test_k 1 --seed $seed \
            --test_only --test_size 12800 \
            --workdir logs/random-init-small/init-state${init_state}_seed${seed} \
            --logdir logs/random-init-small/init-state${init_state}_seed${seed}/test_k1

        if [[ "$init_state" == "random" ]]; then
          for k in 3 5 7; do
              echo "Testing with init_state=random, seed=${seed}, k=${k}"
              uv run --with jax[tpu] main.py \
                  --h_dim 256 --batch_size 128 \
                  --init_state random --test_k $k --seed $seed \
                  --test_only --test_size 12800 \
                  --workdir logs/random-init-small/init-state${init_state}_seed${seed} \
                  --logdir logs/random-init-small/init-state${init_state}_seed${seed}/test_k${k}
          done
        fi

    done
done
