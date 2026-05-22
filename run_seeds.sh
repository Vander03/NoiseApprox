#!/bin/bash
cd /Users/schalk/Desktop/QUT/EGH400/SliQ

SEEDS=(1 2 3 4 5)

echo "Starting runs..."
for seed in "${SEEDS[@]}"; do
    # for dr in "${dropoff[@]}"; do
        echo "                  === NT | seed=${seed} ==="
        echo "$(date): NT lr=${lr} seed=${seed} START" >> run_log.txt
        python main.py \
            --message "NT seed${seed} seeded sample loading knn stale shifts from noiseless model keyword:knnperp" \
            --noise_train True \
            --seed ${seed} \
            --learning_rate 0.1 \
            --ramp 50
        echo "$(date): NT lr=${lr} seed=${seed} DONE" >> run_log.txt

    # done
done
for seed in "${SEEDS[@]}"; do
    echo "                  === $(date): non-NT lr=${lr} seed=${seed} ==="
    echo "$(date): non-NT lr=${lr} seed=${seed} START" >> run_log.txt
    python main.py \
        --message "non-NT seed${seed} seeded sample loading keyword:knnperp" \
        --noise_train False \
        --seed ${seed} \
        --learning_rate 0.1 \
        --ramp 50
    echo "$(date): NT lr=${lr} seed=${seed}" >> run_log.txt
done



echo "$(date): All runs complete" >> run_log.txt
echo "All runs complete"