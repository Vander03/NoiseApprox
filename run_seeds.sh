#!/bin/bash
cd /Users/schalk/Desktop/QUT/EGH400/SliQ

LR=(0.1)
dropoff=(10 25 50 75 100)
SEEDS=(3 1 4 2 5 6 7)

echo "Starting runs..."
for seed in "${SEEDS[@]}"; do
    # for dr in "${dropoff[@]}"; do
        echo "                  === NT | seed=${seed} ==="
        echo "$(date): NT lr=${lr} seed=${seed} START" >> run_log.txt
        python main.py \
            --message "NT gaussian lr${lr} seed${seed} pca32 3class adam fixed loss and gmm keyword:weightinit" \
            --noise_train True \
            --seed ${seed} \
            --learning_rate 0.1 \
            --ramp 50
        echo "$(date): NT lr=${lr} seed=${seed} DONE" >> run_log.txt
        echo "                  === $(date): non-NT lr=${lr} seed=${seed} ==="
        echo "$(date): non-NT lr=${lr} seed=${seed} START" >> run_log.txt
        python main.py \
            --message "non-NT lr${lr} seed${seed} pca32 3class adam fixed loss and gmm keyword:weightinit" \
            --noise_train False \
            --seed ${seed} \
            --learning_rate 0.1 \
            --ramp 50
        echo "$(date): NT lr=${lr} seed=${seed}" >> run_log.txt
    # done
done

echo "$(date): All runs complete" >> run_log.txt
echo "All runs complete"