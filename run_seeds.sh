#!/bin/bash
cd /Users/schalk/Desktop/QUT/EGH400/SliQ

LR=(0.1)
SEEDS=(1 4 2 3 5 6 7)

echo "Starting runs..."
for lr in "${LR[@]}"; do
    for seed in "${SEEDS[@]}"; do
            echo "=== NT | seed=${seed} ==="
        echo "$(date): NT lr=${lr} seed=${seed} START" >> run_log.txt
        python main.py \
            --message "NT gaussian lr${lr} seed${seed} pca32 3class adam fixed loss and gmm keyword:staged" \
            --noise_train True \
            --seed ${seed} \
            --learning_rate ${lr}
        echo "$(date): NT lr=${lr} seed=${seed} DONE" >> run_log.txt
        echo "=== $(date): non-NT lr=${lr} seed=${seed} ==="
        echo "$(date): non-NT lr=${lr} seed=${seed} START" >> run_log.txt
        python main.py \
            --message "non-NT lr${lr} seed${seed} pca32 3class adam fixed loss and gmm keyword:staged" \
            --noise_train False \
            --seed ${seed} \
            --learning_rate ${lr}
        echo "$(date): NT lr=${lr} seed=${seed}" >> run_log.txt
    done
done

echo "$(date): All runs complete" >> run_log.txt
echo "All runs complete"