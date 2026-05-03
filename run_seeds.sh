#!/bin/bash
cd /Users/schalk/Desktop/QUT/EGH400/SliQ

lr=(0.1 0.2 0.4 0.8)
SEEDS=(1 2 3 4 5)

echo "Starting runs..."
for seed in "${SEEDS[@]}"; do
    echo "=== non-NT | seed=${seed} ==="
    echo "$(date): non-NT lr=${lr} seed=${seed} START" >> run_log.txt
    python main.py \
        --message "non-NT lr${lr} seed${seed} pca32 3class adam fixed loss and gmm 200epochLR" \
        --noise_train False \
        --seed ${seed}
    echo "$(date): non-NT lr=${lr} seed=${seed} DONE" >> run_log.txt
    echo "=== NT | seed=${seed} ==="
    echo "$(date): NT lr=${lr} seed=${seed} START" >> run_log.txt
    python main.py \
        --message "NT gaussian lr${lr} seed${seed} pca32 3class adam fixed loss and gmm 200epochLR" \
        --noise_train True \
        --seed ${seed}
    echo "$(date): NT lr=${lr} seed=${seed} DONE" >> run_log.txt
done

echo "$(date): All runs complete" >> run_log.txt
echo "All runs complete"