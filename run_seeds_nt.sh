#!/bin/bash
cd /Users/schalk/Desktop/QUT/EGH400/SliQ

SEEDS=(1 2 3 4 5)
BACKENDS=("kingston" "fez" "kingston")

echo "Starting backend sweep..."
for backend in "${BACKENDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "=== $(date): NT backend=${backend} seed=${seed} ==="
        echo "$(date): NT backend=${backend} seed=${seed} START" >> run_log.txt
        python main.py \
            --message "NT seed${seed} ${backend} keyword:1neighbour3class0.2" \
            --noise_train True \
            --seed ${seed} \
            --learning_rate 0.1 \
            --backend ${backend} \
            --staged 50 \
            --ramp 50 \
            --results_dir "Results/NT_v2/${backend}/seed${seed}"
        echo "$(date): NT backend=${backend} seed=${seed} DONE" >> run_log.txt
    done
done