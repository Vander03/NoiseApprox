#!/bin/bash
cd /Users/schalk/Desktop/QUT/EGH400/SliQ

SEEDS=(1 2 3 4 5)
BACKENDS=("kingston" "fez" "marrakesh")

echo "Starting backend sweep..."
for backend in "${BACKENDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "=== $(date): non-NT backend=${backend} seed=${seed} ==="
        python -m noise.noise_test \
            --path "Results/noiseless/seed${seed}" \
            --backend ${backend}
    done
done