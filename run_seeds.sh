#!/bin/bash
cd /Users/schalk/Desktop/QUT/EGH400/SliQ

SEEDS=(1 2 3 4 5)
BACKENDS=("kingston" "fez" "marrakesh")

echo "Starting full NT sweep: $(date)" >> run_log.txt



echo "NT training complete. Starting noiseless holdout evaluation: $(date)" >> run_log.txt

for backend in "${BACKENDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "=== $(date): noiseless eval backend=${backend} seed=${seed} ==="
        python -m noise.noise_test \
            --path "Results/noiseless/seed${seed}" \
            --backend ${backend}
    done
done

echo "All done: $(date)" >> run_log.txt