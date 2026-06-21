# NoiseApprox

Classical approximation of quantum hardware noise in embedding space for noise-aware contrastive training of quantum neural networks on NISQ-era IBM quantum hardware.

This project proposes a KNN shift bank constructed from historical IBM calibration profiles that approximates how each region of embedding space shifts under real hardware noise. The shift bank is used as a 74x faster alternative to noisy circuit evaluation during training, and as a pre-training diagnostic tool that predicts whether noise-aware training will benefit a given backend before any training is performed.

## Method

Quantum hardware noise induces position-dependent shifts in the embedding space of quantum neural networks. Rather than evaluating noisy circuits at every training step, the shift bank characterises these shifts offline from historical IBM calibration profiles.

**Shift bank construction (offline, pre-training):**
1. Run 1000 training set embeddings through a frozen noise-naive model using noiseless weights
2. Partition into 20 KMeans clusters
3. Select 10 representative samples per cluster (200 total)
4. Evaluate each representative under 5 randomly sampled historical IBM calibration profiles via Qiskit AerSimulator density matrix simulation
5. Store the mean shift vector per representative as (embedding, shift) pairs in `knn_embs.npy` and `knn_shifts.npy`

The shift bank is shared across seeds for a given backend — only the first seed builds it, all subsequent seeds load from `Results/NT_v3/{backend}/shift_bank/`.

**Noise-aware training:**

During training, each anchor embedding is perturbed by the shift vector of its nearest neighbour in the shift bank, constructing a noise-shifted positive pair for the triplet loss:

```
eP = eA + delta_eA
L = max(0, ||eA - eP||^2 - ||eA - eN||^2 + m)
```

This encodes hardware noise invariance into the learned representations without any noisy circuit evaluations during training.

**Evaluation:**

GMM clustering accuracy on held-out test embeddings evaluated under 5 holdout calibration profiles not seen during training or shift bank construction.

## Requirements

```bash
pip install -r requirements.txt
```

Noiseless weights must be present at `noiseless_trained_6_fashion.npy` in the project root. These are the frozen weights used for shift bank construction and are not trained during the noise-aware runs. Future work includes the dynamic refitting of the shift bank at key points during training, or using optimal transport to model the shifts.

IBM calibration profiles must be available under `noise/calibration_data/`. Profiles are loaded via `noise/noise.py` which filters by backend name and selects the 20 most recent non-holdout profiles for training and the 5 subsequent dates as holdout.

## Running training

**Single run:**

```bash
python main.py \
    --message "NT seed1 kingston" \
    --noise_train True \
    --seed 1 \
    --learning_rate 0.1 \
    --backend kingston \
    --results_dir "Results/NT_v3/kingston/seed1"
```

**Key arguments:**

| Argument | Description | Default |
|---|---|---|
| `--message` | Description of the run (required) | - |
| `--noise_train` | Enable noise-aware training | `True` |
| `--seed` | Random seed | `1` |
| `--learning_rate` | Adam learning rate | `0.1` |
| `--backend` | IBM backend name: `kingston`, `fez`, `marrakesh` | `fez` |
| `--results_dir` | Output directory for this run | auto-generated |

**Noise-naive baseline** (set `--noise_train False`):

```bash
python main.py \
    --message "non-NT seed1 kingston" \
    --noise_train False \
    --seed 1 \
    --backend kingston \
    --results_dir "Results/non_NT/kingston/seed1"
```

**Full sweep across seeds and backends** (see shell scripts):

```bash
bash run_seeds_nt.sh   # noise-aware training
bash non_nt.sh         # noise-naive baseline
```

## Results directory structure

```
Results/
  NT_v3/
    kingston/
      shift_bank/           # shared shift bank for all seeds
        knn_embs.npy        # (200, 6) representative embeddings
        knn_shifts.npy      # (200, 6) mean shift vectors
      seed1/
        run_info.json       # full config and noisy eval results
        weights.npy         # weight history across epochs
        best_weights.npy    # weights at best training loss
        loss_history.npy    # training loss per epoch
        knn_embs.npy        # copy of shift bank (for get_results compatibility)
        knn_shifts.npy      # copy of shift bank
        loss.png            # training loss curve
        embeddings_train.png
```

## Evaluating approximation quality

`get_results.py` runs post-hoc analysis on a trained (or noiseless) model, generating shift bank diagnostic plots.

```bash
python get_results.py \
    --path Results/NT_v3/kingston/seed1
```

By default this uses the frozen noiseless weights for shift bank evaluation. To evaluate how well shift bank predictions hold up on trained weights:

```bash
python get_results.py \
    --path Results/NT_v3/kingston/seed1 \
    --trained_weights Results/NT_v3/kingston/seed1/best_weights.npy
```

**What it does:**
1. Loads the shift bank from `{backend_dir}/shift_bank/`
2. Generates 500 test set embeddings using the supplied weights
3. Predicts shifts via KNN lookup into the real shift bank
4. Measures ground-truth shifts by running 5 randomly sampled training profiles through the noisy circuit
5. Computes cosine similarity between predicted and ground-truth shifts per test sample
6. Generates all diagnostic plots and saves to `{backend_dir}/plots/` or `{backend_dir}/trained_validation/`

## Plots

**`loss.png`** — training loss curve with smoothed total loss and optional clean/noisy component fill.

**`embeddings_train.png`** — UMAP of training embeddings coloured by class label, showing learned cluster separation.

**`shift_heatmap_comparison.png`** — three-panel heatmap showing ground truth shifts (simulated noisy circuit), KNN-predicted shifts, and per-dimension prediction error for 200 test samples sorted by cluster. Rows are test embeddings, columns are the 6 embedding dimensions.

**`shift_cos_sim_dist.png`** — two-panel figure showing the distribution of cosine similarities between predicted and ground-truth shifts (histogram + KDE), and a compass plot projecting predicted and ground-truth shift directions onto a 2D plane. The reference axis (black arrow, 0 degrees) is the mean ground-truth shift direction. The red arrow shows the mean holdout shift direction, revealing temporal stability or drift.

**`shift_field_cosine_overlay.png`** — two-panel figure. Left: UMAP of test embeddings coloured by cosine similarity (green = high approximation quality, red = low). Right: scatter of local directional variance vs cosine similarity with a linear fit and Pearson r. A strong negative correlation confirms that approximation quality degrades in regions where neighbouring embeddings experience inconsistent shift directions.

## Parsing and printing results

```bash
python parse_batch.py   # parse all run_info.json files into a summary
python print_runs.py    # print a formatted table of results
```

## Key files

| File | Purpose |
|---|---|
| `model.py` | Triplet class, training loop, shift bank construction, noisy evaluation |
| `main.py` | Entry point, config, training orchestration |
| `get_results.py` | Post-hoc shift bank analysis and diagnostic plots |
| `visualiser.py` | All plotting functions |
| `triplet_generator.py` | PCA preprocessing, triplet construction, train/test split |
| `noise/noise.py` | IBM calibration profile loading and filtering |
| `parse_batch.py` | Batch result parsing from run_info.json files |
| `print_runs.py` | Formatted result table printing |
| `run_seeds_nt.sh` | Shell script for full NT sweep |
| `non_nt.sh` | Shell script for noise-naive baseline sweep |

## Architecture

- 6 qubits, 6 layers
- AmplitudeEmbedding with 64-dimensional PCA preprocessing
- Ring CNOT entanglement
- PauliZ expectation value measurements producing a 6-dimensional embedding in [-1, 1]
- Training: PennyLane `lightning.qubit` statevector
- Noisy evaluation: Qiskit AerSimulator density matrix, 1000 shots
- Optimiser: Adam, learning rate 0.1, margin 0.02, batch size 128, 50 epochs

## Backends evaluated

| Backend | Noise characteristic | NT benefit |
|---|---|---|
| IBM Kingston | High magnitude, spatially coherent, temporally stable | Strong (+8.1%) |
| IBM Marrakesh | Low magnitude, temporally stable | None (diluted signal) |
| IBM Fez | High magnitude, spatially incoherent, temporally unstable | Harmful |