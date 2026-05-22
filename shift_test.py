import sys
sys.path.append('..')
import numpy
import matplotlib.pyplot as plt
import os, json
from tqdm import tqdm
from qiskit_aer import AerSimulator
import pennylane as qml
from pennylane.templates import AmplitudeEmbedding
import triplet_generator
from collections import defaultdict

# ── config ──────────────────────────────────────────────────
RESULTS_ROOT = "Results"
KEYWORD      = "knnconsistency"
SS_SAMPLES   = [63, 550, 1755, 2633, 2653, 3444, 4518]
N_PROFILES   = 5
# ────────────────────────────────────────────────────────────

def build_clean_circuit(num_wires, embed_dims):
    sim = AerSimulator(method='density_matrix', seed_simulator=42)
    dev = qml.device('qiskit.aer', wires=num_wires, backend=sim)

    @qml.qnode(dev, shots=1000)
    def clean_circuit(weights, features):
        AmplitudeEmbedding(features=features.astype('float64'), wires=range(num_wires), normalize=True, pad_with=0)
        for W in weights:
            for i in range(num_wires):
                qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
            for wire in range(num_wires - 1):
                qml.CNOT(wires=[wire, wire + 1])
            qml.CNOT(wires=[num_wires - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(embed_dims)]
    return clean_circuit


def build_noisy_circuit(noise_model, num_wires, embed_dims):
    noise_sim = AerSimulator(noise_model=noise_model, method='density_matrix', seed_simulator=42)
    dev = qml.device('qiskit.aer', wires=num_wires, backend=noise_sim)

    @qml.qnode(dev, shots=5000)
    def noisy_circuit(weights, features):
        AmplitudeEmbedding(features=features.astype('float64'), wires=range(num_wires), normalize=True, pad_with=0)
        for W in weights:
            for i in range(num_wires):
                qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
            for wire in range(num_wires - 1):
                qml.CNOT(wires=[wire, wire + 1])
            qml.CNOT(wires=[num_wires - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(embed_dims)]
    return noisy_circuit


def measure_shifts_at_checkpoint(weights, triplets, clean_circuit, noisy_circuits):
    shifts = []
    for r in SS_SAMPLES:
        sample = numpy.array(triplets[r][0])
        clean_emb = numpy.array([float(z) for z in clean_circuit(weights, sample)])
        for circuit in noisy_circuits:
            noisy_emb = numpy.array([float(z) for z in circuit(weights, sample)])
            shifts.append(numpy.linalg.norm(noisy_emb - clean_emb))
    return numpy.mean(shifts)

_cached_profiles = None

def analyse_run(path, triplets):
    global _cached_profiles
    
    with open(os.path.join(path, "run_info.json")) as f:
        config = json.load(f)["config"]

    num_wires  = config["num_qubits"]
    embed_dims = config["embed_dims"]
    weight_history = numpy.load(os.path.join(path, "weights.npy"), allow_pickle=True)

    from noise.noise import noise as noise_helper
    nh = noise_helper(fake=False, hist_count=config["historic_load"])
    
    if _cached_profiles is None:
        _cached_profiles = nh.load_calibration_data(limit_backends=config.get("backend_name"))
    
    holdouts = nh.holdout_profiles
    train_profiles = [p for p in _cached_profiles if p["filename"] not in holdouts][:N_PROFILES]

    clean_circuit  = build_clean_circuit(num_wires, embed_dims)
    noisy_circuits = [build_noisy_circuit(p["noise_model"], num_wires, embed_dims) for p in train_profiles]

    norms  = []
    shifts = []
    for weights in tqdm(weight_history, desc=f"  checkpoints", leave=False):
        norms.append(numpy.linalg.norm(weights))
        shifts.append(measure_shifts_at_checkpoint(weights, triplets, clean_circuit, noisy_circuits))

    return numpy.array(norms), numpy.array(shifts)


def collect_runs():
    runs = defaultdict(dict)  # seed → {NT: path, non-NT: path}
    for date_dir in sorted(os.listdir(RESULTS_ROOT)):
        date_path = os.path.join(RESULTS_ROOT, date_dir)
        if not os.path.isdir(date_path):
            continue
        for run_dir in sorted(os.listdir(date_path)):
            run_path = os.path.join(date_path, run_dir)
            run_info_path = os.path.join(run_path, "run_info.json")
            weights_path  = os.path.join(run_path, "weights.npy")
            if not os.path.exists(run_info_path) or not os.path.exists(weights_path):
                continue
            with open(run_info_path) as f:
                config = json.load(f)["config"]
            if KEYWORD not in config.get("message", ""):
                continue
            seed = config.get("seed", 0)
            nt   = config["noise_train"]
            runs[seed]["NT" if nt else "non-NT"] = run_path
    return runs


if __name__ == '__main__':
    triplets, _ = triplet_generator.generate_pca_triplets(
        dataset="MNIST", label_space=3, num_triplets=5000, pca_dims=32, testing=False
    )

    runs = collect_runs()
    seeds = sorted(runs.keys())
    n_seeds = len(seeds)

    # pre-compute all data first so we can set consistent scales
    all_data = {}
    for seed in seeds:
        all_data[seed] = {}
        for label in ["NT", "non-NT"]:
            if label in runs[seed]:
                norms, shifts = analyse_run(runs[seed][label], triplets)
                all_data[seed][label] = (norms, shifts)

    # global scale limits
    all_shifts = [s for seed in all_data.values() for _, shifts in seed.values() for s in shifts]
    all_norms  = [n for seed in all_data.values() for norms, _ in seed.values() for n in norms]
    shift_min, shift_max = min(all_shifts), max(all_shifts)
    norm_min,  norm_max  = min(all_norms),  max(all_norms)
    pad = 0.05
    shift_range = shift_max - shift_min
    norm_range  = norm_max  - norm_min

    fig, axes = plt.subplots(1, n_seeds, figsize=(5 * n_seeds, 5))
    if n_seeds == 1:
        axes = [axes]

    for ax, seed in zip(axes, seeds):
        ax2 = ax.twinx()
        for label, color in [("NT", "coral"), ("non-NT", "steelblue")]:
            if label not in all_data[seed]:
                continue
            norms, shifts = all_data[seed][label]
            x = range(len(norms))
            ax.plot(x, shifts, color=color, linewidth=2, label=f'{label} shift')
            ax2.plot(x, norms, color=color, linewidth=1.5, linestyle='--', alpha=0.4)

        ax.set_ylim(shift_min - pad * shift_range, shift_max + pad * shift_range)
        ax2.set_ylim(norm_min  - pad * norm_range,  norm_max  + pad * norm_range)
        ax.set_title(f'Seed {seed}', fontsize=10)
        ax.set_xlabel('Checkpoint (every 20 epochs)', fontsize=8)
        ax.set_ylabel('Mean shift magnitude', fontsize=8)
        ax2.set_ylabel('Weight norm', color='gray', fontsize=7)
        ax2.tick_params(labelsize=6)
        ax.legend(loc='upper right', fontsize=7)
        ax.tick_params(labelsize=7)

    plt.suptitle(f'Shift magnitude vs weight norm over training | {KEYWORD}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(RESULTS_ROOT, "Batch", KEYWORD, f"{KEYWORD}_shift_norm_grid.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nSaved: {save_path}")

    # scatter plot: shift magnitude vs weight norm
    fig2, ax = plt.subplots(figsize=(8, 6))

    for seed in seeds:
        for label, color, marker in [("NT", "coral", "o"), ("non-NT", "steelblue", "s")]:
            if label not in all_data[seed]:
                continue
            norms, shifts = all_data[seed][label]
            ax.scatter(norms, shifts, color=color, marker=marker, alpha=0.6, s=60,
                      label=f'{label}' if seed == seeds[0] else "")

    # add trend lines
    for label, color in [("NT", "coral"), ("non-NT", "steelblue")]:
        all_norms_label  = numpy.concatenate([all_data[s][label][0] for s in seeds if label in all_data[s]])
        all_shifts_label = numpy.concatenate([all_data[s][label][1] for s in seeds if label in all_data[s]])
        z = numpy.polyfit(all_norms_label, all_shifts_label, 1)
        p = numpy.poly1d(z)
        x_line = numpy.linspace(all_norms_label.min(), all_norms_label.max(), 100)
        ax.plot(x_line, p(x_line), color=color, linewidth=2, linestyle='--', alpha=0.8)

    ax.set_xlabel('Weight norm (L2)', fontsize=11)
    ax.set_ylabel('Mean shift magnitude', fontsize=11)
    ax.set_title(f'Shift magnitude vs weight norm | {KEYWORD}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    scatter_path = os.path.join(RESULTS_ROOT, "Batch", KEYWORD, f"{KEYWORD}_shift_vs_norm_scatter.png")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"Saved: {scatter_path}")