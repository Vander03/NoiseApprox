# analyse_batch.py
# prints summary table and generates trajectory plots for all runs matching a keyword

import sys
sys.path.append('..')
from model import Triplet
import triplet_generator
import os, json
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import matplotlib.colors as mcolors

results_root = "Results"
keyword = "clusterNoWeightOvernight"  # change to match your run messages
samples = 500

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────

summary = defaultdict(list)
matched_runs = []  # store paths for trajectory plotting

for date_dir in sorted(os.listdir(results_root)):
    date_path = os.path.join(results_root, date_dir)
    if not os.path.isdir(date_path):
        continue
    for run_dir in sorted(os.listdir(date_path)):
        run_path = os.path.join(date_path, run_dir)
        noisy_path = os.path.join(run_path, "noisy_eval_results.json")
        run_info_path = os.path.join(run_path, "run_info.json")
        weights_path = os.path.join(run_path, "weights.npy")

        if not os.path.exists(noisy_path) or not os.path.exists(run_info_path):
            continue

        with open(run_info_path) as f:
            run_info = json.load(f)
        with open(noisy_path) as f:
            noisy = json.load(f)

        config = run_info["config"]
        message = config.get("message", "")

        if keyword not in message:
            continue

        accs = [r["accuracy"] for r in noisy.get("results", [])
                if r.get("filename") != "clean"]
        mean_acc = numpy.mean(accs) if accs else None

        for backend in ['ibm_kingston', 'ibm_fez', 'ibm_marrakesh']:
            backend_accs = [r['accuracy'] for r in noisy.get("results", [])
                            if r.get('backend') == backend 
                            and r.get('filename') != 'clean']
            print(f"NT{config['noise_train']}: {backend}: mean={numpy.mean(backend_accs):.1f}%")

        if mean_acc is not None:
            lr = config["learning_rate"]
            nt = config["noise_train"]
            seed = config.get("seed", "?")
            key = (f"lr={lr}", "NT" if nt else "non-NT")
            summary[key].append(mean_acc)
            # if nt:
            #     print(f"NT seed:{seed} - Mean acc:{mean_acc:.2f}%")
            matched_runs.append({
                "path": run_path,
                "config": config,
                "mean_acc": mean_acc,
                "has_weights": os.path.exists(weights_path)
            })

print(f"\n{'Condition':<17} {'N':>4} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 60)
for (lr, nt), accs in sorted(summary.items()):
    accs = numpy.array(accs)
    print(f"{lr} {nt:<10} {len(accs):>4} {accs.mean():>8.2f} {accs.std():>8.2f} {accs.min():>8.2f} {accs.max():>8.2f}")


# ─────────────────────────────────────────────────────────────
# PER-BACKEND SUMMARY
# ─────────────────────────────────────────────────────────────

backends = ['ibm_kingston', 'ibm_fez', 'ibm_marrakesh']
backend_summary = defaultdict(lambda: defaultdict(list))  # backend → condition → [accs]

for run in matched_runs:
    path   = run["path"]
    config = run["config"]
    nt     = config["noise_train"]
    condition = "NT" if nt else "non-NT"
    if (keyword not in config['message']):
        continue

    noisy_path = os.path.join(path, "noisy_eval_results.json")
    with open(noisy_path) as f:
        noisy_data = json.load(f)

    results = noisy_data.get("results", [])
    clean_acc = next((r["accuracy"] for r in results if r.get("filename") == "clean"), None)

    for backend in backends:
        backend_accs = [r["accuracy"] for r in results
                        if r.get("backend") == backend
                        and r.get("filename") != "clean"]
        if backend_accs:
            backend_summary[backend][condition].extend(backend_accs)
            backend_summary[backend][f"{condition}_clean"].extend([clean_acc] if clean_acc else [])

print(f"\n{'Backend':<20} {'Condition':<10} {'N':>4} {'Clean':>8} {'Noisy Mean':>12} {'Drop':>8} {'Filename':>8}")
print("-" * 70)
for backend in backends:
    for condition in ["non-NT", "NT"]:
        noisy_accs = backend_summary[backend][condition]
        clean_accs = backend_summary[backend][f"{condition}_clean"]
        if not noisy_accs:
            continue
        noisy_mean = numpy.mean(noisy_accs)
        clean_mean = numpy.mean(clean_accs) if clean_accs else 0
        drop = clean_mean - noisy_mean
        n = len(noisy_accs)
        print(f"{backend:<20} {condition:<10} {n:>4} {clean_mean:>8.1f}% {noisy_mean:>11.1f}% {drop:>7.1f}% ")
    print()
# in summarise_seeds.py or a new script


print(f"\n{'Seed':<8} {'Condition':<10} {'Mean':>8} {'Clean':>8} {'Min Noisy':>12} {'Max Noisy':>12}")
print("-" * 60)

# collect per-seed details
seed_runs = sorted(matched_runs, key=lambda r: (r["config"].get("seed", 0), int(r["config"]["noise_train"])))

for run in seed_runs:
    path   = run["path"]
    config = run["config"]
    seed   = config.get("seed", "?")
    nt     = "NT" if config["noise_train"] else "non-NT"

    noisy_path = os.path.join(path, "noisy_eval_results.json")
    with open(noisy_path) as f:
        noisy_data = json.load(f)

    results    = noisy_data.get("results", [])
    clean_acc  = next((r["accuracy"] for r in results if r.get("filename") == "clean"), None)
    noisy_accs = [r["accuracy"] for r in results if r.get("filename") != "clean"]

    if not noisy_accs:
        continue

    mean   = numpy.mean(noisy_accs)
    mn     = numpy.min(noisy_accs)
    mx     = numpy.max(noisy_accs)
    clean  = clean_acc if clean_acc else 0

    print(f"{seed:<8} {nt:<10} {mean:>8.1f}% {clean:>7.1f}% {mn:>11.1f}% {mx:>11.1f}%")

sys.exit()

# ─────────────────────────────────────────────────────────────
# TRAJECTORY PLOTS
# ─────────────────────────────────────────────────────────────

def plot_trajectory_on_landscape(model, triplets, weight_history, save_path, resolution=25, range_=1.0):
    if len(weight_history) < 2:
        print("  Not enough checkpoints for trajectory plot")
        return

    w_start = weight_history[0]
    w_end   = weight_history[-1]

    d1 = w_end - w_start
    norm = numpy.linalg.norm(d1)
    if norm < 1e-10:
        print("  Weights didn't change — skipping trajectory")
        return
    d1 = d1 / norm

    grads = []
    for _ in range(10):
        w_perturb = model.weights + 0.01 * numpy.random.randn(*model.weights.shape)
        g = (model.loss(w_perturb, batch)[0] - model.loss(model.weights, batch)[0]) / 0.01
        grads.append(w_perturb.flatten() - model.weights.flatten())

    grads = numpy.array(grads)
    U, S, Vt = numpy.linalg.svd(grads, full_matrices=False)
    d1 = Vt[0].reshape(model.weights.shape)
    d2 = Vt[1].reshape(model.weights.shape)

    # project checkpoints onto (d1, d2)
    trajectory = []
    for w in weight_history:
        delta = w - w_start
        x = numpy.dot(delta.flatten(), d1.flatten())
        y = numpy.dot(delta.flatten(), d2.flatten())
        trajectory.append((x, y))

    # build landscape around midpoint
    mid = weight_history[len(weight_history) // 2]
    alphas = numpy.linspace(-range_, range_, resolution)
    betas  = numpy.linspace(-range_, range_, resolution)
    loss_grid = numpy.zeros((resolution, resolution))

    batch_idx = numpy.random.randint(0, len(triplets), 16)
    batch = [triplets[i] for i in batch_idx]

    for i, alpha in enumerate(tqdm(alphas, desc="  Landscape", leave=False)):
        for j, beta in enumerate(betas):
            w = mid + alpha * d1 + beta * d2
            try:
                loss_grid[i, j] = numpy.clip(float(model.loss(w, batch)[0]), -3.0, 3.0)
            except Exception:
                loss_grid[i, j] = 0.0

    fig, ax = plt.subplots(figsize=(9, 7))
    A, B = numpy.meshgrid(alphas, betas)
    cf = ax.contourf(A, B, loss_grid.T, levels=40, cmap='viridis', alpha=0.85)
    ax.contour(A, B, loss_grid.T, levels=40, colors='white', alpha=0.15, linewidths=0.4)
    plt.colorbar(cf, ax=ax, label='Loss')

    xs = [t[0] for t in trajectory]
    ys = [t[1] for t in trajectory]
    ax.plot(xs, ys, 'r-', linewidth=1.5, alpha=0.7)
    ax.scatter(xs[0],  ys[0],  color='lime', s=120, zorder=6, label='Start')
    ax.scatter(xs[-1], ys[-1], color='red',  s=120, zorder=6, label='End (final weights)')

    sc = ax.scatter(xs, ys, c=range(len(xs)), cmap='autumn', s=40, zorder=7)
    plt.colorbar(sc, ax=ax, label='Checkpoint index')

    nt_str  = "NT" if model.noise_train else "non-NT"
    seed    = model.params.get("seed", "?")
    ax.set_title(f'Optimisation trajectory | {nt_str} seed={seed} | noisy mean={model._last_mean_acc:.1f}%')
    ax.set_xlabel('Direction 1 (start → end)')
    ax.set_ylabel('Direction 2 (orthogonal)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Trajectory saved: {save_path}")


print(f"\n\nGenerating trajectory plots for {len(matched_runs)} runs...\n")

# ─────────────────────────────────────────────────────────────
# BATCH TRAJECTORY GRID
# ─────────────────────────────────────────────────────────────

def compute_trajectory(model, triplets, weight_history, resolution=25, range_=1.0):
    """compute landscape + trajectory for one run, return fig data"""
    if len(weight_history) < 2:
        return None

    w_start = weight_history[0]
    w_end   = weight_history[-1]
    d1 = w_end - w_start
    norm = numpy.linalg.norm(d1)
    if norm < 1e-10:
        return None
    d1 = d1 / norm

    d2 = numpy.random.randn(*d1.shape)
    d2 = d2 - numpy.dot(d2.flatten(), d1.flatten()) * d1
    d2 = d2 / numpy.linalg.norm(d2)

    trajectory = []
    for w in weight_history:
        delta = w - w_start
        x = numpy.dot(delta.flatten(), d1.flatten())
        y = numpy.dot(delta.flatten(), d2.flatten())
        trajectory.append((x, y))

    mid = weight_history[len(weight_history) // 2]
    alphas = numpy.linspace(-range_, range_, resolution)
    betas  = numpy.linspace(-range_, range_, resolution)
    loss_grid = numpy.zeros((resolution, resolution))

    batch_idx = numpy.random.randint(0, len(triplets), 16)
    batch = [triplets[i] for i in batch_idx]

    for i, alpha in enumerate(tqdm(alphas, desc="  Landscape", leave=False)):
        for j, beta in enumerate(betas):
            w = mid + alpha * d1 + beta * d2
            try:
                loss_grid[i, j] = numpy.clip(float(model.loss(w, batch)[0]), -3.0, 3.0)
            except Exception:
                loss_grid[i, j] = 0.0

    return {
        "alphas": alphas,
        "betas": betas,
        "loss_grid": loss_grid,
        "trajectory": trajectory
    }


def make_batch_trajectory_grid(matched_runs, keyword, results_root, resolution=25, range_=1.0):
    batch_dir = os.path.join(results_root, "Batch", keyword)
    os.makedirs(batch_dir, exist_ok=True)

    # sort: by seed, then non-NT before NT
    matched_runs = sorted(matched_runs, key=lambda r: (
        r["config"].get("seed", 0),
        int(r["config"]["noise_train"])
    ))

    n_runs = len(matched_runs)
    n_cols = 4
    n_rows = (n_runs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = numpy.array(axes).flatten()

    for idx, run in enumerate(matched_runs):
        ax = axes[idx]
        path   = run["path"]
        config = run["config"]
        weights_path = os.path.join(path, "weights.npy")
        noisy_path   = os.path.join(path, "noisy_eval_results.json")

        nt_str = "NT" if config["noise_train"] else "non-NT"
        seed   = config.get("seed", "?")

        # get clean + noisy stats
        with open(noisy_path) as f:
            noisy_data = json.load(f)
        results   = noisy_data.get("results", [])
        clean_acc = next((r["accuracy"] for r in results if r.get("filename") == "clean"), None)
        noisy_accs = [r["accuracy"] for r in results if r.get("filename") != "clean"]
        noisy_mean   = numpy.mean(noisy_accs)   if noisy_accs else 0
        noisy_median = numpy.median(noisy_accs) if noisy_accs else 0

        title = (f"{nt_str} | seed={seed}\n"
                 f"clean={clean_acc:.1f}%  mean={noisy_mean:.1f}%  med={noisy_median:.1f}%")

        if not os.path.exists(weights_path):
            ax.set_title(title + "\n[no weights]", fontsize=8)
            ax.set_visible(True)
            continue

        try:
            model = Triplet(config, testing=True, results_dir=path)
            weight_history = numpy.load(weights_path, allow_pickle=True)

            best_path = os.path.join(path, "best_weights.npy")
            model.weights = (numpy.load(best_path, allow_pickle=True)
                             if os.path.exists(best_path)
                             else weight_history[-1])
            model.noise_train = config["noise_train"]

            t_triplets, _ = triplet_generator.generate_pca_triplets(
                dataset=config['dataset'],
                label_space=config['label_space'],
                num_triplets=config['num_triplets'],
                pca_dims=config['PCA_dims'],
                testing=True
            )

            print(f"  [{idx+1}/{n_runs}] {nt_str} seed={seed} — computing landscape...")
            data = compute_trajectory(model, t_triplets[:200], weight_history,
                                      resolution=resolution, range_=range_)

            if data is None:
                ax.set_title(title + "\n[insufficient checkpoints]", fontsize=8)
                continue

            A, B = numpy.meshgrid(data["alphas"], data["betas"])

            # percentile normalisation so flat NT landscapes use full colour range
            p5  = numpy.percentile(data["loss_grid"], 5)
            p95 = numpy.percentile(data["loss_grid"], 95)
            norm = mcolors.Normalize(vmin=p5, vmax=p95)

            cf = ax.contourf(A, B, data["loss_grid"].T, levels=30, cmap='viridis', norm=norm, alpha=0.85)
            ax.contour(A, B, data["loss_grid"].T, levels=30,
                    colors='white', alpha=0.15, linewidths=0.4)

            xs = [t[0] for t in data["trajectory"]]
            ys = [t[1] for t in data["trajectory"]]
            ax.plot(xs, ys, 'r-', linewidth=1.5, alpha=0.7)
            ax.scatter(xs[0],  ys[0],  color='lime', s=80, zorder=6)
            ax.scatter(xs[-1], ys[-1], color='red',  s=80, zorder=6)
            sc = ax.scatter(xs, ys, c=range(len(xs)), cmap='autumn', s=30, zorder=7)

        except Exception as e:
            ax.set_title(title + f"\n[ERROR: {e}]", fontsize=7)
            print(f"  ERROR on {nt_str} seed={seed}: {e}")
            continue

        ax.set_title(title, fontsize=9,
                     color='tomato' if config["noise_train"] else 'steelblue')
        ax.set_xlabel('Dir 1', fontsize=7)
        ax.set_ylabel('Dir 2', fontsize=7)
        ax.tick_params(labelsize=6)

    # hide unused axes
    for idx in range(n_runs, len(axes)):
        axes[idx].set_visible(False)

    # overall title
    nt_means  = [r["mean_acc"] for r in matched_runs if r["config"]["noise_train"]]
    non_means = [r["mean_acc"] for r in matched_runs if not r["config"]["noise_train"]]
    parts = [f"Batch: {keyword}"]
    if nt_means:
        parts.append(f"NT: {numpy.mean(nt_means):.1f}% ± {numpy.std(nt_means):.1f}%")
    if non_means:
        parts.append(f"non-NT: {numpy.mean(non_means):.1f}% ± {numpy.std(non_means):.1f}%")
    fig.suptitle("  |  ".join(parts), fontsize=12, fontweight='bold', y=1.005)

    plt.tight_layout()
    save_path = os.path.join(batch_dir, f"{keyword}_trajectories.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nBatch trajectory grid saved: {save_path}")


print(f"\n\nGenerating batch trajectory grid for {len(matched_runs)} runs...")
make_batch_trajectory_grid(matched_runs, keyword, results_root, resolution=25, range_=1.0)