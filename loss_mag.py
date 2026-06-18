# plot_loss_curves.py
import os
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results_root = "Results"
backends     = ["kingston", "fez", "marrakesh"]
seeds        = [1, 2, 3, 4, 5]
plots_dir    = os.path.join(results_root, "loss_plots")
os.makedirs(plots_dir, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# LOAD LOSS HISTORIES
# ─────────────────────────────────────────────────────────────

def load_losses(paths):
    losses = []
    for p in paths:
        if os.path.exists(p):
            arr = numpy.load(p, allow_pickle=True)
            losses.append(arr)
    return losses


noiseless_paths = [
    os.path.join(results_root, "noiseless", f"seed{s}", "loss_history.npy")
    for s in seeds
]
noiseless_losses = load_losses(noiseless_paths)

backend_losses = {}
for backend in backends:
    paths = [
        os.path.join(results_root, "NT_v3", backend, f"seed{s}", "loss_history.npy")
        for s in seeds
    ]
    backend_losses[backend] = load_losses(paths)


def mean_and_std(loss_list):
    if not loss_list:
        return None, None
    min_len = min(len(l) for l in loss_list)
    arr     = numpy.array([l[:min_len] for l in loss_list])
    return arr.mean(axis=0), arr.std(axis=0)


noiseless_mean, noiseless_std = mean_and_std(noiseless_losses)

# ─────────────────────────────────────────────────────────────
# PLOT — 3 subplots, one per backend
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
fig.suptitle("Mean Training Loss per Backend vs Noiseless", fontsize=13)

colors = {"noiseless": "steelblue", "nt": "coral"}

for ax, backend in zip(axes, backends):
    nt_mean, nt_std = mean_and_std(backend_losses[backend])
    epochs          = numpy.arange(1, len(nt_mean) + 1) if nt_mean is not None else []

    if noiseless_mean is not None:
        nl_len = min(len(noiseless_mean), len(epochs)) if nt_mean is not None else len(noiseless_mean)
        nl_x   = numpy.arange(1, nl_len + 1)
        ax.plot(nl_x, noiseless_mean[:nl_len], color=colors["noiseless"], label="Noise-Naive", linewidth=1.5)
        ax.fill_between(nl_x,
                         noiseless_mean[:nl_len] - noiseless_std[:nl_len],
                         noiseless_mean[:nl_len] + noiseless_std[:nl_len],
                         color=colors["noiseless"], alpha=0.15)

    if nt_mean is not None:
        ax.plot(epochs, nt_mean, color=colors["nt"], label="Noise-Aware", linewidth=1.5)
        ax.fill_between(epochs,
                         nt_mean - nt_std,
                         nt_mean + nt_std,
                         color=colors["nt"], alpha=0.15)

    ax.set_title(f"IBM {backend.capitalize()}", fontsize=11)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(plots_dir, "loss_curves_by_backend.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")