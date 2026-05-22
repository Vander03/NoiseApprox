import numpy
import matplotlib.pyplot as plt
import pandas as pd
import os, json, pickle
from umap import UMAP
from tqdm import tqdm


class Visualiser:
    def __init__(self, results_dir):
        self.results_dir = results_dir

    def _ensure_results_dir(self):
        os.makedirs(self.results_dir, exist_ok=True)

    def plot_loss(self, loss_history, clean_loss_history=None, noisy_loss_history=None, dataset="MNIST"):
        epochs = range(len(loss_history))
        smoothed_total = pd.Series(loss_history).rolling(window=10, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(10, 5))

        if clean_loss_history and noisy_loss_history:
            smoothed_clean = pd.Series(clean_loss_history).rolling(window=10, min_periods=1).mean()
            smoothed_noisy = pd.Series(noisy_loss_history).rolling(window=10, min_periods=1).mean()
            ax.fill_between(epochs, smoothed_clean, alpha=0.25, color='steelblue', label='Clean component')
            ax.fill_between(epochs, smoothed_noisy, alpha=0.25, color='coral', label='Noisy component')

        ax.plot(loss_history, alpha=0.2, color='steelblue')
        ax.plot(smoothed_total, color='steelblue', linewidth=2, label='Total loss (smoothed)')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Triplet Loss')
        ax.set_title(f'SLIQ Training Loss ({dataset})')
        ax.legend()
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'loss.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_embedding_spread(self, embeddings, save_name='embedding_spread.png',
                               title='Per-dimension shift spread by sample', y_label="Shift Magnitude"):
        self._ensure_results_dir()
        emb_array = numpy.array([e[0] for e in embeddings])
        label     = [e[1] for e in embeddings]

        unique_labels = sorted(set(label))
        if 'Noiseless' in unique_labels:
            unique_labels = ['Noiseless'] + [b for b in unique_labels if b != 'Noiseless']

        n_labels = len(unique_labels)
        n_dims   = emb_array.shape[1]
        colors   = ['steelblue', 'coral', 'mediumseagreen', 'gold', 'orchid']

        candle_width = 0.12
        fig_width    = max(6, n_dims * (candle_width * n_labels + 0.4) + 1)
        fig, ax      = plt.subplots(figsize=(fig_width, 4))

        for i, l in enumerate(unique_labels):
            mask      = numpy.array([b == l for b in label])
            pts       = emb_array[mask]
            positions = numpy.arange(n_dims) + (i - n_labels / 2 + 0.5) * candle_width
            ax.boxplot(
                [pts[:, d] for d in range(n_dims)],
                positions=positions,
                widths=candle_width * 0.8,
                patch_artist=True,
                boxprops=dict(facecolor=colors[i % len(colors)], alpha=0.6),
                medianprops=dict(color='black', linewidth=1.5),
                whiskerprops=dict(color=colors[i % len(colors)]),
                capprops=dict(color=colors[i % len(colors)]),
                flierprops=dict(marker='o', color=colors[i % len(colors)], alpha=0.3, markersize=3),
                label=f"{l}"
            )

        ax.set_xticks(numpy.arange(n_dims))
        ax.set_xticklabels([f'Z{d}' for d in range(n_dims)])
        ax.set_xlim(-0.5, n_dims - 0.5)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_ylabel(y_label)
        ax.set_xlabel('Embedding dimension')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8, title='Sample')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, save_name)
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_embedding_spread_umap(self, embeddings, save_name='embedding_spread_umap.png'):
        self._ensure_results_dir()
        emb_array  = numpy.array([e[0] for e in embeddings])
        backends   = [e[1] for e in embeddings]
        umap_model = UMAP(n_components=2, random_state=42)
        embs_2d    = umap_model.fit_transform(emb_array)

        fig, ax = plt.subplots(figsize=(9, 7))
        colors  = ['steelblue', 'coral', 'mediumseagreen', 'gold', 'orchid']

        for i, backend in enumerate(sorted(set(backends))):
            mask = numpy.array([b == backend for b in backends])
            pts  = embs_2d[mask]
            marker = "D" if backend == "Noiseless" else "X" if "holdout" in backend else "o"
            alpha  = 1.0 if backend == "Noiseless" else 0.55
            ax.scatter(pts[:, 0], pts[:, 1], s=20, alpha=alpha,
                       color=colors[i % len(colors)], label=backend, marker=marker)

        ax.legend(fontsize=8, title='Backend')
        ax.set_title(f'Embedding spread (n={len(emb_array)})')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, save_name)
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_embedding_shift_magnitude(self, embeddings, save_name="embedding_shift_mag.png"):
        self._ensure_results_dir()
        emb_array       = numpy.array([e[0] for e in embeddings])
        backends        = [e[1] for e in embeddings]
        magnitudes      = numpy.linalg.norm(emb_array, axis=1)
        unique_backends = sorted(set(backends))
        data = [magnitudes[[b == backend for b in backends]] for backend in unique_backends]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.violinplot(data, positions=range(len(unique_backends)), showmedians=True)
        ax.set_xticks(range(len(unique_backends)))
        ax.set_xticklabels(unique_backends, rotation=20, ha='right')
        ax.set_ylabel('Shift magnitude (L2)')
        ax.set_title('Embedding shift magnitude by backend')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, save_name)
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"Saved: {save_path}")

    def measure_shift_distribution(self, triplets, model, ss_samples, n_samples=30, save_name="shift_spread.png"):
        """measure and plot shift vectors for given samples — call after training"""
        all_shift_vectors = []
        shift_backends    = []

        for r in ss_samples:
            for _ in tqdm(range(n_samples), desc=f"Shifts for sample {r}", leave=False):
                sample    = triplets[r][0]
                clean_emb = numpy.array([float(z) for z in model.qiskit_circuit(model, model.weights, numpy.array(sample))])
                for prof in model.np_train:
                    noisy_emb    = numpy.array([float(z) for z in prof["circuit"](model, model.weights, numpy.array(sample))])
                    shift_vector = noisy_emb - clean_emb
                    all_shift_vectors.append(shift_vector)
                    shift_backends.append(r)

        all_shift_vectors = numpy.array(all_shift_vectors)
        self.plot_embedding_spread(
            list(zip(all_shift_vectors, shift_backends)),
            save_name=save_name
        )
        return all_shift_vectors

    def evaluate_embedding_space(self, triplets, labels, model, save_name="embedding_dims_variance.png"):
        """plot clean embedding distribution per class"""
        emb = []
        for i, triplet in enumerate(tqdm(triplets, desc="Getting Clean Embeddings")):
            sample    = triplet[0]
            clean_emb = numpy.array([float(z) for z in model.qiskit_circuit(model, model.weights, numpy.array(sample))])
            emb.append((clean_emb, str(labels[i])))
        self.plot_embedding_spread(emb, save_name=save_name,
                                   title="Embedding distribution by dimension",
                                   y_label="Expectation value")