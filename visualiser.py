import numpy
import matplotlib.pyplot as plt
import pandas as pd
import os
from umap import UMAP
from tqdm import tqdm


class Visualiser:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        import matplotlib as mpl

    def _ensure_results_dir(self):
        os.makedirs(self.results_dir, exist_ok=True)

    def plot_loss(self, loss_history, clean_loss_history=None, noisy_loss_history=None, dataset="MNIST"):
        smoothed_total = pd.Series(loss_history).rolling(window=10, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(loss_history, alpha=0.2, color="steelblue")
        ax.plot(smoothed_total, color="steelblue", linewidth=2, label="Total loss (smoothed)")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Triplet Loss")
        ax.set_title(f"SLIQ Training Loss ({dataset})")
        ax.legend()
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, "loss.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_embedding_spread(
        self, embeddings, save_name="embedding_spread.png", title="Per-dimension shift spread by sample", y_label="Shift Magnitude"
    ):
        self._ensure_results_dir()

        emb_array = numpy.array([e[0] for e in embeddings])
        label = [e[1] for e in embeddings]

        unique_labels = sorted(set(label))
        if "Noiseless" in unique_labels:
            unique_labels = ["Noiseless"] + [b for b in unique_labels if b != "Noiseless"]

        n_labels = len(unique_labels)
        n_dims = emb_array.shape[1]
        colors = ["steelblue", "coral", "mediumseagreen", "gold", "orchid"]

        candle_width = 0.12
        group_width = candle_width * n_labels
        fig_width = max(6, n_dims * (group_width + 0.4) + 1)

        fig, ax = plt.subplots(figsize=(fig_width, 4))

        for i, l in enumerate(unique_labels):
            mask = numpy.array([b == l for b in label])
            pts = emb_array[mask]
            positions = numpy.arange(n_dims) + (i - n_labels / 2 + 0.5) * candle_width
            ax.boxplot(
                [pts[:, d] for d in range(n_dims)],
                positions=positions,
                widths=candle_width * 0.8,
                patch_artist=True,
                boxprops=dict(facecolor=colors[i % len(colors)], alpha=0.6),
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(color=colors[i % len(colors)]),
                capprops=dict(color=colors[i % len(colors)]),
                flierprops=dict(marker="o", color=colors[i % len(colors)], alpha=0.3, markersize=3),
                label=f"{l}",
            )

        ax.set_xticks(numpy.arange(n_dims))
        ax.set_xticklabels([f"Z{d}" for d in range(n_dims)])
        ax.set_xlim(-0.5, n_dims - 0.5)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(y_label)
        ax.set_xlabel("Embedding dimension")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, title="Sample")
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, save_name)
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_embedding_spread_umap(self, embeddings, save_name="embedding_spread_umap.png"):
        self._ensure_results_dir()

        emb_array = numpy.array([e[0] for e in embeddings])
        backends = [e[1] for e in embeddings]

        umap_model = UMAP(n_components=2, random_state=42)
        embs_2d = umap_model.fit_transform(emb_array)
        fig, ax = plt.subplots(figsize=(12, 8))

        unique_backends = sorted(set(backends))
        colors = ["steelblue", "coral", "mediumseagreen", "gold", "orchid"]
        for i, backend in enumerate(unique_backends):
            mask = numpy.array([b == backend for b in backends])
            pts = embs_2d[mask]
            if backend == "Noiseless":
                ax.scatter(pts[:, 0], pts[:, 1], s=30, alpha=1, color=colors[i % len(colors)], label=backend, marker="D", zorder=5)
            elif "holdout" in backend:
                ax.scatter(pts[:, 0], pts[:, 1], s=20, alpha=1, color=colors[i % len(colors)], label=backend, marker="X", zorder=1)
            else:
                ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.55, color=colors[i % len(colors)], label=backend, zorder=1)

        ax.legend(fontsize=8, title="Backend")
        ax.set_title(f"Embedding spread (n={len(emb_array)})")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        save_path = os.path.join(self.results_dir, save_name)
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_embedding_shift_magnitude(self, embeddings, save_name="embedding_shift_mag.png"):
        self._ensure_results_dir()

        emb_array = numpy.array([e[0] for e in embeddings])
        backends = [e[1] for e in embeddings]
        magnitudes = numpy.linalg.norm(emb_array, axis=1)

        unique_backends = sorted(set(backends))
        data = [magnitudes[[b == backend for b in backends]] for backend in unique_backends]

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.violinplot(data, positions=range(len(unique_backends)), showmedians=True)
        ax.set_xticks(range(len(unique_backends)))
        ax.set_xticklabels(unique_backends, rotation=20, ha="right")
        ax.set_ylabel("Shift magnitude (L2)")
        ax.set_title("Embedding shift magnitude by backend")
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, save_name)
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"Saved: {save_path}")

    def evaluate_embedding_space(self, triplets, labels, model, save_name="embedding_dims_variance.png"):
        emb = []
        for i, triplet in enumerate(tqdm(triplets, desc="Getting Clean Embeddings")):
            sample = triplet[0]
            clean_emb = numpy.array([float(z) for z in model.qiskit_circuit(model, model.weights, numpy.array(sample))])
            emb.append((clean_emb, str(labels[i])))
        self.plot_embedding_spread(emb, save_name=save_name, title="Embedding distribution by dimension", y_label="Expectation value")

    def measure_shift_distribution(self, triplets, model, ss_samples, n_samples=30, save_name="shift_spread.png"):
        all_shift_vectors = []
        shift_backends = []

        for r in ss_samples:
            for _ in tqdm(range(n_samples), desc=f"Shifts for sample {r}", leave=False):
                sample = triplets[r][0]
                clean_emb = numpy.array([float(z) for z in model.qiskit_circuit(model, model.weights, numpy.array(sample))])
                for prof in model.np_train:
                    noisy_emb = numpy.array([float(z) for z in prof["circuit"](model, model.weights, numpy.array(sample))])
                    all_shift_vectors.append(noisy_emb - clean_emb)
                    shift_backends.append(r)

        all_shift_vectors = numpy.array(all_shift_vectors)
        self.plot_embedding_spread(list(zip(all_shift_vectors, shift_backends)), save_name=save_name)
        return all_shift_vectors

    def plot_shift_approximation_quality(self, clean_embs, shift_bank, kmeans, backend, predicted_shifts=None, cosine_sims=None, holdout_shifts=None):
        """cosine similarity distribution and compass plot for direction"""
        from umap import UMAP
        from sklearn.neighbors import NearestNeighbors
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        rep_embs = numpy.array([s[0] for s in shift_bank])
        rep_shifts = numpy.array([s[1] for s in shift_bank])

        if cosine_sims is None:
            actual_shifts = rep_shifts
            cosine_sims = []
            for i, (emb, actual) in enumerate(zip(rep_embs, rep_shifts)):
                other_embs = numpy.delete(rep_embs, i, axis=0)
                other_shifts = numpy.delete(rep_shifts, i, axis=0)
                knn_loo = NearestNeighbors(n_neighbors=3)
                knn_loo.fit(other_embs)
                distances, idx = knn_loo.kneighbors(emb.reshape(1, -1))
                weights = 1.0 / (distances[0] + 1e-8)
                weights = weights / weights.sum()
                pred = sum(w * other_shifts[j] for w, j in zip(weights, idx[0]))
                norm = numpy.linalg.norm(pred) * numpy.linalg.norm(actual)
                cosine_sims.append(numpy.dot(pred, actual) / norm if norm > 1e-8 else 0.0)
            cosine_sims = numpy.array(cosine_sims)
            proj_embs = rep_embs
            actual_shifts = rep_shifts
        else:
            actual_shifts = numpy.array([s[1] for s in shift_bank])
            cosine_sims = numpy.array(cosine_sims)
            proj_embs = clean_embs

        magnitudes = numpy.array([numpy.linalg.norm(a) for a in actual_shifts])
        total_magnitude = magnitudes.sum()
        weighted_mean = float(numpy.sum(cosine_sims * magnitudes) / total_magnitude) if total_magnitude > 1e-8 else 0.0
        mean_cos = float(numpy.mean(cosine_sims))
        median_cos = float(numpy.median(cosine_sims))

        # ── compass setup ────────────────────────────────────────────────
        actual_norms = numpy.linalg.norm(actual_shifts, axis=1, keepdims=True)
        actual_unit = actual_shifts / numpy.where(actual_norms > 1e-8, actual_norms, 1)
        mean_actual = numpy.mean(actual_unit, axis=0)
        mean_actual = mean_actual / numpy.linalg.norm(mean_actual)

        has_compass = False
        if predicted_shifts is not None:
            pred_norms = numpy.linalg.norm(predicted_shifts, axis=1, keepdims=True)
            pred_unit = predicted_shifts / numpy.where(pred_norms > 1e-8, pred_norms, 1)

            pca = PCA(n_components=2)
            pca.fit(pred_unit)
            pc1 = pca.components_[0]
            pc1 = pc1 - numpy.dot(pc1, mean_actual) * mean_actual
            pc1_norm = numpy.linalg.norm(pc1)
            pc1 = pc1 / pc1_norm if pc1_norm > 1e-8 else numpy.eye(len(mean_actual))[0]

            def project(vecs):
                return numpy.dot(vecs, mean_actual), numpy.dot(vecs, pc1)

            actual_x, actual_y = project(actual_unit)
            pred_x, pred_y = project(pred_unit)
            actual_angles = numpy.arctan2(actual_y, actual_x)
            pred_angles = numpy.arctan2(pred_y, pred_x)
            p25, p75 = numpy.percentile(pred_angles, [25, 75])
            angle_min = numpy.min(pred_angles)
            angle_max = numpy.max(pred_angles)
            angle_med = numpy.median(pred_angles)
            has_compass = True

            # ── holdout mean direction ────────────────────────────────────
            holdout_angle = None
            if holdout_shifts is not None:
                holdout_shifts = numpy.array(holdout_shifts)
                h_norms = numpy.linalg.norm(holdout_shifts, axis=1, keepdims=True)
                h_unit = holdout_shifts / numpy.where(h_norms > 1e-8, h_norms, 1)
                mean_holdout = numpy.mean(h_unit, axis=0)
                mean_holdout = mean_holdout / (numpy.linalg.norm(mean_holdout) + 1e-8)
                h_x, h_y = project(mean_holdout.reshape(1, -1))
                holdout_angle = float(numpy.arctan2(h_y[0], h_x[0]))

        # ── figure ────────────────────────────────────────────────────────
        if has_compass:
            fig = plt.figure(figsize=(14, 5))
            ax_hist = fig.add_subplot(1, 2, 1)
            ax_compass = fig.add_subplot(1, 2, 2, projection="polar")
        else:
            fig, ax_hist = plt.subplots(figsize=(8, 5))

        # histogram
        from scipy.stats import gaussian_kde

        ax_hist.hist(cosine_sims, bins=20, color="steelblue", alpha=0.5, edgecolor="white")
        ax_hist.set_xlabel("Cosine similarity (predicted vs ground truth shift)", fontsize=11)
        ax_hist.set_ylabel("Count", fontsize=11)
        ax_hist.set_xlim(-1, 1)
        ax_hist.set_ylim(0, 150)
        ax_hist.set_title("Approximation quality distribution", fontsize=12)

        ax_kde = ax_hist.twinx()
        kde = gaussian_kde(cosine_sims, bw_method=0.05)
        x_range = numpy.linspace(-1, 1, 500)
        ax_kde.plot(x_range, kde(x_range), color="steelblue", linewidth=2.5)
        ax_kde.set_ylabel("Density", fontsize=10)
        ax_kde.tick_params(labelsize=9)
        ax_kde.set_ylim(bottom=0)

        if has_compass:
            ax_compass.scatter(actual_angles, numpy.ones_like(actual_angles) * 0.7, alpha=0.2, color="coral", s=20, zorder=2)
            ax_compass.scatter(pred_angles, numpy.ones_like(pred_angles) * 1.0, alpha=0.2, color="steelblue", s=20, zorder=3)

            arc_angles = numpy.linspace(angle_min, angle_max, 100)
            ax_compass.plot(arc_angles, numpy.ones_like(arc_angles) * 1.05, "steelblue", linestyle=":", linewidth=1.5, alpha=0.6)

            iqr_angles = numpy.linspace(p25, p75, 100)
            ax_compass.plot(iqr_angles, numpy.ones_like(iqr_angles) * 1.05, "steelblue", linestyle="-", linewidth=3, alpha=0.8)

            ax_compass.annotate(
                "", xy=(angle_med, 0.95), xytext=(angle_med, 0.0), arrowprops=dict(arrowstyle="->", color="steelblue", lw=2.5), zorder=5
            )
            ax_compass.annotate("", xy=(0.0, 0.95), xytext=(0.0, 0.0), arrowprops=dict(arrowstyle="->", color="black", lw=3), zorder=6)

            # holdout mean direction arrow
            if holdout_angle is not None:
                ax_compass.annotate(
                    "", xy=(holdout_angle, 0.95), xytext=(holdout_angle, 0.0), arrowprops=dict(arrowstyle="->", color="red", lw=2.5), zorder=7
                )

            ax_compass.set_yticklabels([])
            ax_compass.set_xticks(numpy.linspace(0, 2 * numpy.pi, 8, endpoint=False))
            ax_compass.set_xticklabels(["0°", "45°", "90°", "135°", "±180°", "-135°", "-90°", "-45°"], fontsize=9)
            ax_compass.set_ylim(0, 1.2)
            ax_compass.set_title("Predicted direction distribution\n(reference: ground truth mean = 0°)", fontsize=11)

            legend_elements = [
                Line2D([0], [0], color="black", lw=3, label="Ground Truth mean direction (0°)"),
                Line2D([0], [0], color="steelblue", lw=2.5, label="Median predicted direction"),
                Line2D([0], [0], color="steelblue", lw=3, label="IQR of predicted"),
                Line2D([0], [0], color="steelblue", lw=1.5, linestyle=":", label="Full range of predicted"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue", markersize=8, alpha=0.6, label="Predicted directions"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="coral", markersize=8, alpha=0.6, label="Ground Truth directions"),
            ]
            if holdout_angle is not None:
                legend_elements.append(Line2D([0], [0], color="red", lw=2.5, label="Holdout mean direction"))
            fig.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.75, -0.12), ncol=3, fontsize=8, framealpha=0.9)

        plt.suptitle(f"{backend}   |   Mean: {mean_cos:.3f}   Median: {median_cos:.3f}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "shift_cos_sim_dist.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: shift_cos_sim_dist.png | Mean: {mean_cos:.3f} | Median: {median_cos:.3f} | Weighted: {weighted_mean:.3f}")

    def plot_shift_bank(self, clean_embs, shift_bank, kmeans, backend, external_knn_embs=None, external_knn_shifts=None):
        from umap import UMAP
        from sklearn.manifold import TSNE
        from sklearn.neighbors import NearestNeighbors
        from matplotlib.lines import Line2D
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        mpl.rcParams.update({"font.size": 12})

        rep_embs = numpy.array([s[0] for s in shift_bank])
        rep_shifts = numpy.array([s[1] for s in shift_bank])
        n_clean = len(clean_embs)
        n_rep = len(rep_embs)

        if external_knn_embs is not None and external_knn_shifts is not None:
            knn_embs_src = external_knn_embs
            knn_shifts_src = external_knn_shifts
        else:
            knn_embs_src = rep_embs
            knn_shifts_src = rep_shifts

        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(knn_embs_src)

        def _predict(emb):
            _, idx = knn.kneighbors(emb.reshape(1, -1))
            return knn_shifts_src[idx[0][0]]

        # ── shared cluster sort ──────────────────────────────────────────
        cluster_assignments = kmeans.predict(rep_embs)
        sort_idx = numpy.argsort(cluster_assignments)
        sorted_clusters = cluster_assignments[sort_idx]
        boundaries = numpy.where(numpy.diff(sorted_clusters))[0] + 1

        # ── stratified subsample for heatmap ────────────────────────────
        if len(rep_embs) > 200:
            samples_per_cluster = 200 // len(numpy.unique(cluster_assignments))
            stratified_idx = []
            for k in numpy.unique(cluster_assignments):
                cluster_idx = numpy.where(cluster_assignments == k)[0]
                n_select = min(samples_per_cluster, len(cluster_idx))
                selected = cluster_idx[numpy.linspace(0, len(cluster_idx) - 1, n_select, dtype=int)]
                stratified_idx.extend(selected)
            stratified_idx = numpy.array(stratified_idx)

            heatmap_embs = rep_embs[stratified_idx]
            heatmap_shifts = rep_shifts[stratified_idx]
            heatmap_clusters = kmeans.predict(heatmap_embs)
            heatmap_sort = numpy.argsort(heatmap_clusters)
            sorted_shifts = heatmap_shifts[heatmap_sort]
            heatmap_pred = numpy.array([_predict(e) for e in heatmap_embs])
            sorted_predicted = heatmap_pred[heatmap_sort]
            heatmap_sorted_clusters = heatmap_clusters[heatmap_sort]
            heatmap_boundaries = numpy.where(numpy.diff(heatmap_sorted_clusters))[0] + 1
        else:
            sorted_shifts = rep_shifts[sort_idx]
            predicted_shifts = numpy.array([_predict(emb) for emb in rep_embs])
            sorted_predicted = predicted_shifts[sort_idx]
            heatmap_boundaries = boundaries

        def _add_cluster_lines(ax):
            for b in heatmap_boundaries:
                ax.axhline(b - 0.5, color="black", linewidth=1.5)

        def _heatmap_axes(ax, data, title):
            im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-0.2, vmax=0.2)
            _add_cluster_lines(ax)
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Embedding dimension", fontsize=12)
            ax.set_xticks(range(data.shape[1]))
            ax.set_xticklabels([f"Z{i}" for i in range(data.shape[1])], fontsize=11)
            ax.tick_params(axis="y", labelsize=10)
            return im

        # ── side by side heatmap with difference ─────────────────────────
        diff = sorted_shifts - sorted_predicted

        fig, axes = plt.subplots(1, 4, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})
        im = _heatmap_axes(axes[0], sorted_shifts, "Ground Truth shifts")
        _heatmap_axes(axes[1], sorted_predicted, "Predicted shifts — KNN")

        axes[2].imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-0.2, vmax=0.2)
        _add_cluster_lines(axes[2])
        axes[2].set_title("Prediction error (Ground Truth - Predicted)", fontsize=14)
        axes[2].set_xlabel("Embedding dimension", fontsize=12)
        axes[2].set_xticks(range(diff.shape[1]))
        axes[2].set_xticklabels([f"Z{i}" for i in range(diff.shape[1])], fontsize=11)
        axes[2].tick_params(axis="y", labelsize=10)

        cbar = fig.colorbar(im, cax=axes[3])
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("Shift magnitude (signed)", fontsize=12)
        axes[0].set_ylabel("Sample (sorted by cluster)", fontsize=12)
        axes[1].set_ylabel("")
        axes[2].set_ylabel("")
        plt.suptitle(f"{backend} - Ground Truth vs predicted noise shift direction", fontweight="bold", fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "shift_heatmap_comparison.png"), dpi=150)
        plt.close()
        print(f"Saved: shift_heatmap_comparison.png")

    def plot_shift_field_cosine_overlay(self, clean_embs, shift_bank, kmeans, backend, cosine_sims, external_knn_embs=None, external_knn_shifts=None):
        from sklearn.neighbors import NearestNeighbors
        import matplotlib.pyplot as plt

        rep_embs = numpy.array([s[0] for s in shift_bank])
        rep_shifts = numpy.array([s[1] for s in shift_bank])

        cosine_sims = numpy.array(cosine_sims)

        # ── assign shift vectors to clean points via nearest rep neighbour ────
        knn_vis = NearestNeighbors(n_neighbors=1)
        knn_vis.fit(rep_embs)
        _, idx_clean = knn_vis.kneighbors(clean_embs)
        clean_cos = cosine_sims
        clean_shifts = rep_shifts[idx_clean[:, 0]]

        # ── per-point circular variance in original embedding space ───────────
        # unit vectors of assigned shifts
        shift_norms = numpy.linalg.norm(clean_shifts, axis=1, keepdims=True)
        shift_unit = clean_shifts / numpy.where(shift_norms > 1e-8, shift_norms, 1)

        K = min(8, len(clean_embs) - 1)
        knn_var = NearestNeighbors(n_neighbors=K)
        knn_var.fit(clean_embs)
        _, nn_idx = knn_var.kneighbors(clean_embs)

        circ_var = numpy.zeros(len(clean_embs))
        for i in range(len(clean_embs)):
            neighbour_units = shift_unit[nn_idx[i]]  # (K, n_dims)
            # mean resultant length generalised to n-D: norm of mean unit vector
            mean_vec = numpy.mean(neighbour_units, axis=0)
            circ_var[i] = 1.0 - numpy.linalg.norm(mean_vec)

        print(f"circ_var  min={circ_var.min():.3f}  max={circ_var.max():.3f}  mean={circ_var.mean():.3f}")
        print(f"cosine_sims  min={clean_cos.min():.3f}  max={clean_cos.max():.3f}  mean={clean_cos.mean():.3f}")

        # ── scatter: directional variance vs cosine similarity ────────────────
        fig, ax = plt.subplots(figsize=(7, 5))

        sc = ax.scatter(circ_var, clean_cos, c=clean_cos, cmap="RdYlGn", vmin=-1, vmax=1, s=18, alpha=0.6, edgecolors="none", zorder=2)

        # correlation line
        m, b = numpy.polyfit(circ_var, clean_cos, 1)
        x_line = numpy.linspace(circ_var.min(), circ_var.max(), 100)
        ax.plot(
            x_line,
            m * x_line + b,
            color="black",
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            zorder=3,
            label=f"Linear fit (r={numpy.corrcoef(circ_var, clean_cos)[0,1]:.2f})",
        )

        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)

        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Cosine similarity", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

        ax.set_xlabel("Local directional variance (circular, original embedding space)", fontsize=11)
        ax.set_ylabel("Cosine similarity (predicted vs Ground-Truth shift)", fontsize=11)
        ax.set_title(f"{backend} - Approximation quality vs local shift consistency", fontsize=12)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "shift_field_cosine_overlay.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: shift_field_cosine_overlay.png")

    def plot_shift_field_cosine_overlay(self, clean_embs, shift_bank, kmeans, backend, cosine_sims, external_knn_embs=None, external_knn_shifts=None):
        from umap import UMAP
        from sklearn.neighbors import NearestNeighbors
        from matplotlib.lines import Line2D
        import matplotlib.pyplot as plt

        rep_embs = numpy.array([s[0] for s in shift_bank])
        rep_shifts = numpy.array([s[1] for s in shift_bank])

        cosine_sims = numpy.array(cosine_sims)

        # ── assign cosine sim and shifts to clean points ───────────────────────
        knn_vis = NearestNeighbors(n_neighbors=1)
        knn_vis.fit(rep_embs)
        _, idx_clean = knn_vis.kneighbors(clean_embs)
        clean_cos = cosine_sims
        clean_shifts = rep_shifts[idx_clean[:, 0]]

        # ── circular variance in original embedding space ─────────────────────
        shift_norms = numpy.linalg.norm(clean_shifts, axis=1, keepdims=True)
        shift_unit = clean_shifts / numpy.where(shift_norms > 1e-8, shift_norms, 1)

        K = min(8, len(clean_embs) - 1)
        knn_var = NearestNeighbors(n_neighbors=K)
        knn_var.fit(clean_embs)
        _, nn_idx = knn_var.kneighbors(clean_embs)

        circ_var = numpy.zeros(len(clean_embs))
        for i in range(len(clean_embs)):
            neighbour_units = shift_unit[nn_idx[i]]
            mean_vec = numpy.mean(neighbour_units, axis=0)
            circ_var[i] = 1.0 - numpy.linalg.norm(mean_vec)

        r = numpy.corrcoef(circ_var, clean_cos)[0, 1]
        print(f"circ_var  min={circ_var.min():.3f}  max={circ_var.max():.3f}  mean={circ_var.mean():.3f}")
        print(f"r={r:.3f}")

        # ── UMAP projection ───────────────────────────────────────────────────
        print("Fitting UMAP...")
        proj_clean = UMAP(n_components=2, random_state=42).fit_transform(clean_embs)

        # ── figure: UMAP left, correlation right ──────────────────────────────
        fig, (ax_umap, ax_corr) = plt.subplots(1, 2, figsize=(14, 6))

        # left: UMAP coloured by cosine similarity
        sc_umap = ax_umap.scatter(
            proj_clean[:, 0],
            proj_clean[:, 1],
            c=clean_cos,
            cmap="RdYlGn",
            vmin=-1,
            vmax=1,
            s=25,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.3,
            zorder=2,
        )
        cbar_umap = fig.colorbar(sc_umap, ax=ax_umap, fraction=0.04, pad=0.02)
        cbar_umap.set_label("Cosine similarity", fontsize=10)
        cbar_umap.ax.tick_params(labelsize=9)
        ax_umap.set_xlabel("UMAP 1", fontsize=11)
        ax_umap.set_ylabel("UMAP 2", fontsize=11)
        ax_umap.set_title("Approximation quality in embedding space", fontsize=12)
        ax_umap.tick_params(labelsize=10)

        # right: directional variance vs cosine similarity
        sc_corr = ax_corr.scatter(circ_var, clean_cos, c=clean_cos, cmap="RdYlGn", vmin=-1, vmax=1, s=18, alpha=0.6, edgecolors="none", zorder=2)
        m, b = numpy.polyfit(circ_var, clean_cos, 1)
        x_line = numpy.linspace(circ_var.min(), circ_var.max(), 100)
        ax_corr.plot(x_line, m * x_line + b, color="black", linewidth=1.5, linestyle="--", alpha=0.7, zorder=3, label=f"Linear fit (r={r:.2f})")
        ax_corr.axhline(0, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
        ax_corr.set_xlabel("Local directional variance", fontsize=11)
        ax_corr.set_ylabel("Cosine similarity (predicted vs Ground-Truth shift)", fontsize=11)
        ax_corr.set_title("Approximation quality vs local shift consistency", fontsize=12)
        ax_corr.tick_params(labelsize=10)
        ax_corr.legend(fontsize=9)

        plt.suptitle(f"{backend}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "shift_field_cosine_overlay.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: shift_field_cosine_overlay.png")
