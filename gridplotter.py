import matplotlib.pyplot as plt
import numpy as np
import os

class GridPlotter:
    """Accumulates embedding plots during noise evaluation, renders as a grid at the end"""
    def __init__(self, train_labels, save_dir):
        self.train_labels = train_labels
        self.save_dir = save_dir
        self.panels = []  # list of (title, embeddings, labels, accuracy, csc)

    def add(self, title, embeddings, labels, accuracy, csc):
        self.panels.append((title, embeddings, labels, accuracy, csc))

    def render(self, filename='noise_grid.png'):
        n = len(self.panels)
        if n == 0:
            return
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        unique_labels = sorted(set(self.train_labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for ax, (title, embs, lbls, acc, csc) in zip(axes, self.panels):
            for label, color in zip(unique_labels, colors):
                mask = [i for i, l in enumerate(lbls) if l == label]
                ax.scatter(embs[mask, 0], embs[mask, 1],
                           label=f'Class {label}', color=color,
                           alpha=0.7, s=20, edgecolors='none')
            ax.set_title(f'{title}\nAcc: {acc:.1f}% | CSC: {csc:.3f}', fontsize=9)
            ax.set_xlabel('Z₀')
            ax.set_ylabel('Z₁')
            ax.legend(fontsize=7)

        # hide unused axes
        for ax in axes[n:]:
            ax.set_visible(False)

        plt.suptitle('Noise Profile Embedding Grid', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=120)
        plt.close()
        print(f"Grid saved: {filename} ({n} profiles)")