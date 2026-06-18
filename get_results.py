# get_results.py
import sys
sys.path.append('..')
import argparse
import numpy
from model import Triplet
from visualiser import Visualiser
import triplet_generator
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pennylane as qml
import json, os


def evaluate_test_approximation(model, test_triplets, knn_embs, knn_shifts, n_samples=500):
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(knn_embs)

    test_embs        = []
    predicted_shifts = []
    samples          = test_triplets[:n_samples]

    print("Getting test embeddings...")
    for t in tqdm(samples):
        emb = numpy.array([float(z) for z in model.qiskit_circuit(model, model.weights, numpy.array(t[0]))])
        test_embs.append(emb)
        _, idx = knn.kneighbors(emb.reshape(1, -1))
        predicted_shifts.append(knn_shifts[idx[0][0]])

    test_embs        = numpy.array(test_embs)
    predicted_shifts = numpy.array(predicted_shifts)

    print("Getting actual noisy shifts...")
    actual_shifts = []
    for i, t in enumerate(tqdm(samples)):
        sample    = t[0]
        clean_emb = test_embs[i]
        shifts    = []
        for prof in numpy.random.choice(model.np_train, min(5, len(model.np_train)), replace=False):
            noisy_emb = numpy.array([float(z) for z in prof["circuit"](model, model.weights, numpy.array(sample))])
            shifts.append(noisy_emb - clean_emb)
        actual_shifts.append(numpy.mean(shifts, axis=0))

    actual_shifts = numpy.array(actual_shifts)

    cosine_sims = []
    for pred, actual in zip(predicted_shifts, actual_shifts):
        norm    = numpy.linalg.norm(pred) * numpy.linalg.norm(actual)
        cos_sim = numpy.dot(pred, actual) / norm if norm > 1e-8 else 0.0
        cosine_sims.append(cos_sim)

    cosine_sims = numpy.array(cosine_sims)
    print(f"Test set — Mean: {numpy.mean(cosine_sims):.4f} | Median: {numpy.median(cosine_sims):.4f} | Std: {numpy.std(cosine_sims):.4f}")
    return test_embs, actual_shifts, predicted_shifts, cosine_sims


def analyse(seed_path, dataset="fashionMNIST", label_space=3, trained_weights_path=None):
    with open(os.path.join(seed_path, "run_info.json")) as f:
        config = json.load(f)["config"]

    backend_name = config.get("backend_name", "unknown")

    backend_dir    = os.path.dirname(seed_path)
    shift_bank_dir = os.path.join(backend_dir, "shift_bank")

    if trained_weights_path is not None:
        plots_dir = os.path.join(backend_dir, "trained_validation")
    else:
        plots_dir = os.path.join(backend_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Plots dir: {plots_dir}")
    print(f"Seed path: {seed_path}")
    print(f"Backend dir: {backend_dir}")

    knn_shifts = numpy.load(os.path.join(shift_bank_dir, "knn_shifts.npy"), allow_pickle=True)
    knn_embs   = numpy.load(os.path.join(shift_bank_dir, "knn_embs.npy"),   allow_pickle=True)

    model = Triplet(config, testing=True, results_dir=seed_path)

    if trained_weights_path is not None:
        model.weights = numpy.load(trained_weights_path, allow_pickle=True)
        print(f"Using trained weights: {trained_weights_path}")
    else:
        model.weights = numpy.load("/Users/schalk/Desktop/QUT/EGH400/SliQ/noiseless_trained_6_fashion.npy", allow_pickle=True)
        print("Using noiseless reference weights")

    model.knn_shifts = knn_shifts
    model.knn        = NearestNeighbors(n_neighbors=1)
    model.knn.fit(knn_embs)

    kmeans = KMeans(n_clusters=10, random_state=42)
    kmeans.fit(knn_embs)
    shift_bank = list(zip(knn_embs, knn_shifts))

    vis = Visualiser(plots_dir)

    test_triplets, test_labels = triplet_generator.generate_pca_triplets(
        dataset=config["dataset"], label_space=config["label_space"],
        num_triplets=5000, pca_dims=config["PCA_dims"], testing=True
    )

    # circuit diagram — save to seed dir
    from pennylane import numpy as np
    sample = test_triplets[0][0]
    fig, ax = qml.draw_mpl(model.circuit, decimals=None, fontsize=9)(model, model.weights, np.array(sample))
    fig.set_size_inches(24, 4)
    fig.savefig(os.path.join(seed_path, "circuit.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: circuit.png")

    # approximation quality
    test_embs, actual_shifts, predicted_shifts, cosine_sims = evaluate_test_approximation(
        model, test_triplets, knn_embs, knn_shifts
    )

    test_shift_bank = list(zip(test_embs, actual_shifts))
    test_kmeans     = KMeans(n_clusters=10, random_state=42)
    test_kmeans.fit(test_embs)

    vis.plot_shift_bank(test_embs, test_shift_bank, backend=backend_name, kmeans=test_kmeans)

    # holdout shifts for compass
    print("Computing holdout shifts for compass...")
    holdout_shifts = []
    for t in tqdm(test_triplets[:200]):
        sample    = t[0]
        clean_emb = numpy.array([float(z) for z in model.qiskit_circuit(
            model, model.weights, numpy.array(sample))])
        for prof in model.np_test:
            noisy_emb = numpy.array([float(z) for z in prof["circuit"](
                model, model.weights, numpy.array(sample))])
            holdout_shifts.append(noisy_emb - clean_emb)
    holdout_shifts = numpy.array(holdout_shifts)

    vis.plot_shift_approximation_quality(
        test_embs,
        list(zip(test_embs, actual_shifts)),
        kmeans=kmeans,
        backend=backend_name,
        predicted_shifts=predicted_shifts,
        cosine_sims=cosine_sims,
        holdout_shifts=holdout_shifts
    )

    vis.plot_shift_field_cosine_overlay(
        test_embs,
        test_shift_bank,
        test_kmeans,
        backend=backend_name,
        cosine_sims=cosine_sims
    )

    numpy.save(os.path.join(plots_dir, f"{backend_name}_cosine_sims.npy"), cosine_sims)
    print(f"All plots saved to: {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",            type=str, required=True,  help="path to seed directory e.g. Results/NT_v3/kingston/seed1")
    parser.add_argument("--dataset",         type=str, default="fashionMNIST")
    parser.add_argument("--label_space",     type=int, default=3)
    parser.add_argument("--trained_weights", type=str, default=None,   help="path to trained weights .npy file, enables trained_validation mode")
    args = parser.parse_args()
    analyse(args.path, args.dataset, args.label_space, args.trained_weights)