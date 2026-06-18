# sanity_check.py
import sys
sys.path.append('..')
import numpy
import json
import os
import argparse
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from tqdm import tqdm
from model import Triplet
import triplet_generator


def cosine_sim(a, b):
    norm = numpy.linalg.norm(a) * numpy.linalg.norm(b)
    return numpy.dot(a, b) / norm if norm > 1e-8 else 0.0


def get_test_embeddings(model, test_triplets, n_samples=200):
    print("Computing clean test embeddings...")
    test_embs = []
    for t in tqdm(test_triplets[:n_samples]):
        emb = numpy.array([float(z) for z in model.qiskit_circuit(model, model.weights, numpy.array(t[0]))])
        test_embs.append(emb)
    return numpy.array(test_embs)


def get_actual_shifts(model, test_triplets, test_embs, n_samples=200):
    print("Computing actual noisy shifts...")
    actual_shifts = []
    for i, t in enumerate(tqdm(test_triplets[:n_samples])):
        sample    = t[0]
        clean_emb = test_embs[i]
        shifts    = []
        for prof in numpy.random.choice(model.np_train, min(5, len(model.np_train)), replace=False):
            noisy_emb = numpy.array([float(z) for z in prof["circuit"](model, model.weights, numpy.array(sample))])
            shifts.append(noisy_emb - clean_emb)
        actual_shifts.append(numpy.mean(shifts, axis=0))
    return numpy.array(actual_shifts)


def predict_shifts(test_embs, knn_embs, knn_shifts):
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(knn_embs)
    predicted = []
    for emb in test_embs:
        _, idx = knn.kneighbors(emb.reshape(1, -1))
        predicted.append(knn_shifts[idx[0][0]])
    return numpy.array(predicted)


def mean_cosine(predicted, actual):
    sims = [cosine_sim(p, a) for p, a in zip(predicted, actual)]
    return float(numpy.mean(sims))


def run_checks(path, n_samples=200):
    print(f"\n{'='*60}")
    print(f"SANITY CHECK: {path}")
    print(f"{'='*60}\n")

    with open(os.path.join(path, "run_info.json")) as f:
        config = json.load(f)["config"]

    model = Triplet(config, testing=True, results_dir=path)
    model.weights = numpy.load("/Users/schalk/Desktop/QUT/EGH400/SliQ/noiseless_trained_6_fashion.npy", allow_pickle=True)

    knn_shifts = numpy.load(os.path.join(path, "knn_shifts.npy"), allow_pickle=True)
    knn_embs   = numpy.load(os.path.join(path, "knn_embs.npy"),   allow_pickle=True)

    test_triplets, _ = triplet_generator.generate_pca_triplets(
        dataset=config["dataset"], label_space=config["label_space"],
        num_triplets=5000, pca_dims=config["PCA_dims"], testing=True
    )

    # get test embeddings and actual shifts — computed once, reused for all checks
    test_embs     = get_test_embeddings(model, test_triplets, n_samples)
    actual_shifts = get_actual_shifts(model, test_triplets, test_embs, n_samples)

    print(f"\n--- STEP 1: verify predicted and actual are different ---")
    predicted_real = predict_shifts(test_embs, knn_embs, knn_shifts)
    print(f"predicted_shifts mean: {predicted_real.mean():.6f}")
    print(f"actual_shifts mean:    {actual_shifts.mean():.6f}")
    print(f"are they identical?    {numpy.allclose(predicted_real, actual_shifts)}")
    print(f"mean absolute diff:    {numpy.abs(predicted_real - actual_shifts).mean():.6f}")

    print(f"\n--- STEP 2: real shift bank (your method) ---")
    cos_real = mean_cosine(predicted_real, actual_shifts)
    print(f"mean cosine similarity: {cos_real:.4f}  <-- this should be your reported number")

    print(f"\n--- STEP 3: random shift bank ---")
    random_shifts = numpy.random.randn(*knn_shifts.shape)
    predicted_random = predict_shifts(test_embs, knn_embs, random_shifts)
    cos_random = mean_cosine(predicted_random, actual_shifts)
    print(f"mean cosine similarity: {cos_random:.4f}  <-- should be near 0")

    print(f"\n--- STEP 4: zero shift bank ---")
    zero_shifts = numpy.zeros_like(knn_shifts)
    predicted_zero = predict_shifts(test_embs, knn_embs, zero_shifts)
    cos_zero = mean_cosine(predicted_zero, actual_shifts)
    print(f"mean cosine similarity: {cos_zero:.4f}  <-- should be near 0 or undefined")

    print(f"\n--- STEP 5: shuffled shift bank (breaks position-dependence) ---")
    shuffled_shifts = knn_shifts.copy()
    numpy.random.shuffle(shuffled_shifts)
    predicted_shuffled = predict_shifts(test_embs, knn_embs, shuffled_shifts)
    cos_shuffled = mean_cosine(predicted_shuffled, actual_shifts)
    print(f"mean cosine similarity: {cos_shuffled:.4f}  <-- should be lower than real")

    print(f"\n--- STEP 6: global mean shift (position-independent baseline) ---")
    global_mean_shift = numpy.mean(knn_shifts, axis=0)
    predicted_global  = numpy.tile(global_mean_shift, (len(test_embs), 1))
    cos_global = mean_cosine(predicted_global, actual_shifts)
    print(f"mean cosine similarity: {cos_global:.4f}  <-- position-independent baseline")

    print(f"\n--- SUMMARY ---")
    print(f"Real bank:      {cos_real:.4f}")
    print(f"Global mean:    {cos_global:.4f}")
    print(f"Shuffled bank:  {cos_shuffled:.4f}")
    print(f"Random bank:    {cos_random:.4f}")
    print(f"Zero bank:      {cos_zero:.4f}")
    print(f"\nIf real > shuffled > random ≈ 0, your method is working correctly.")
    print(f"If real ≈ shuffled, position-dependence is not the driver.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=200)
    args = parser.parse_args()
    run_checks(args.path, args.n_samples)