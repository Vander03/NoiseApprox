import sys
sys.path.append('..')
from model import Triplet
import triplet_generator
import numpy as np
import json, os, argparse
import pickle

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import numpy
import tqdm

def build_clustered_shift_bank(model, triplets, n_clusters=10, samples_per_cluster=5, n_noise_samples=10):
    """get clean embeddings for all triplets, cluster them, select representatives, measure real shifts"""
    
    # get clean embeddings for all training samples
    print("Getting clean embeddings...")
    clean_embs = []
    for t in tqdm.tqdm(triplets):
        emb = numpy.array([float(z) for z in model.qiskit_circuit(model, model.weights, numpy.array(t[0]))])
        clean_embs.append(emb)
    clean_embs = numpy.array(clean_embs)
    
    # cluster clean embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(clean_embs)
    
    # select representative samples per cluster
    calibration_indices = []
    for k in range(n_clusters):
        cluster_mask = kmeans.labels_ == k
        cluster_indices = numpy.where(cluster_mask)[0]
        selected = cluster_indices[:samples_per_cluster]
        calibration_indices.extend(selected)
    
    # measure real shifts for calibration samples
    shift_bank = []  # list of (clean_emb, shift_vector) tuples
    for idx in calibration_indices:
        sample = triplets[idx][0]
        clean_emb = clean_embs[idx]
        prof = numpy.random.choice(model.np_train)
        noisy_emb = numpy.array([float(z) for z in prof["circuit"](model, model.weights, numpy.array(sample))])
        shift = noisy_emb - clean_emb
        shift_bank.append((clean_emb, shift))
    
    return clean_embs, shift_bank, kmeans


def compare_approximations(model, triplets, shift_bank, clean_embs):
    """compare global GMM vs KNN shift approximation against real noisy shifts"""
    
    knn_embs = numpy.array([s[0] for s in shift_bank])
    knn_shifts = numpy.array([s[1] for s in shift_bank])
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(knn_embs)
    
    results = []
    test_indices = numpy.random.choice(len(triplets), 20, replace=False)
    
    for idx in test_indices:
        sample = triplets[idx][0]
        clean_emb = clean_embs[idx]
        
        # real shift from actual noisy circuit
        prof = numpy.random.choice(model.np_train)
        noisy_emb = numpy.array([float(z) for z in prof["circuit"](model, model.weights, numpy.array(sample))])
        real_shift = noisy_emb - clean_emb
        
        # global GMM approximation
        gmm_shift = model.sample_noise()
        
        # KNN approximation
        _, neighbour_idx = knn.kneighbors(clean_emb.reshape(1, -1))
        knn_shift = knn_shifts[neighbour_idx[0][0]]
        
        results.append({
            'real': real_shift,
            'gmm': gmm_shift,
            'knn': knn_shift
        })
    
    # compute metrics
    for name, key in [('GMM', 'gmm'), ('KNN', 'knn')]:
        shift_errors = [numpy.linalg.norm(r['real'] - r[key]) for r in results]
        mag_errors = [abs(numpy.linalg.norm(r['real']) - numpy.linalg.norm(r[key])) for r in results]
        cos_sims = [numpy.dot(r['real'], r[key]) / (numpy.linalg.norm(r['real']) * numpy.linalg.norm(r[key]) + 1e-8) for r in results]
        
        print(f"\n{name} Approximation:")
        print(f"  Shift error (L2):     {numpy.mean(shift_errors):.4f} ± {numpy.std(shift_errors):.4f}")
        print(f"  Magnitude error:      {numpy.mean(mag_errors):.4f} ± {numpy.std(mag_errors):.4f}")
        print(f"  Cosine similarity:    {numpy.mean(cos_sims):.4f} ± {numpy.std(cos_sims):.4f}")
    
    return results

def analyse_model(path):
    with open(os.path.join(path, "run_info.json")) as f:
        run_info = json.load(f)
    config = run_info["config"]
    config['shots'] = 1000

    if 'qubit' in config['backend']:
        config['backend'] = "qiskit.aer"
        config['sim'] = "density_matrix"

    model = Triplet(config, testing=False, results_dir=path)
    # try best weights first, fall back to last
    best_path = os.path.join(path, 'best_weights.npy')
    if os.path.exists(best_path):
        model.weights = np.load(best_path, allow_pickle=True)
    else:
        weights = np.load(os.path.join(path, 'weights.npy'), allow_pickle=True)
        model.weights = weights[-1]
    
    triplets, labels = triplet_generator.generate_pca_triplets(
        dataset=config['dataset'],
        label_space=config['label_space'],
        num_triplets=config['num_triplets'],
        pca_dims=config['PCA_dims'],
        testing=False
    )
    
    with open(os.path.join(path, 'noise_gmm.pkl'), 'rb') as f:
        model.noise_gmm = pickle.load(f)
    model.ss_samples = [63, 550, 1755, 2633, 2653, 3444, 4518]
    clean_embs, shift_bank, kmeans = build_clustered_shift_bank(model=model, triplets=triplets)
    results = compare_approximations(model=model, triplets=triplets, shift_bank=shift_bank, clean_embs=clean_embs)
    return

path = "Results/2026-05-21/15-45-14__NT1_e150_shotsNone_lr0.1_cNone_histTrue__MNIST_l3"
analyse_model("Results/2026-05-21/15-45-14__NT1_e150_shotsNone_lr0.1_cNone_histTrue__MNIST_l3")
