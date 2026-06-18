# silhouette_sweep.py
import sys
sys.path.append('..')
import argparse
import numpy
import json
import os
import glob
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from model import Triplet
import triplet_generator


def get_embeddings(model, triplets, labels, n_samples=300, noisy=False, profiles=None):
    embs = []
    labs = []
    samples = list(zip(triplets[:n_samples], labels[:n_samples]))
    for triplet, label in tqdm(samples, desc=f"{'Noisy' if noisy else 'Clean'} embeddings", leave=False):
        sample = triplet[0]
        if noisy and profiles:
            prof = profiles[numpy.random.randint(len(profiles))]
            emb = numpy.array([float(z) for z in prof["circuit"](model, model.weights, numpy.array(sample))])
        else:
            emb = numpy.array([float(z) for z in model.qiskit_circuit(model, model.weights, numpy.array(sample))])
        embs.append(emb)
        labs.append(label)
    return numpy.array(embs), numpy.array(labs)


def load_model(path, backend=None):
    with open(os.path.join(path, 'run_info.json')) as f:
        config = json.load(f)['config']
    config['shots'] = 1000
    if 'qubit' in config['backend']:
        config['backend'] = 'qiskit.aer'
        config['sim'] = 'density_matrix'
    if backend:
        config['backend_name'] = backend
    model = Triplet(config, testing=True, results_dir=path)
    best = os.path.join(path, 'best_weights.npy')
    model.weights = numpy.load(best if os.path.exists(best) else os.path.join(path, 'weights.npy'), allow_pickle=True)
    if model.weights.ndim == 3 and not os.path.exists(best):
        model.weights = model.weights[-1]
    return model, config


def score_seed(path, backend, n_samples=300):
    """returns (clean_score, noisy_score) for one seed directory"""
    model, config = load_model(path, backend)

    triplets, labels = triplet_generator.generate_pca_triplets(
        dataset=config['dataset'],
        label_space=config['label_space'],
        num_triplets=config['num_triplets'],
        pca_dims=config['PCA_dims'],
        testing=True
    )

    clean_embs, clean_labs = get_embeddings(model, triplets, labels, n_samples, noisy=False)
    noisy_embs, _          = get_embeddings(model, triplets, labels, n_samples, noisy=True, profiles=model.np_test)

    clean_score = silhouette_score(clean_embs, clean_labs)
    noisy_score = silhouette_score(noisy_embs, clean_labs)
    return clean_score, noisy_score


def sweep(nt_base, noiseless_base, backend, n_samples=300):
    """sweep all seeds for NT and noiseless models on a given backend"""
    nt_seed_dirs        = sorted(glob.glob(os.path.join(nt_base,        'seed*')))
    noiseless_seed_dirs = sorted(glob.glob(os.path.join(noiseless_base, 'seed*')))

    print(f"\n{'='*65}")
    print(f"  Backend: {backend}")
    print(f"{'='*65}")

    # ── noiseless model under no noise (clean reference) ─────────────
    nl_clean_scores = []
    for path in noiseless_seed_dirs:
        model, config = load_model(path, backend)
        triplets, labels = triplet_generator.generate_pca_triplets(
            dataset=config['dataset'], label_space=config['label_space'],
            num_triplets=config['num_triplets'], pca_dims=config['PCA_dims'], testing=True
        )
        embs, labs = get_embeddings(model, triplets, labels, n_samples, noisy=False)
        nl_clean_scores.append(silhouette_score(embs, labs))

    print(f"\n  Noiseless model — clean (no noise):  {numpy.mean(nl_clean_scores):.4f} ± {numpy.std(nl_clean_scores):.4f}")

    # ── NT model under no noise (clean reference) ─────────────────────
    nt_clean_scores = []
    for path in nt_seed_dirs:
        model, config = load_model(path, backend)
        triplets, labels = triplet_generator.generate_pca_triplets(
            dataset=config['dataset'], label_space=config['label_space'],
            num_triplets=config['num_triplets'], pca_dims=config['PCA_dims'], testing=True
        )
        embs, labs = get_embeddings(model, triplets, labels, n_samples, noisy=False)
        nt_clean_scores.append(silhouette_score(embs, labs))

    print(f"  NT model       — clean (no noise):  {numpy.mean(nt_clean_scores):.4f} ± {numpy.std(nt_clean_scores):.4f}")

    print(f"\n  {'Seed':<8} {'NL Clean':>10} {'NL Noisy':>10} {'NL Drop':>10} {'NT Clean':>10} {'NT Noisy':>10} {'NT Drop':>10}")
    print(f"  {'-'*68}")

    nl_noisy_scores = []
    nt_noisy_scores = []

    for i, (nl_path, nt_path) in enumerate(zip(noiseless_seed_dirs, nt_seed_dirs), 1):
        nl_clean, nl_noisy = score_seed(nl_path, backend, n_samples)
        nt_clean, nt_noisy = score_seed(nt_path, backend, n_samples)

        nl_noisy_scores.append(nl_noisy)
        nt_noisy_scores.append(nt_noisy)

        print(f"  {i:<8} {nl_clean:>10.4f} {nl_noisy:>10.4f} {nl_clean-nl_noisy:>10.4f} "
              f"{nt_clean:>10.4f} {nt_noisy:>10.4f} {nt_clean-nt_noisy:>10.4f}")

    print(f"  {'-'*68}")
    print(f"  {'Mean':<8} "
          f"{numpy.mean(nl_clean_scores):>10.4f} "
          f"{numpy.mean(nl_noisy_scores):>10.4f} "
          f"{numpy.mean(nl_clean_scores)-numpy.mean(nl_noisy_scores):>10.4f} "
          f"{numpy.mean(nt_clean_scores):>10.4f} "
          f"{numpy.mean(nt_noisy_scores):>10.4f} "
          f"{numpy.mean(nt_clean_scores)-numpy.mean(nt_noisy_scores):>10.4f}")
    print(f"  {'Std':<8} "
          f"{'':>10} "
          f"{numpy.std(nl_noisy_scores):>10.4f} "
          f"{'':>10} "
          f"{'':>10} "
          f"{numpy.std(nt_noisy_scores):>10.4f} "
          f"{'':>10}")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nt',        type=str, required=True, help='base path for NT runs e.g. Results/NT/kingston')
    parser.add_argument('--noiseless', type=str, required=True, help='base path for noiseless runs e.g. Results/noiseless')
    parser.add_argument('--backend',   type=str, required=True, help='backend name e.g. kingston')
    parser.add_argument('--n_samples', type=int, default=300)
    args = parser.parse_args()

    sweep(args.nt, args.noiseless, args.backend, args.n_samples)