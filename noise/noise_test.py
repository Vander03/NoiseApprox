import sys
sys.path.append('..')
from model import Triplet
import triplet_generator
import numpy as np
import json, os, argparse
from sklearn.neighbors import NearestNeighbors
from visualiser import Visualiser

SAMPLES = 500


def print_noisy_eval_results(path):
    """print all saved noisy eval results for this run directory"""
    import glob
    result_files = glob.glob(os.path.join(path, '*noisy_eval_results.json'))
    
    if not result_files:
        print("No noisy eval results found.")
        return

    for result_file in sorted(result_files):
        backend = os.path.basename(result_file).replace('_noisy_eval_results.json', '').replace('noisy_eval_results.json', 'default')
        with open(result_file) as f:
            data = json.load(f)
        
        results = data.get('results', [])
        if not results:
            continue

        clean = next((r['accuracy'] for r in results if r['filename'] == 'clean'), None)
        noisy = [r for r in results if r['filename'] != 'clean']
        
        if not noisy:
            continue

        noisy_mean = np.mean([r['accuracy'] for r in noisy])
        noisy_min  = np.min([r['accuracy'] for r in noisy])
        noisy_max  = np.max([r['accuracy'] for r in noisy])
        drop       = clean - noisy_mean if clean else 0

        print(f"\n  [{backend}]")
        print(f"  Clean: {clean:.1f}% | Noisy Mean: {noisy_mean:.1f}% | Drop: {drop:.1f}% | Min: {noisy_min:.1f}% | Max: {noisy_max:.1f}%")
        for r in noisy:
            short = r['filename'].replace('hist_', '').replace('.json', '')
            print(f"    {short:<35} {r['accuracy']:.1f}%")


def load_model(path, backend=None):
    with open(os.path.join(path, "run_info.json")) as f:
        run_info = json.load(f)
    config = run_info["config"]
    config['shots'] = 1000
    if 'qubit' in config['backend']:
        config['backend'] = "qiskit.aer"
        config['sim'] = "density_matrix"
    
    if backend is not None:
        print(f"Changing Backend to: {backend}")
        config['backend_name'] = backend
    model = Triplet(config, testing=True, results_dir=path)
    best_path = os.path.join(path, 'best_weights.npy')
    if os.path.exists(best_path):
        model.weights = np.load(best_path, allow_pickle=True)
    else:
        weights = np.load(os.path.join(path, 'weights.npy'), allow_pickle=True)
        model.weights = weights[-1]

    return model, config


def load_triplets(config, samples=SAMPLES):
    triplets, labels = triplet_generator.generate_pca_triplets(
        dataset=config['dataset'],
        label_space=config['label_space'],
        num_triplets=config['num_triplets'],
        pca_dims=config['PCA_dims'],
        testing=False
    )
    t_triplets, t_labels = triplet_generator.generate_pca_triplets(
        dataset=config['dataset'],
        label_space=config['label_space'],
        num_triplets=config['num_triplets'],
        pca_dims=config['PCA_dims'],
        testing=True
    )
    return triplets[:samples], labels[:samples], t_triplets[:samples], t_labels[:samples]


def pca_gmm_baseline(config):
    from sklearn.mixture import GaussianMixture
    import itertools

    triplets, labels, t_triplets, t_labels = load_triplets(config)
    train_emb = np.array([t[0] for t in triplets])
    test_emb = np.array([t[0] for t in t_triplets])

    def permutation_accuracy(y_hat, labels, num_classes):
        perms = list(itertools.permutations(range(num_classes)))
        return 100 * max(
            sum(1 for i, j in zip([p[y] for y in y_hat], labels) if i == j)
            for p in perms
        ) / len(labels)

    num_classes = config['label_space']
    gmm = GaussianMixture(n_components=num_classes, random_state=42, n_init=10)
    gmm.fit(train_emb)
    train_acc = permutation_accuracy(gmm.predict(train_emb), [int(l) for l in labels], num_classes)
    test_acc = permutation_accuracy(gmm.predict(test_emb), [int(l) for l in t_labels], num_classes)

    print(f"PCA-only GMM baseline: Train {train_acc:.1f}% | Test {test_acc:.1f}%")
    return train_acc, test_acc


def compare_approximations(model, triplets, shift_bank):
    knn_embs = np.array([s[0] for s in shift_bank])
    knn_shifts = np.array([s[1] for s in shift_bank])
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(knn_embs)

    results = []
    test_indices = np.random.choice(len(triplets), min(20, len(triplets)), replace=False)

    for idx in test_indices:
        sample = triplets[idx][0]
        clean_emb = np.array([float(z) for z in model.qiskit_circuit(model, model.weights, np.array(sample))])
        prof = np.random.choice(model.np_train)
        noisy_emb = np.array([float(z) for z in prof["circuit"](model, model.weights, np.array(sample))])
        real_shift = noisy_emb - clean_emb
        gmm_shift = model.noise_gmm.sample(1)[0][0] if hasattr(model, 'noise_gmm') else np.zeros_like(clean_emb)
        _, idx_knn = knn.kneighbors(clean_emb.reshape(1, -1))
        knn_shift = knn_shifts[idx_knn[0][0]]
        results.append({'real': real_shift, 'gmm': gmm_shift, 'knn': knn_shift})

    print("\nShift approximation quality:")
    for name, key in [('GMM', 'gmm'), ('KNN', 'knn')]:
        shift_errors = [np.linalg.norm(r['real'] - r[key]) for r in results]
        mag_errors = [abs(np.linalg.norm(r['real']) - np.linalg.norm(r[key])) for r in results]
        cos_sims = [np.dot(r['real'], r[key]) / (np.linalg.norm(r['real']) * np.linalg.norm(r[key]) + 1e-8) for r in results]
        print(f"  {name}: shift_err={np.mean(shift_errors):.4f} | mag_err={np.mean(mag_errors):.4f} | cos_sim={np.mean(cos_sims):.4f}")

    return results


def analyse_model(path, baseline=False, variance=False, approximation=False, backend=None):
    print(f"\n=== Analysing: {path} ===")
    model, config = load_model(path, backend)
    triplets, labels, t_triplets, t_labels = load_triplets(config)


    if baseline:
        pca_gmm_baseline(config)

    print("\nEvaluating noise robustness...")
    results = model.predict_noisy_clustering(
        x_train=triplets,
        y_train=labels,
        x_test=t_triplets,
        y_test=t_labels,
    )

    # group profiles by backend
    backends = {}
    for prof in model.np_train:
        if prof['backend'] == 'ibm_kingston':
            if 'ibm_kingston' not in backends:
                backends['ibm_kingston'] = []
            backends['ibm_kingston'].append(prof)

    vis = Visualiser(model.results_dir)
    # vis.plot_shift_approximation_comparison(
    #     model=model,
    #     triplets=t_triplets,  # full test triplets, not sliced
    #     sample_indices=[10, 50, 100, 200, 400],  # safe indices within range
    #     backends=backends
    # )

    if variance:
        print("\nEvaluating clean variance...")
        model.predict_clustering_variance(x_test=t_triplets, y_test=t_labels, variance=4)

    if approximation and hasattr(model, 'np_train'):
        print("\nBuilding shift bank for approximation comparison...")
        _, shift_bank, _ = model.build_clustered_shift_bank(triplets)
        compare_approximations(model, triplets, shift_bank)

    print("\n=== Noisy Eval Results ===")
    print_noisy_eval_results(path)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='path to results directory')
    parser.add_argument('--baseline', action='store_true', help='run PCA GMM baseline')
    parser.add_argument('--variance', action='store_true', help='run clean variance evaluation')
    parser.add_argument('--approximation', action='store_true', help='compare GMM vs KNN shift approximation')
    parser.add_argument('--backend')
    args = parser.parse_args()

    analyse_model(args.path, baseline=args.baseline, variance=args.variance, approximation=args.approximation, backend=args.backend)