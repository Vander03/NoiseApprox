# add_qiskit_clean.py
# adds clean qiskit density matrix accuracy to noisy_eval_results.json
# run this on a results directory after noise eval has been completed

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Triplet
import triplet_generator
import numpy as np
import json, os

samples = 500

def add_qiskit_clean(path):
    noisy_path = os.path.join(path, "noisy_eval_results.json")
    run_info_path = os.path.join(path, "run_info.json")

    if not os.path.exists(noisy_path) or not os.path.exists(run_info_path):
        print(f"Skipping {path} — missing files")
        return

    with open(run_info_path) as f:
        run_info = json.load(f)
    config = run_info["config"]
    if "seed" not in config["message"]: return

    # check if already has qiskit_clean
    with open(noisy_path) as f:
        noisy_data = json.load(f)
    if "qiskit_clean" in noisy_data:
        print(f"Already has qiskit_clean: {path}")
        return

    # reconstruct on qiskit density matrix
    config['shots'] = 1000
    config['backend'] = "qiskit.aer"
    config['sim'] = "density_matrix"

    model = Triplet(config, testing=True, results_dir=path)
    weights = np.load(os.path.join(path, 'weights.npy'), allow_pickle=True)
    model.weights = weights[-1]

    t_triplets, t_labels = triplet_generator.generate_pca_triplets(
        dataset=config['dataset'],
        label_space=config['label_space'],
        num_triplets=config['num_triplets'],
        pca_dims=config['PCA_dims'],
        testing=True
    )

    # get clean qiskit embeddings and evaluate against saved GMM
    print(f"Running clean Qiskit eval: {os.path.basename(path)}")
    clean_emb = np.array([
        [float(z) for z in model.circuit(model, model.weights, np.array(im[0]))]
        for im in t_triplets[:samples]
    ])

    # refit GMM on qiskit clean embeddings
    qiskit_clean_acc = model.evaluate_embeddings(
        clean_emb, [int(l) for l in t_labels[:samples]]
    )

    # also get pennylane clean accuracy from run_info
    pennylane_test_acc = None
    results = run_info.get("results", {})
    gmm_test = results.get("gmm_accuracy_test", [None, None])
    if isinstance(gmm_test, list) and len(gmm_test) > 1:
        pennylane_test_acc = gmm_test[1]

    # noisy mean from existing results
    noisy_results = noisy_data.get("results", [])
    noisy_mean = np.mean([r['accuracy'] for r in noisy_results]) if noisy_results else None

    # compute gap
    gap = None
    if pennylane_test_acc and qiskit_clean_acc:
        gap = round(qiskit_clean_acc - pennylane_test_acc, 2)

    print(f"  Pennylane test: {pennylane_test_acc}%")
    print(f"  Qiskit clean:   {qiskit_clean_acc:.1f}%")
    print(f"  Gap (Q-P):      {gap}%")
    print(f"  Noisy mean:     {noisy_mean:.2f}%" if noisy_mean else "  Noisy mean: N/A")

    # save back
    noisy_data['qiskit_clean'] = round(qiskit_clean_acc, 2)
    noisy_data['pennylane_test'] = pennylane_test_acc
    noisy_data['qiskit_pennylane_gap'] = gap
    noisy_data['noisy_mean'] = round(noisy_mean, 2) if noisy_mean else None

    with open(noisy_path, 'w') as f:
        json.dump(noisy_data, f, indent=4)

    print(f"  Saved to {noisy_path}\n")
    return {
        'path': path,
        'pennylane_test': pennylane_test_acc,
        'qiskit_clean': round(qiskit_clean_acc, 2),
        'gap': gap,
        'noisy_mean': round(noisy_mean, 2) if noisy_mean else None
    }


def summarise_correlation(all_stats):
    """print correlation between qiskit-pennylane gap and noisy mean"""
    valid = [s for s in all_stats if s and s['gap'] is not None and s['noisy_mean'] is not None]
    if len(valid) < 2:
        print("Not enough data for correlation")
        return

    gaps = np.array([s['gap'] for s in valid])
    noisy = np.array([s['noisy_mean'] for s in valid])
    corr = np.corrcoef(gaps, noisy)[0, 1]

    print(f"\n{'Path':<60} {'Gap':>8} {'Noisy mean':>12}")
    print("-" * 82)
    for s in sorted(valid, key=lambda x: x['gap']):
        print(f"{os.path.basename(s['path']):<60} {s['gap']:>8.2f} {s['noisy_mean']:>12.2f}")

    print(f"\nCorrelation (gap vs noisy mean): {corr:.4f}")
    if corr > 0.5:
        print("Strong positive — models that transfer better to Qiskit are more noise robust")
    elif corr > 0.2:
        print("Weak positive — some tendency for better Qiskit transfer = better noise robustness")
    else:
        print("No clear correlation")


if __name__ == '__main__':
    # run on all seed results
    all_stats = []
    for date_dir in sorted(os.listdir("Results")):
        date_path = os.path.join("Results", date_dir)
        if not os.path.isdir(date_path):
            continue
        for run_dir in sorted(os.listdir(date_path)):
            run_path = os.path.join(date_path, run_dir)
            if not os.path.isdir(run_path):
                continue
            # only process seed runs
            run_info_path = os.path.join(run_path, "run_info.json")
            if not os.path.exists(run_info_path):
                continue
            with open(run_info_path) as f:
                info = json.load(f)
            if "seed" not in info["config"].get("message", ""):
                continue
            stats = add_qiskit_clean(run_path)
            if stats:
                all_stats.append(stats)

    summarise_correlation(all_stats)