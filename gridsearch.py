# gridsearch.py
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import triplet_generator
import model
import numpy
import json
from itertools import product
from tqdm import tqdm
import datetime

pennylane = "lightning.qubit"

GRID = {
    "learning_rate":     [0.05, 0.1, 0.2, 0.5],
    "perturbation_rate": [0.01, 0.05, 0.1, 0.2],
}

BASE_PARAMS = {
    "dataset":            "MNIST",
    "epochs":             100,          # shorter for grid search
    "num_qubits":         5,
    "backend":            pennylane,
    "shots":              None,
    "num_triplets":       1000,          # smaller for speed
    "label_space":        3,            # 2-class for speed
    "layers":             4,
    "batch_size":         32,
    "max_train_samples":  500,
    "embed_dims":         4,
    "optimiser":          "SPSA",
    "noise_train":        False,
    "noise_samp_per_batch": 2,
    "historic_load":      10,
    "fake":               False,
    "message":            "gridsearch",
    "noise_profiles":     [],
    "holdout_profiles":   [],
    "results":            {},
    "PCA_dims":           16
}

def run_combination(lr, cr, pca):
    params = {**BASE_PARAMS, 
              "learning_rate": lr, 
              "perturbation_rate": cr, 
              "results": {}}

    triplets, labels = triplet_generator.generate_pca_triplets(
        params['dataset'],
        label_space=params['label_space'],
        num_triplets=params['num_triplets'],
        pca_dims=params['PCA_dims'],
        testing=False
    )
    t_triplets, t_labels = triplet_generator.generate_pca_triplets(
        params['dataset'],
        label_space=params['label_space'],
        num_triplets=params['num_triplets'],
        pca_dims=params['PCA_dims'],
        testing=True
    )

    network = model.Triplet(params)
    network.train(triplets)

    train_emb = network.get_embeddings(triplets, network.circuit)
    test_emb  = network.get_embeddings(t_triplets, network.circuit)
    
    network._ensure_results_dir()
    gmm_train = network.evaluate_embeddings(train_emb, [int(l) for l in labels])
    gmm_test  = network.evaluate_embeddings_test(test_emb, [int(l) for l in t_labels])
    final_loss = float(network.loss_history[-1]) if network.loss_history else None
    min_loss   = float(min(network.loss_history)) if network.loss_history else None

    return {
        "learning_rate":     lr,
        "perturbation_rate": cr,
        "PCA_dims":          pca,
        "gmm_train":         gmm_train,
        "gmm_test":          gmm_test,
        "final_loss":        final_loss,
        "min_loss":          min_loss,
        "results_dir":       network.results_dir
    }

if __name__ == '__main__':
    keys   = list(GRID.keys())
    values = list(GRID.values())
    combos = list(product(*values))

    print(f"Running {len(combos)} combinations...")
    results = []

    for combo in tqdm(combos, desc="Grid search"):
        lr, cr, pca = combo
        print(f"\n→ lr={lr} cr={cr} pca={pca}")
        try:
            result = run_combination(lr, cr, pca)
            results.append(result)
            print(f"  train={result['gmm_train']:.1f}% test={result['gmm_test']:.1f}% loss={result['min_loss']:.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"learning_rate": lr, "perturbation_rate": cr, "PCA_dims": pca, "error": str(e)})

    # save all results
    _now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_path = f"Results/gridsearch_{_now}.json"
    os.makedirs("Results", exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # print summary sorted by test accuracy
    valid = [r for r in results if 'gmm_test' in r]
    valid.sort(key=lambda x: x['gmm_test'], reverse=True)
    print(f"\n{'='*60}")
    print(f"TOP 5 RESULTS (by test GMM accuracy):")
    for r in valid[:5]:
        print(f"  lr={r['learning_rate']} cr={r['perturbation_rate']} pca={r['PCA_dims']} "
              f"→ train={r['gmm_train']:.1f}% test={r['gmm_test']:.1f}% loss={r['min_loss']:.4f}")
    print(f"\nFull results saved to {out_path}")