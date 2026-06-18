# shift_magnitude_analysis.py
import sys
sys.path.append('..')
import argparse
import numpy
import os
import json
from tqdm import tqdm


def analyse_shift_bank_magnitudes(paths, backend_names):
    """load knn_shifts.npy for each path and report shift magnitude statistics"""
    print(f"\n{'='*65}")
    print(f"{'Backend':<15} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"{'='*65}")

    results = []
    for path, backend in zip(paths, backend_names):
        shifts_path = os.path.join(path, 'knn_shifts.npy')
        if not os.path.exists(shifts_path):
            print(f"{backend:<15} {'NOT FOUND':>8}")
            continue

        shifts     = numpy.load(shifts_path, allow_pickle=True)
        magnitudes = numpy.linalg.norm(shifts, axis=1)

        mean   = numpy.mean(magnitudes)
        median = numpy.median(magnitudes)
        std    = numpy.std(magnitudes)
        mn     = numpy.min(magnitudes)
        mx     = numpy.max(magnitudes)

        print(f"{backend:<15} {mean:>8.4f} {median:>8.4f} {std:>8.4f} {mn:>8.4f} {mx:>8.4f}")
        results.append({
            'backend': backend,
            'mean':    float(mean),
            'median':  float(median),
            'std':     float(std),
            'min':     float(mn),
            'max':     float(mx)
        })

    print(f"{'='*65}\n")
    return results


def analyse_holdout_magnitudes(backend_name, run_path, n_samples=100):
    """measure actual shift magnitudes from holdout profiles on test samples"""
    from model import Triplet
    import triplet_generator
    import json as json_mod

    with open(os.path.join(run_path, 'run_info.json')) as f:
        config = json_mod.load(f)["config"]

    model = Triplet(config, testing=True, results_dir=run_path)
    model.weights = numpy.load("noiseless_trained_6_fashion.npy", allow_pickle=True)

    triplets, _ = triplet_generator.generate_pca_triplets(
        dataset=config['dataset'],
        label_space=config['label_space'],
        num_triplets=2000,
        pca_dims=config['PCA_dims'],
        testing=True
    )

    samples = triplets[:n_samples]
    noiseless_weights = model.weights

    print(f"Measuring holdout shift magnitudes for {backend_name}...")
    all_magnitudes = []
    for t in tqdm(samples):
        sample    = t[0]
        clean_emb = numpy.array([float(z) for z in model.qiskit_circuit(
            model, noiseless_weights, numpy.array(sample))])
        for prof in model.np_test:
            noisy_emb = numpy.array([float(z) for z in prof["circuit"](
                model, noiseless_weights, numpy.array(sample))])
            shift = noisy_emb - clean_emb
            all_magnitudes.append(numpy.linalg.norm(shift))

    mags = numpy.array(all_magnitudes)
    result = {
        'backend': backend_name,
        'mean':    float(numpy.mean(mags)),
        'median':  float(numpy.median(mags)),
        'std':     float(numpy.std(mags)),
        'min':     float(numpy.min(mags)),
        'max':     float(numpy.max(mags)),
    }
    print(f"{backend_name:<15} mean={result['mean']:.4f}  "
          f"median={result['median']:.4f}  "
          f"std={result['std']:.4f}  "
          f"min={result['min']:.4f}  "
          f"max={result['max']:.4f}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths',    type=str, nargs='+', required=True,
                        help='paths to run directories containing knn_shifts.npy')
    parser.add_argument('--backends', type=str, nargs='+', required=True,
                        help='backend names corresponding to each path')
    parser.add_argument('--holdout',  action='store_true',
                        help='also measure holdout profile shift magnitudes')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='number of test samples to use for holdout magnitude estimation')
    args = parser.parse_args()

    if len(args.paths) != len(args.backends):
        print("ERROR: number of paths must match number of backend names")
        sys.exit(1)

    print("\n=== SHIFT BANK MAGNITUDES ===")
    bank_results = analyse_shift_bank_magnitudes(args.paths, args.backends)

    holdout_results = []
    if args.holdout:
        print("\n=== HOLDOUT PROFILE MAGNITUDES ===")
        print(f"\n{'='*65}")
        print(f"{'Backend':<15} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"{'='*65}")
        for path, backend in zip(args.paths, args.backends):
            result = analyse_holdout_magnitudes(backend, path, args.n_samples)
            holdout_results.append(result)
        print(f"{'='*65}")

    # comparison table
    if holdout_results:
        print("\n=== COMPARISON: BANK vs HOLDOUT ===")
        print(f"\n{'Backend':<15} {'Bank Mean':>12} {'Holdout Mean':>14} {'Ratio':>8}")
        print(f"{'-'*55}")
        for bank, holdout in zip(bank_results, holdout_results):
            ratio = holdout['mean'] / bank['mean'] if bank['mean'] > 1e-8 else 0
            print(f"{bank['backend']:<15} {bank['mean']:>12.4f} {holdout['mean']:>14.4f} {ratio:>8.3f}")

    # save
    summary = {'bank': bank_results, 'holdout': holdout_results}
    with open('shift_magnitude_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved: shift_magnitude_summary.json")