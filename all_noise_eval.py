import sys
sys.path.append('..')
from model import Triplet
import triplet_generator
import numpy as np
import json, os, argparse
from sklearn.neighbors import NearestNeighbors
from visualiser import Visualiser


def print_summary_table(base_paths):
    """print mean across seeds for each backend in each condition"""
    import glob
    from collections import defaultdict

    # structure: condition -> backend -> list of (clean, noisy_mean, drop)
    data = defaultdict(lambda: defaultdict(list))

    for base_path in base_paths:
        condition = os.path.basename(base_path)
        seed_dirs = sorted(glob.glob(os.path.join(base_path, 'seed*')))

        for seed_dir in seed_dirs:
            result_files = sorted(glob.glob(os.path.join(seed_dir, '*noisy_eval_results.json')))
            for result_file in result_files:
                backend_tag = os.path.basename(result_file).replace('_noisy_eval_results.json', '').replace('noisy_eval_results.json', 'default')
                with open(result_file) as f:
                    d = json.load(f)
                results = d.get('results', [])
                clean = next((r['accuracy'] for r in results if r['filename'] == 'clean'), None)
                noisy = [r['accuracy'] for r in results if r['filename'] != 'clean']
                if clean and noisy:
                    data[condition][backend_tag].append({
                        'clean': clean,
                        'noisy_mean': np.mean(noisy),
                        'drop': clean - np.mean(noisy)
                    })

    print(f"\n{'='*80}")
    print(f"  SUMMARY — Mean across seeds")
    print(f"{'='*80}")
    print(f"  {'Condition':<15} {'Backend':<15} {'N':>4} {'Clean':>8} {'Noisy':>8} {'Drop':>8} {'Std':>8}")
    print(f"  {'-'*70}")

    for condition in sorted(data.keys()):
        for backend in sorted(data[condition].keys()):
            entries     = data[condition][backend]
            n           = len(entries)
            clean_mean  = np.mean([e['clean']      for e in entries])
            noisy_mean  = np.mean([e['noisy_mean'] for e in entries])
            drop_mean   = np.mean([e['drop']       for e in entries])
            noisy_std   = np.std([e['noisy_mean']  for e in entries])
            print(f"  {condition:<15} {backend:<15} {n:>4} {clean_mean:>8.1f} {noisy_mean:>8.1f} {drop_mean:>8.1f} {noisy_std:>8.2f}")

    print(f"{'='*80}\n")


def print_all_results(base_paths):
    """iterate over all seed directories and print noisy eval results"""
    import glob

    for base_path in base_paths:
        seed_dirs = sorted(glob.glob(os.path.join(base_path, 'seed*')))
        if not seed_dirs:
            print(f"\nNo seed directories found in {base_path}")
            continue

        print(f"\n{'='*70}")
        print(f"  {base_path}")
        print(f"{'='*70}")

        for seed_dir in seed_dirs:
            seed_name = os.path.basename(seed_dir)
            result_files = sorted(glob.glob(os.path.join(seed_dir, '*noisy_eval_results.json')))

            if not result_files:
                continue

            print(f"\n  -- {seed_name} --")

            for result_file in result_files:
                backend_tag = os.path.basename(result_file).replace('_noisy_eval_results.json', '').replace('noisy_eval_results.json', 'default')

                with open(result_file) as f:
                    data = json.load(f)

                results = data.get('results', [])
                if not results:
                    continue

                clean  = next((r['accuracy'] for r in results if r['filename'] == 'clean'), None)
                noisy  = [r for r in results if r['filename'] != 'clean']

                if not noisy:
                    continue

                noisy_accs = [r['accuracy'] for r in noisy]
                noisy_mean = np.mean(noisy_accs)
                noisy_min  = np.min(noisy_accs)
                noisy_max  = np.max(noisy_accs)
                drop       = clean - noisy_mean if clean else 0

                print(f"    [{backend_tag}] Clean: {clean:.1f}%  Noisy Mean: {noisy_mean:.1f}%  Drop: {drop:.1f}%  Min: {noisy_min:.1f}%  Max: {noisy_max:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',   type=str, default=None, help='single run path to analyse')
    parser.add_argument('--scan',   type=str, nargs='+', default=None,
                        help='base directories to scan for seed subdirectories e.g. Results/NT/kingston Results/noiseless')
    parser.add_argument('--baseline',      action='store_true')
    parser.add_argument('--variance',      action='store_true')
    parser.add_argument('--approximation', action='store_true')
    parser.add_argument('--backend',       type=str, default=None)
    args = parser.parse_args()

    if args.scan:
        print_all_results(args.scan)
        print_summary_table(args.scan)

    
    # if args.path:
    #     analyse_model(
    #         args.path,
    #         baseline=args.baseline,
    #         variance=args.variance,
    #         approximation=args.approximation,
    #         backend=args.backend
    #     )