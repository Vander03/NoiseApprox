# add_mean_to_noisy_eval.py
import json
import os
import sys
import numpy as np

def add_mean(results_dir):
    path = os.path.join(results_dir, 'noisy_eval_results.json')
    if not os.path.exists(path):
        print(f"No noisy_eval_results.json found in {results_dir}")
        return

    with open(path) as f:
        data = json.load(f)

    results = data.get('results', data)
    if not isinstance(results, list):
        print(f"Unexpected format in {results_dir}")
        return

    accs = [r['accuracy'] for r in results if 'accuracy' in r]
    if not accs:
        print(f"No accuracy values found in {results_dir}")
        return

    summary = {
        'mean': round(float(np.mean(accs)), 2),
        'std': round(float(np.std(accs)), 2),
        'min': round(float(np.min(accs)), 2),
        'max': round(float(np.max(accs)), 2),
        'n_profiles': len(accs)
    }

    if isinstance(data, dict):
        data['summary'] = summary
    else:
        data = {'results': data, 'summary': summary}

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"{results_dir}")
    print(f"  mean={summary['mean']}% std={summary['std']}% min={summary['min']}% max={summary['max']}% n={summary['n_profiles']}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # single directory passed as arg
        add_mean(sys.argv[1])
    else:
        # walk all Results subdirs and update any that have noisy_eval_results.json
        for root, dirs, files in os.walk('Results'):
            if 'noisy_eval_results.json' in files:
                add_mean(root)