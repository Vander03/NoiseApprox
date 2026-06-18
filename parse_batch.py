# analyse_batch.py
import sys
sys.path.append('..')
import os
import json
import numpy
import matplotlib
matplotlib.use('Agg')
from collections import defaultdict

results_root = "Results"
keyword      = "keyword:1neighbour3class0.2"

# ─────────────────────────────────────────────────────────────
# CORE PROCESSING
# ─────────────────────────────────────────────────────────────

summary        = defaultdict(list)
matched_runs   = []
backends       = ['ibm_kingston', 'ibm_fez', 'ibm_marrakesh']
backend_summary = defaultdict(lambda: defaultdict(list))


def _process_run(run_path, force_condition=None):
    run_info_path = os.path.join(run_path, "run_info.json")
    if not os.path.exists(run_info_path):
        return

    with open(run_info_path) as f:
        run_info = json.load(f)
    config  = run_info["config"]
    message = config.get("message", "")

    # noiseless runs bypass keyword filter
    if force_condition != "noiseless" and keyword not in message:
        return

    if force_condition == "noiseless":
        nt = "noiseless"
    else:
        nt = "NT" if config["noise_train"] else "non-NT"

    lr        = config["learning_rate"]
    seed      = config.get("seed", "?")
    condition = (f"lr={lr}", nt)

    noisy_files = []
    for fname in os.listdir(run_path):
        if fname.endswith("noisy_eval_results.json"):
            noisy_files.append(os.path.join(run_path, fname))

    if not noisy_files:
        return

    all_accs = []
    for noisy_path in noisy_files:
        with open(noisy_path) as f:
            noisy_data = json.load(f)
        results   = noisy_data.get("results", [])
        clean_acc = next((r["accuracy"] for r in results if r.get("filename") == "clean"), None)
        noisy_accs = [r["accuracy"] for r in results if r.get("filename") != "clean"]
        all_accs.extend(noisy_accs)

        for backend in backends:
            b_accs = [r["accuracy"] for r in results
                      if r.get("backend") == backend
                      and r.get("filename") != "clean"]
            if b_accs:
                backend_summary[backend][nt].append(numpy.mean(b_accs))
                if clean_acc:
                    backend_summary[backend][f"{nt}_clean"].append(clean_acc)

    if not all_accs:
        return

    mean_acc = numpy.mean(all_accs)
    summary[condition].append(mean_acc)
    matched_runs.append({
        "path":        run_path,
        "config":      config,
        "mean_acc":    mean_acc,
        "condition":   nt,
        "has_weights": os.path.exists(os.path.join(run_path, "weights.npy"))
    })


# ─────────────────────────────────────────────────────────────
# SCAN — old dated structure: Results/YYYY-MM-DD/HH-MM-SS__*/
# ─────────────────────────────────────────────────────────────

for date_dir in sorted(os.listdir(results_root)):
    date_path = os.path.join(results_root, date_dir)
    if not os.path.isdir(date_path):
        continue
    if date_dir in ('NT', 'NT_v2', 'NT_v3', 'noiseless', 'Batch'):
        continue
    for run_dir in sorted(os.listdir(date_path)):
        run_path = os.path.join(date_path, run_dir)
        if os.path.isdir(run_path):
            _process_run(run_path)

# ─────────────────────────────────────────────────────────────
# SCAN — new structured: Results/NT_v3/backend/seedX/
# ─────────────────────────────────────────────────────────────

for condition_dir in ['NT', 'NT_v2']:
    condition_path = os.path.join(results_root, condition_dir)
    if not os.path.isdir(condition_path):
        continue
    for backend_dir in sorted(os.listdir(condition_path)):
        backend_path = os.path.join(condition_path, backend_dir)
        if not os.path.isdir(backend_path):
            continue
        for seed_dir in sorted(os.listdir(backend_path)):
            seed_path = os.path.join(backend_path, seed_dir)
            if os.path.isdir(seed_path):
                _process_run(seed_path)

# ─────────────────────────────────────────────────────────────
# SCAN — noiseless: Results/noiseless/seedX/
# ─────────────────────────────────────────────────────────────

noiseless_path = os.path.join(results_root, "noiseless")
if os.path.isdir(noiseless_path):
    for seed_dir in sorted(os.listdir(noiseless_path)):
        seed_path = os.path.join(noiseless_path, seed_dir)
        if os.path.isdir(seed_path):
            _process_run(seed_path, force_condition="noiseless")

# ─────────────────────────────────────────────────────────────
# OVERALL SUMMARY
# ─────────────────────────────────────────────────────────────

print(f"\n{'Condition':<17} {'N':>4} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 60)
for (lr, nt), accs in sorted(summary.items()):
    accs = numpy.array(accs)
    print(f"{lr} {nt:<10} {len(accs):>4} {accs.mean():>8.2f} {accs.std():>8.2f} {accs.min():>8.2f} {accs.max():>8.2f}")

# ─────────────────────────────────────────────────────────────
# PER-BACKEND SUMMARY
# ─────────────────────────────────────────────────────────────

print(f"\n{'Backend':<20} {'Condition':<10} {'N':>4} {'Clean':>8} {'Noisy Mean':>12} {'Std':>8} {'Drop':>8}")
print("=" * 75)
for backend in backends:
    for condition in ["noiseless", "non-NT", "NT"]:
        noisy_means = backend_summary[backend][condition]
        clean_accs  = backend_summary[backend][f"{condition}_clean"]
        if not noisy_means:
            continue
        noisy_mean = numpy.mean(noisy_means)
        noisy_std  = numpy.std(noisy_means)
        clean_mean = numpy.mean(clean_accs) if clean_accs else 0
        drop       = clean_mean - noisy_mean
        n          = len(noisy_means)
        print(f"{backend:<20} {condition:<10} {n:>4} {clean_mean:>8.1f}% "
              f"{noisy_mean:>11.1f}% {noisy_std:>7.2f} {drop:>7.1f}%")
    print()

# ─────────────────────────────────────────────────────────────
# PER-SEED DETAIL
# ─────────────────────────────────────────────────────────────

print(f"\n{'Seed':<8} {'Condition':<10} {'Backend':<8} {'Mean':>8} {'Clean':>8} {'Min Noisy':>12} {'Max Noisy':>12}")
print("-" * 60)

seed_runs = sorted(matched_runs, key=lambda r: (
    r["config"].get("seed", 0),
    r["condition"]
))

for run in seed_runs:
    path   = run["path"]
    config = run["config"]
    seed   = config.get("seed", "?")
    nt     = run["condition"]

    for fname in sorted(os.listdir(path)):
        if not fname.endswith("noisy_eval_results.json"):
            continue
        backend_tag = fname.replace("_noisy_eval_results.json", "").replace("noisy_eval_results.json", "default")
        with open(os.path.join(path, fname)) as f:
            noisy_data = json.load(f)
        results   = noisy_data.get("results", [])
        clean_acc = next((r["accuracy"] for r in results if r.get("filename") == "clean"), None)
        accs      = [r["accuracy"] for r in results if r.get("filename") != "clean"]
        if not accs:
            continue
        print(f"{seed:<8} {nt:<10} {backend_tag:<12} {numpy.mean(accs):>8.1f}% "
              f"{clean_acc or 0:>7.1f}% {numpy.min(accs):>11.1f}% {numpy.max(accs):>11.1f}%")

# ─────────────────────────────────────────────────────────────
# WEIGHT NORMS
# ─────────────────────────────────────────────────────────────

# print(f"\n{'Seed':<8} {'Condition':<10} {'Weight Norm':>12}")
# print("-" * 35)

# for run in seed_runs:
#     path   = run["path"]
#     config = run["config"]
#     seed   = config.get("seed", "?")
#     nt     = "NT" if config["noise_train"] else "non-NT"

#     best_path    = os.path.join(path, 'best_weights.npy')
#     weights_path = os.path.join(path, 'weights.npy')

#     if os.path.exists(best_path):
#         weights = numpy.load(best_path, allow_pickle=True)
#     elif os.path.exists(weights_path):
#         weights = numpy.load(weights_path, allow_pickle=True)[-1]
#     else:
#         continue

#     print(f"{seed:<8} {nt:<10} {numpy.linalg.norm(weights):>12.4f}")