# summarise_seeds.py
import os, json
import numpy as np
from collections import defaultdict

results_root = "Results"
summary = defaultdict(list)

for date_dir in os.listdir(results_root):
    date_path = os.path.join(results_root, date_dir)
    if not os.path.isdir(date_path):
        continue
    for run_dir in os.listdir(date_path):
        run_path = os.path.join(date_path, run_dir)
        noisy_path = os.path.join(run_path, "noisy_eval_results.json")
        run_info_path = os.path.join(run_path, "run_info.json")
        if not os.path.exists(noisy_path) or not os.path.exists(run_info_path):
            continue
        with open(run_info_path) as f:
            run_info = json.load(f)
        with open(noisy_path) as f:
            noisy = json.load(f)
        
        config = run_info["config"]
        message = config.get("message", "")
        
        # only include seed runs
        if "400epoch" not in message:
            continue
        
        lr = config["learning_rate"]
        nt = config["noise_train"]
        mean_acc = noisy.get("summary", {}).get("mean", None)
        if mean_acc is None:
            accs = [r["accuracy"] for r in noisy.get("results", [])]
            mean_acc = np.mean(accs) if accs else None
        
        if mean_acc is not None:
            key = (f"lr={lr}", "NT" if nt else "non-NT")
            summary[key].append(mean_acc)

print(f"\n{'Condition':<17} {'N':>4} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 60)
for (lr, nt), accs in sorted(summary.items()):
    accs = np.array(accs)
    print(f"{lr} {nt:<10} {len(accs):>4} {accs.mean():>8.2f} {accs.std():>8.2f} {accs.min():>8.2f} {accs.max():>8.2f}")