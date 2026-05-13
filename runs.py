import os, json
import numpy as np

def print_run_means(results_root="Results", keyword=None, seed=None):
    runs = []
    
    for date_dir in sorted(os.listdir(results_root)):
        date_path = os.path.join(results_root, date_dir)
        if not os.path.isdir(date_path):
            continue
        for run_dir in sorted(os.listdir(date_path)):
            run_path = os.path.join(date_path, run_dir)
            noisy_path = os.path.join(run_path, "noisy_eval_results.json")
            run_info_path = os.path.join(run_path, "run_info.json")
            if not os.path.exists(noisy_path) or not os.path.exists(run_info_path):
                continue
            
            with open(run_info_path) as f:
                run_info = json.load(f)
            with open(noisy_path) as f:
                noisy = json.load(f)
            
            message = run_info["config"].get("message", "")
            if (keyword and keyword not in message) or (run_info["config"].get("seed", 1) != seed):
                continue
            
            results = noisy.get("results", [])
            accs = []
            for r in results:
                if r.get("filename") == "clean":
                    continue
                acc = r.get("accuracy") or r.get("cluster_acc")
                if acc is not None:
                    accs.append(acc)

            clean = next(
                (r.get("accuracy") or r.get("cluster_acc") 
                for r in results if r.get("filename") == "clean" or r.get("backend") == "clean"),
                None
            )
            
            if not accs:
                continue
            
            mean = np.mean(accs)
            nt = run_info["config"].get("noise_train", False)
            lr = run_info["config"].get("learning_rate", "?")
            seed = run_info["config"].get("seed", "?")
            
            runs.append({
                "dir": run_dir,
                "nt": nt,
                "lr": lr,
                "seed": seed,
                "clean": clean,
                "mean": mean,
                "message": message,
                "filename": run_dir
            })
    
    print(f"\n{'NT':<8} {'LR':<6} {'Seed':<6} {'Clean':>8} {'Noisy Mean':>12}  Message")
    print("-" * 80)
    for r in runs:
        nt_str = "NT" if r["nt"] else "non-NT"
        clean_str = f"{r['clean']:.1f}%" if r["clean"] else "N/A"
        print(f"{nt_str:<8} {str(r['lr']):<6} {str(r['seed']):<6} {clean_str:>8} {r['mean']:>11.2f}%  {r['message'][len(r["message"])-13:]}  {r['filename'][:50]}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    print_run_means(keyword=args.keyword, seed=args.seed)