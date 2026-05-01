import sys
sys.path.append('..')
from model import Triplet
import triplet_generator
import numpy as np
import json, os, argparse

noise_path = "Results/2026-04-29/17-03-15__NT1_e150_shots1024_lr0.8_c0.3_histTrue__MNIST_l2"
noiseless_path = "Results/2026-04-29/17-26-03__NT0_e150_shots1024_lr0.5_c0.3_histTrue__MNIST_l2"
pennylane = "Results/2026-04-30/20-09-43__NT0_e200_shotsNone_lr0.3_c0.05_histTrue__MNIST_l2"

samples = 1000

def summarise_holdout_results(results):
    if len(results) == 0: return
    cluster = np.array([r['cluster_acc'] for r in results])
    # probe = np.array([r['probe_acc'] for r in results])
    cscs = np.array([r['csc'] for r in results])
    
    return {
        'n_profiles': len(results),
        'cluster': {
            'mean': round(cluster.mean(), 2),
            'median': round(np.median(cluster), 2),
            'std': round(cluster.std(), 2),
            'min': round(cluster.min(), 2),
            'max': round(cluster.max(), 2),
        },
        # 'probe': {
        #     'mean': round(probe.mean(), 2),
        #     'median': round(np.median(probe), 2),
        #     'std': round(probe.std(), 2),
        #     'min': round(probe.min(), 2),
        #     'max': round(probe.max(), 2),
        # },
        'csc_range': [round(cscs.min(), 3), round(cscs.max(), 3)],
    }


def analyse_model(path, num_profiles):
    # Load the run config from the previous run
    with open(os.path.join(path, "run_info.json")) as f:
        run_info = json.load(f)
    config = run_info["config"]
    config['shots'] = 1000
    # config['label_space'] = 2
    
    if 'qubit' in config['backend']:
        config['backend'] = "qiskit.aer"

    # Reconstruct the model with the same config
    model = Triplet(config, testing=True, results_dir=path)

    weights = np.load(os.path.join(path, 'weights.npy'), allow_pickle=True)
    model.weights = weights[-1]  # use the last saved checkpoint

    t_triplets, t_labels = triplet_generator.generate_pca_triplets(
        dataset=config['dataset'],
        label_space=config['label_space'],
        num_triplets=config['num_triplets'],
        testing=True
    )

    # Run noise robustness evaluation
    print("\nEvaluating noise robustness...")
    results = model.predict_noisy_clustering(
        x_test=t_triplets[:samples],
        y_test=t_labels[:samples],
        noise_profile="hist_ibm_fez_2025-04-30.json"
    )

    # Run noise robustness evaluation
    # print("\nEvaluating Clean Variance...")
    # results = model.predict_noisy_clustering(
    #     x_train=triplets[:samples],
    #     y_train=labels[:samples],
    #     x_test=t_triplets[:samples],
    #     y_test=t_labels[:samples],
    # )

    # for r in results:
    #     print(f"Backend: {r['backend']} | Cluster: {r['cluster_acc']:.1f}% | CSC: {r['csc']}")

    # Save results
    # with open(os.path.join(args.results_dir, f"Fake_{config['fake']}_noise_robustness.json"), "w") as f:
    #     json.dump(results, f, indent=4)

    # summary = summarise_holdout_results(results)
    # with open(os.path.join(path, "holdout_summary.json"), "w") as f:
    #     json.dump({
    #         "per_profile": results,
    #         "summary": summary,
    #         "samples_train_test": samples,
    #         "holdout_profile_list": model.holdout_profiles,
    #     }, f, indent=4)



date = "Results/2026-04-23"
all_results = {}
# TODO: rerun the NT grad descent, this is the best guess I have for a profile trained on the historical data, as it is the first profile that specifies the number of historical NP to include
NT_best_grad_desc = "Results/2026-04-17/23:04:52/qiskit.aer__MNIST__e20_l2_pca32_np5000_mr0.4_eq6_pq6_el6_pl6_ed6_ema0.996_batch10_shots300_NL5_NTTrue"
n_NT_best_grad_descent = "Results/2026-04-17/09:58:53/qiskit.aer__MNIST__e20_l2_pca32_np5000_mr0.4_eq6_pq6_el6_pl6_ed6_ema0.996_batch10_shots300_NL5_NTFalse"
NT_best_SPSA = 'Results/2026-04-24/13:33:08__NT1_e20_shots300_lr0.1_c0.05_histTrue'
n_NT_BEST_SPSA = 'Results/2026-04-24/13:40:41__NT0_e20_shots300_lr0.1_c0.05_histTrue'

toRun = [NT_best_grad_desc, n_NT_best_grad_descent, NT_best_SPSA, n_NT_BEST_SPSA]

mult = False

if mult:
    for run_dir in os.listdir(date):
        path = os.path.join(date, run_dir)
        if not os.path.isdir(path):
            continue
        if not os.path.exists(os.path.join(path, "run_info.json")):
            continue
        if not os.path.exists(os.path.join(path, "best_weights.npz")):
            continue
        
        print(f"\n=== Analysing: {run_dir} ===")
        try:
            analyse_model(path, num_profiles=10)
            all_results[run_dir] = "done"
        except Exception as e:
            print(f"  Skipped {run_dir}: {e}")
            all_results[run_dir] = f"error: {e}"
else:
    dir = noise_path
    # for dir in toRun:
    print(f"\n=== Analysing: {dir} ===")
    analyse_model(dir, num_profiles=10)

print("\nDone. Summary:")
for run, status in all_results.items():
    print(f"  {run}: {status}")


# NT=true
# Backend: fake_manila | Cluster: 81.7% | Probe: 84.5%
# Backend: fake_montreal | Cluster: 80.9% | Probe: 84.7%
# Backend: fake_lagos | Cluster: 69.4% | Probe: 73.7%
# Backend: fake_perth | Cluster: 75.5% | Probe: 78.7%
# Backend: fake_nighthawk | Cluster: 80.9% | Probe: 86.4%

# NT = False
# Backend: fake_nighthawk | Cluster: 81.7% | Probe: 87.7%
# Backend: fake_lagos | Cluster: 64.3% | Probe: 72.9%
# Backend: fake_perth | Cluster: 76.4% | Probe: 80.2%
# Backend: fake_manila | Cluster: 80.7% | Probe: 84.2%


# low CSC filtering

# NT - 300 shots
# Backend: fake_torino | Cluster: 80.6% | Probe: 84.4% | CSC: 0.0
# Backend: fake_kyoto | Cluster: 75.9% | Probe: 82.3% | CSC: 0.0
# Backend: fake_marrakesh | Cluster: 81.4% | Probe: 86.3% | CSC: 0.32110826727001174
# Backend: fake_cusco | Cluster: 81.5% | Probe: 84.7% | CSC: 0.6007118225640561
# Backend: fake_cambridge | Cluster: 72.7% | Probe: 77.4% | CSC: 0.5274051744936826
# Backend: fake_fez | Cluster: 82.0% | Probe: 87.4% | CSC: 0.44866360115664083

# NT - NO CSC
# Backend: fake_torino | Cluster: 76.0% | Probe: 82.8% | CSC: 0.0
# Backend: fake_kyoto | Cluster: 69.8% | Probe: 80.6% | CSC: 0.0
# Backend: fake_marrakesh | Cluster: 81.6% | Probe: 87.8% | CSC: 0.32110826727001174
# Backend: fake_cusco | Cluster: 77.0% | Probe: 85.5% | CSC: 0.6007118225640561
# Backend: fake_cambridge | Cluster: 58.8% | Probe: 71.8% | CSC: 0.5274051744936826
# Backend: fake_fez | Cluster: 83.0% | Probe: 87.1% | CSC: 0.44866360115664083

# Non NT - 400 shots
# Backend: fake_torino | Cluster: 80.0% | Probe: 85.8% | CSC: 0.0
# Backend: fake_kyoto | Cluster: 77.6% | Probe: 82.9% | CSC: 0.0
# Backend: fake_marrakesh | Cluster: 82.0% | Probe: 88.9% | CSC: 0.32110826727001174
# Backend: fake_cusco | Cluster: 81.6% | Probe: 87.7% | CSC: 0.6007118225640561
# Backend: fake_cambridge | Cluster: 70.5% | Probe: 73.7% | CSC: 0.5274051744936826
# Backend: fake_fez | Cluster: 81.5% | Probe: 90.3% | CSC: 0.44866360115664083

# NON NT - 300 shots eval, trained on 400
# Backend: fake_torino | Cluster: 78.9% | Probe: 85.3% | CSC: 0.0
# Backend: fake_kyoto | Cluster: 69.7% | Probe: 81.2% | CSC: 0.0
# Backend: fake_marrakesh | Cluster: 82.5% | Probe: 87.9% | CSC: 0.32110826727001174
# Backend: fake_cusco | Cluster: 80.0% | Probe: 85.2% | CSC: 0.6007118225640561
# Backend: fake_cambridge | Cluster: 68.4% | Probe: 72.7% | CSC: 0.5274051744936826
# Backend: fake_fez | Cluster: 79.7% | Probe: 88.0% | CSC: 0.44866360115664083