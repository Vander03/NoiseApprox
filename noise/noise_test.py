import sys
sys.path.append('..')
from model import Triplet
import triplet_generator
import numpy as np
import json, os, argparse

noise_path = "Results/2026-04-29/17-03-15__NT1_e150_shots1024_lr0.8_c0.3_histTrue__MNIST_l2"
noiseless_path = "Results/2026-05-01/15-32-09__NT0_e150_shots300_lr0.3_c0.1_histTrue__MNIST_l3"
pennylane = "Results/2026-04-30/20-09-43__NT0_e200_shotsNone_lr0.3_c0.05_histTrue__MNIST_l2"

eval_path = "Results/2026-05-02/22-01-44__NT1_e150_shotsNone_lr0.4_cNone_histTrue__MNIST_l3"

samples = 100

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

def pca_gmm_baseline(dataset, label_space, pca_dims, num_triplets, num_classes):
    """fit GMM directly on PCA embeddings"""
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    import itertools

    triplets, labels = triplet_generator.generate_pca_triplets(
        dataset=dataset,
        label_space=label_space,
        num_triplets=num_triplets,
        pca_dims=pca_dims,
        testing=False
    )
    t_triplets, t_labels = triplet_generator.generate_pca_triplets(
        dataset=dataset,
        label_space=label_space,
        num_triplets=num_triplets,
        pca_dims=pca_dims,
        testing=True
    )

    # extract anchor embeddings
    train_emb = np.array([t[0] for t in triplets])
    test_emb = np.array([t[0] for t in t_triplets])
    train_labels = [int(l) for l in labels]
    test_labels = [int(l) for l in t_labels]

    def permutation_accuracy(y_hat, labels, num_classes):
        perms = list(itertools.permutations(range(num_classes)))
        max_cor = max(
            sum(1 for i, j in zip([p[y] for y in y_hat], labels) if i == j)
            for p in perms
        )
        return 100 * max_cor / len(labels)

    gmm = GaussianMixture(n_components=num_classes, random_state=42, n_init=10)
    gmm.fit(train_emb)

    train_acc = permutation_accuracy(gmm.predict(train_emb), train_labels, num_classes)
    test_acc = permutation_accuracy(gmm.predict(test_emb), test_labels, num_classes)

    print(f"PCA-only GMM baseline (pca_dims={pca_dims}, {num_classes}-class):")
    print(f"  Train: {train_acc:.1f}%")
    print(f"  Test:  {test_acc:.1f}%")

    return train_acc, test_acc


def continue_training(path, additional_epochs=150):
    with open(os.path.join(path, "run_info.json")) as f:
        run_info = json.load(f)
    config = run_info["config"]
    config['epochs'] = additional_epochs
    config['shots'] = None
    if 'qubit' in config['backend']:
        config['backend'] = "lightning.qubit"

    results_dir = path + "/further_training" # add subfolder to compare to previous result
    
    model = Triplet(config, testing=False, results_dir=results_dir)
    weights = np.load(os.path.join(path, 'weights.npy'), allow_pickle=True)
    model.load_weights(weights[-1])
    
    triplets, labels = triplet_generator.generate_pca_triplets(
        config['dataset'],
        label_space=config['label_space'],
        num_triplets=config['num_triplets'],
        testing=False,
        pca_dims=config['PCA_dims']
    )
    t_triplets, t_labels = triplet_generator.generate_pca_triplets(
        dataset=config['dataset'],
        label_space=config['label_space'],
        num_triplets=config['num_triplets'],
        testing=True,
        pca_dims=config['PCA_dims']
    )

    model.train(triplets)
    model.save_experiment(triplets, labels, t_triplets, t_labels)
    return results_dir

def plot_loss_landscape_3d(model, triplets, labels,
                            resolution=25, range_=0.5,
                            n_batch=16, save_path=None):
    """
    2D slice through weight space rendered as 3D surface + contour.
    picks two random orthogonal directions from current weights.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from tqdm import tqdm

    weights = np.array(model.weights).copy()

    # two random orthogonal directions in weight space
    d1 = np.random.randn(*weights.shape)
    d1 = d1 / np.linalg.norm(d1)
    d2 = np.random.randn(*weights.shape)
    d2 = d2 - np.dot(d2.flatten(), d1.flatten()) * d1
    d2 = d2 / np.linalg.norm(d2)

    alphas = np.linspace(-range_, range_, resolution)
    betas  = np.linspace(-range_, range_, resolution)
    loss_grid = np.zeros((resolution, resolution))

    # small fixed batch for speed
    batch_idx = np.random.randint(0, len(triplets), n_batch)
    batch = [triplets[i] for i in batch_idx]
    variance = model.variance if hasattr(model, 'variance') else 0.07

    for i, alpha in enumerate(tqdm(alphas, desc="Landscape")):
        for j, beta in enumerate(betas):
            w = weights + alpha * d1 + beta * d2
            loss_val = float(model.loss(w, batch, variance)[0])
            loss_grid[i, j] = np.clip(loss_val, -2.0, 2.0)  # clip spikes

    A, B = np.meshgrid(alphas, betas)
    Z = loss_grid.T  # transpose so axes align

    fig = plt.figure(figsize=(18, 7))

    # --- 3D surface ---
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(A, B, Z, cmap='viridis', alpha=0.85, linewidth=0)
    # mark current weights (origin)
    origin_loss = float(model.loss(weights, batch, variance)[0])
    ax1.scatter([0], [0], [origin_loss], color='red', s=80, zorder=5, label='Current weights')
    ax1.set_xlabel('Dir 1')
    ax1.set_ylabel('Dir 2')
    ax1.set_zlabel('Loss')
    ax1.set_title('Loss landscape (surface)')
    ax1.legend(fontsize=8)
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    # --- contour (top-down view) ---
    ax2 = fig.add_subplot(132)
    cf = ax2.contourf(A, B, Z, levels=40, cmap='viridis')
    ax2.contour(A, B, Z, levels=40, colors='white', alpha=0.2, linewidths=0.5)
    ax2.scatter(0, 0, color='red', s=80, zorder=5, label='Current weights')
    ax2.set_xlabel('Dir 1')
    ax2.set_ylabel('Dir 2')
    ax2.set_title('Loss landscape (contour)')
    ax2.legend(fontsize=8)
    plt.colorbar(cf, ax=ax2)

    # --- 1D slice along dir 1 ---
    ax3 = fig.add_subplot(133)
    mid = resolution // 2
    ax3.plot(alphas, loss_grid[:, mid], color='steelblue', linewidth=2, label='Dir 1 slice')
    ax3.plot(betas,  loss_grid[mid, :], color='coral',     linewidth=2, label='Dir 2 slice', linestyle='--')
    ax3.axvline(0, color='red', linewidth=1, linestyle=':')
    ax3.axhline(origin_loss, color='red', linewidth=1, linestyle=':', label='Current loss')
    ax3.set_xlabel('Step size')
    ax3.set_ylabel('Loss')
    ax3.set_title('1D slices through origin')
    ax3.legend(fontsize=8)

    plt.suptitle(
        f'Loss landscape | range=±{range_} | resolution={resolution}x{resolution}',
        fontsize=12
    )
    plt.tight_layout()

    path = save_path or os.path.join(model.results_dir, 'loss_landscape.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # sharpness metric — mean loss increase from perturbation
    center_loss = loss_grid[resolution//2, resolution//2]
    sharpness = float(np.mean(loss_grid) - center_loss)
    print(f"Sharpness metric: {sharpness:.4f} (higher = sharper minimum)")

    return loss_grid, sharpness

# def analyse_model(path, num_profiles):
#     # Load the run config from the previous run
#     with open(os.path.join(path, "run_info.json")) as f:
#         run_info = json.load(f)
#     config = run_info["config"]
#     config['shots'] = 1000
#     # config['label_space'] = 3
#     # config['label_space'] = 2
    
#     if 'qubit' in config['backend']:
#         config['backend'] = "qiskit.aer"

#     # Reconstruct the model with the same config
#     model = Triplet(config, testing=True, results_dir=path)

#     weights = np.load(os.path.join(path, 'weights.npy'), allow_pickle=True)
#     model.weights = weights[-1]  # use the last saved checkpoint
    
#     t_triplets, t_labels = triplet_generator.generate_pca_triplets(
#         dataset=config['dataset'],
#         label_space=config['label_space'],
#         num_triplets=config['num_triplets'],
#         pca_dims=config['PCA_dims'],
#         testing=True
#     )

#     train_acc, test_acc = pca_gmm_baseline(
#         dataset=config['dataset'],
#         label_space=config['label_space'],
#         pca_dims=config['PCA_dims'],
#         num_triplets=config['num_triplets'],
#         num_classes=config['label_space']
#     )

#     # Run noise robustness evaluation
#     print("\nEvaluating noise robustness...")
#     results = model.predict_noisy_clustering(
#         x_test=t_triplets[:samples],
#         y_test=t_labels[:samples],
#         noise_profile=None # "hist_ibm_fez_2025-04-30.json"
#     )

#     print("\nEvaluating Clean Variance...")
#     results = model.predict_clustering_variance(
#         x_test=t_triplets[:samples],
#         y_test=t_labels[:samples],
#         variance=4
#     )

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



# date = "Results/2026-04-23"
# all_results = {}
# # TODO: rerun the NT grad descent, this is the best guess I have for a profile trained on the historical data, as it is the first profile that specifies the number of historical NP to include
# NT_best_grad_desc = "Results/2026-04-17/23:04:52/qiskit.aer__MNIST__e20_l2_pca32_np5000_mr0.4_eq6_pq6_el6_pl6_ed6_ema0.996_batch10_shots300_NL5_NTTrue"
# n_NT_best_grad_descent = "Results/2026-04-17/09:58:53/qiskit.aer__MNIST__e20_l2_pca32_np5000_mr0.4_eq6_pq6_el6_pl6_ed6_ema0.996_batch10_shots300_NL5_NTFalse"
# NT_best_SPSA = 'Results/2026-04-24/13:33:08__NT1_e20_shots300_lr0.1_c0.05_histTrue'
# n_NT_BEST_SPSA = 'Results/2026-04-24/13:40:41__NT0_e20_shots300_lr0.1_c0.05_histTrue'

# toRun = [NT_best_grad_desc, n_NT_best_grad_descent, NT_best_SPSA, n_NT_BEST_SPSA]

# mult = False

# if mult:
#     for run_dir in os.listdir(date):
#         path = os.path.join(date, run_dir)
#         if not os.path.isdir(path):
#             continue
#         if not os.path.exists(os.path.join(path, "run_info.json")):
#             continue
#         if not os.path.exists(os.path.join(path, "best_weights.npz")):
#             continue
        
#         print(f"\n=== Analysing: {run_dir} ===")
#         try:
#             analyse_model(path, num_profiles=10)
#             all_results[run_dir] = "done"
#         except Exception as e:
#             print(f"  Skipped {run_dir}: {e}")
#             all_results[run_dir] = f"error: {e}"
# else:
#     dir = eval_path
#     # for dir in toRun:
#     print(f"\n=== Analysing: {dir} ===")
#     analyse_model(dir, num_profiles=10)

# print("\nDone. Summary:")
# for run, status in all_results.items():
#     print(f"  {run}: {status}")

# at the bottom of noise_test.py, replace the if __name__ == '__main__' block with:


def analyse_model(path, num_profiles=10):
    with open(os.path.join(path, "run_info.json")) as f:
        run_info = json.load(f)
    config = run_info["config"]
    config['shots'] = 1000

    if 'qubit' in config['backend']:
        config['backend'] = "qiskit.aer"
        config['sim'] = "density_matrix"

    model = Triplet(config, testing=True, results_dir=path)
    # try best weights first, fall back to last
    best_path = os.path.join(path, 'best_weights.npy')
    if os.path.exists(best_path):
        model.weights = np.load(best_path, allow_pickle=True)
    else:
        weights = np.load(os.path.join(path, 'weights.npy'), allow_pickle=True)
        model.weights = weights[-1]
    
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

    # loss_grid, sharpness = plot_loss_landscape_3d(
    #     model, t_triplets[:500], t_labels[:500],
    #     resolution=25,
    #     range_=0.5,
    #     save_path=os.path.join(path, 'loss_landscape.png')
    # )
    # print(f"Sharpness: {sharpness:.4f}")
    # return

    pca_gmm_baseline(
        dataset=config['dataset'],
        label_space=config['label_space'],
        pca_dims=config['PCA_dims'],
        num_triplets=config['num_triplets'],
        num_classes=config['label_space']
    )

    print("\nEvaluating noise robustness...")
    results = model.predict_noisy_clustering(
        x_train=triplets[:samples],
        y_train=triplets[:samples],
        x_test=t_triplets[:samples],
        y_test=t_labels[:samples],
    )

    print("\nEvaluating Clean Variance...")
    model.predict_clustering_variance(
        x_test=t_triplets[:samples],
        y_test=t_labels[:samples],
        variance=4
    )

    return results


if __name__ == '__main__':
    print(f"\n=== Analysing: {eval_path} ===")
    # retrain_path = continue_training(eval_path, 150)
    # if retrain_path:
    #     analyse_model(retrain_path, num_profiles=10)
    # else:
    eval_path = "Results/2026-05-04/22-19-48__NT1_e250_shotsNone_lr0.1_cNone_histTrue__MNIST_l3"
    analyse_model(eval_path, num_profiles=10)






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