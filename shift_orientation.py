import sys
sys.path.append('..')
import numpy
import os, json, pickle
from sklearn.mixture import GaussianMixture
from collections import defaultdict
import triplet_generator
from noise.noise import noise as noise_helper

RESULTS_ROOT = "Results"
KEYWORD      = "knnconsistency"
SS_SAMPLES   = [63, 550, 1755, 2633, 2653, 3444, 4518]
N_PROFILES   = 5

def get_mean_shift_vector(path):
    shifts_path = os.path.join(path, "knn_shifts.npy")
    if os.path.exists(shifts_path):
        shifts = numpy.load(shifts_path, allow_pickle=True)
        return numpy.mean(shifts, axis=0)
    return None


def get_cluster_vector(path, triplets, weights):
    """fit GMM on clean embeddings, return vector between class cluster centres"""
    import pennylane as qml
    from pennylane.templates import AmplitudeEmbedding
    from pennylane import numpy as np

    with open(os.path.join(path, "run_info.json")) as f:
        config = json.load(f)["config"]

    num_wires  = config["num_qubits"]
    embed_dims = config["embed_dims"]

    dev = qml.device('lightning.qubit', wires=num_wires)

    @qml.qnode(dev, shots=None)
    def circuit(w, features):
        AmplitudeEmbedding(features=features.astype('float64'), wires=range(num_wires), normalize=True, pad_with=0)
        for W in w:
            for i in range(num_wires):
                qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
            for wire in range(num_wires - 1):
                qml.CNOT(wires=[wire, wire + 1])
            qml.CNOT(wires=[num_wires - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(embed_dims)]

    # get embeddings for a subset of triplets
    embs = []
    for t in triplets[:500]:
        emb = numpy.array([float(z) for z in circuit(weights, numpy.array(t[0]))])
        embs.append(emb)
    embs = numpy.array(embs)

    gmm = GaussianMixture(n_components=3, random_state=42, n_init=5)
    gmm.fit(embs)

    centres = gmm.means_  # shape (3, embed_dims)
    # take the vector between the two most separated centres
    max_dist = 0
    best_vec = None
    for i in range(len(centres)):
        for j in range(i+1, len(centres)):
            vec  = centres[j] - centres[i]
            dist = numpy.linalg.norm(vec)
            if dist > max_dist:
                max_dist = dist
                best_vec = vec
    return best_vec


def cosine_similarity(a, b):
    return numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b) + 1e-8)


def collect_runs():
    runs = defaultdict(dict)
    for date_dir in sorted(os.listdir(RESULTS_ROOT)):
        date_path = os.path.join(RESULTS_ROOT, date_dir)
        if not os.path.isdir(date_path):
            continue
        for run_dir in sorted(os.listdir(date_path)):
            run_path = os.path.join(date_path, run_dir)
            run_info_path = os.path.join(run_path, "run_info.json")
            if not os.path.exists(run_info_path):
                continue
            with open(run_info_path) as f:
                config = json.load(f)["config"]
            if KEYWORD not in config.get("message", ""):
                continue
            seed = config.get("seed", 0)
            nt   = config["noise_train"]
            runs[seed]["NT" if nt else "non-NT"] = (run_path, config)
    return runs


def reconstruct_shift_bank(path, triplets):
    with open(os.path.join(path, "run_info.json")) as f:
        config = json.load(f)["config"]

    from model import Triplet
    model = Triplet(config, testing=True, results_dir=path)
    
    best_path = os.path.join(path, "best_weights.npy")
    model.weights = numpy.load(best_path, allow_pickle=True)

    # load noise profiles
    nh = noise_helper(fake=False, hist_count=config["historic_load"])
    profiles = nh.load_calibration_data(limit_backends=config.get("backend_name"))
    holdouts = nh.holdout_profiles
    model.np_train = [p for p in profiles if p["filename"] not in holdouts]
    for prof in model.np_train:
        prof["circuit"] = model.build_noisy_circuit(prof["noise_model"])

    _, shift_bank, _ = model.build_clustered_shift_bank(triplets)
    
    knn_shifts = numpy.array([s[1] for s in shift_bank])
    return numpy.mean(knn_shifts, axis=0)
                                   

if __name__ == '__main__':
    triplets, labels = triplet_generator.generate_pca_triplets(
        dataset="MNIST", label_space=3, num_triplets=5000, pca_dims=32, testing=False
    )

    runs = collect_runs()
    seeds = sorted(runs.keys())

    print(f"\n{'Seed':<6} {'Condition':<10} {'Cos Sim (boundary vs noise)':>28}")
    print("-" * 50)

    for seed in seeds:
        for label in ["NT", "non-NT"]:
            if label not in runs[seed]:
                continue
            path, config = runs[seed][label]

            best_path = os.path.join(path, "best_weights.npy")
            w = numpy.load(best_path, allow_pickle=True)

            shift_vec = reconstruct_shift_bank(path, triplets)
            cluster_vec = get_cluster_vector(path, triplets, w)

            if shift_vec is None:
                print(f"{seed:<6} {label:<10} {'no GMM found':>28}")
                continue

            cos_sim = cosine_similarity(shift_vec, cluster_vec)
            print(f"{seed:<6} {label:<10} {cos_sim:>28.4f}")