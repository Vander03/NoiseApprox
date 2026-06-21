import numpy
from matplotlib import pyplot as plt
import random
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import pandas as pd
import os, json
import datetime
import pickle
from umap import UMAP

from noise.noise import noise
from gridplotter import GridPlotter

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AmplitudeEmbedding
from qiskit_aer import AerSimulator
import sys
from sklearn.neighbors import NearestNeighbors
from visualiser import Visualiser


class Triplet:
    def __init__(self, params, testing=False, results_dir=None):
        self.noise_helper = noise(fake=params["fake"], hist_count=params["historic_load"])
        self.params = params
        self.weights_list = []
        self.loss_history = []
        self.clean_loss_history = []
        self.noisy_loss_history = []
        # self.holdout_profiles = self.noise_helper.holdout_profiles
        self.best_loss = float("inf")
        self.ss_samples = 0

        # unpack params
        self.num_wires = params["num_qubits"]
        self.num_layers = params["layers"]
        self.batch_size = params["batch_size"]
        self.epochs = params["epochs"]
        self.embed_dims = params["embed_dims"]
        self.shots = params["shots"]
        self.learning_rate = params["learning_rate"]
        self.cooldown_lr = params["cooldown_lr"]
        self.perturbation_rate = params["perturbation_rate"]
        self.backend = params["backend"]
        self.noise_train = params["noise_train"]
        self.noise_samp_per_batch = params["noise_samp_per_batch"]
        self.historic_load = params["historic_load"]
        self.fake = params["fake"]
        self.optimiser = params["optimiser"]
        self.pca_dims = params["PCA_dims"]
        self.label_space = params["label_space"]
        self.num_triplets = params["num_triplets"]
        self.dataset = params["dataset"]
        self.testing = testing
        self.sim = "density_matrix" if testing else params.get("sim", "statevector")
        print(f"Sim: {self.sim}, Shots: {self.shots}")
        self.variance_samples = params["variance_samples"]
        self.cluster_weight = params.get("cluster_weight", 5)
        self.backend_name = params.get("backend_name", None)
        print(f"Loading {self.backend_name} only...")
        self.neighbours = params.get("neighbours", 1)

        self.holdout_profiles = self.noise_helper.get_holdout_profiles(self.backend_name) if self.backend_name else self.noise_helper.kingston_holdout
        # randomly initialise the weights
        self.weights = 0.01 * np.random.randn(self.num_layers, self.num_wires, 3)

        # file management
        if testing and results_dir:
            self.results_dir = results_dir
        elif testing and results_dir is None:
            print("ERROR: Test directory not supplied")
        elif params.get("results_dir"):
            self.results_dir = params["results_dir"]
        else:
            _now = datetime.datetime.now()
            run_name = (
                f"{_now.strftime('%Y-%m-%d')}/{_now.strftime('%H-%M-%S')}"
                f"__NT{int(self.params['noise_train'])}"
                f"_e{self.params['epochs']}"
                f"_shots{self.params['shots']}"
                f"_lr{self.params['learning_rate']}"
                f"_c{self.params['perturbation_rate']}"
                f"_hist{not self.params['fake']}"
                f"__{self.params['dataset']}"
                f"_l{self.params['label_space']}"
            )
            self.results_dir = os.path.join("Results", run_name)

        print(f"Saving to: {self.results_dir}")
        # load noise profiles
        print("\nLoading Calibration Data:")
        # load and build noisy encoders for constructing the shift bank
        self.noise_profiles = self.noise_helper.load_calibration_data(limit_backends=self.backend_name)
        for prof in tqdm(self.noise_profiles, desc="Building Encoders"):
            prof["circuit"] = self.build_noisy_circuit(prof["noise_model"])

        # load the training set
        self.np_train = self.noise_profiles  # already excludes holdouts via load_calibration_data

        # load holdout profiles separately
        holdout_raw = self.noise_helper.load_calibration_data(load_prof=self.holdout_profiles)
        for prof in tqdm(holdout_raw, desc="Building Holdout Encoders"):
            prof["circuit"] = self.build_noisy_circuit(prof["noise_model"])
        self.np_test = holdout_raw

        # init circuit backends
        qiskit = self.init_qiskit()
        pennylane = qml.device("lightning.qubit", wires=self.num_wires)

        # create pennylane circuits
        @qml.qnode(pennylane, shots=self.shots)
        def circuit(self, weights, features=None):
            AmplitudeEmbedding(
                features=features.astype("float64"),
                wires=range(self.num_wires),
                normalize=True,
                pad_with=0,
            )
            for W in weights:
                self.layer(W)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.embed_dims)]

        # create qiskit noiseless circuit
        @qml.qnode(qiskit, shots=1000)
        def qiskit_circuit(self, weights, features=None):
            AmplitudeEmbedding(
                features=features.astype("float64"),
                wires=range(self.num_wires),
                normalize=True,
                pad_with=0,
            )
            for W in weights:
                self.layer(W)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.embed_dims)]

        self.circuit = circuit
        self.qiskit_circuit = qiskit_circuit

    # ring entanglement layer
    def layer(self, W):
        for i in range(self.num_wires):
            qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
        for wire in range(self.num_wires - 1):
            qml.CNOT(wires=[wire, wire + 1])  # each qubit talks to its neighbour
        qml.CNOT(wires=[self.num_wires - 1, 0])  # close the ring

    def init_qiskit(self, noise_model=None):
        """Clean or noisy device initialisation"""
        sim = AerSimulator(
            noise_model=noise_model,
            method="density_matrix",
            seed_simulator=42,
            max_parallel_threads=5,
            max_parallel_experiments=5,
        )
        return qml.device("qiskit.aer", wires=self.num_wires, backend=sim)

    def build_noisy_circuit(self, noise_model):
        """build a noisy encoder with the provided noise model"""
        noise_sim = AerSimulator(
            noise_model=noise_model,
            method="density_matrix",
            seed_simulator=42,
            max_parallel_threads=5,
            max_parallel_experiments=5,
            basis_gates=[
                "cx",
                "u1",
                "u2",
                "u3",
                "rx",
                "ry",
                "rz",
                "id",
                "x",
                "y",
                "z",
                "h",
                "s",
                "sdg",
                "t",
                "tdg",
                "swap",
                "cx",
                "ccx",
            ],
        )
        noisy_dev = qml.device("qiskit.aer", wires=self.num_wires, backend=noise_sim)

        @qml.qnode(noisy_dev, shots=5000)
        def noisy_circuit(self, weights, features):
            AmplitudeEmbedding(
                features=features.astype("float64"),
                wires=range(self.num_wires),
                normalize=True,
                pad_with=0,
            )
            for W in weights:
                self.layer(W)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.embed_dims)]

        return noisy_circuit

    def sample_noise_knn(self, clean_embedding):
        """Return the shift of the clean_embedding nearest neighbour in the shift bank"""
        from autograd.tracer import getval

        emb_np = numpy.array(getval(clean_embedding).tolist())
        _, neighbour_idx = self.knn.kneighbors(emb_np.reshape(1, -1))
        return self.knn_shifts[neighbour_idx[0][0]]

    def build_clustered_shift_bank(self, triplets, n_clusters=20, samples_per_cluster=10):
        """get clean embeddings for all triplets, cluster them, select representatives, measure real shifts"""
        self._ensure_results_dir()

        # check for shared shift bank first
        shared_bank_dir = os.path.join(os.path.dirname(self.results_dir), "shift_bank")
        shared_shifts = os.path.join(shared_bank_dir, "knn_shifts.npy")
        shared_embs = os.path.join(shared_bank_dir, "knn_embs.npy")

        if os.path.exists(shared_shifts) and os.path.exists(shared_embs):
            print(f"Loading shared shift bank from {shared_bank_dir}...")
            knn_shifts_arr = numpy.load(shared_shifts, allow_pickle=True)
            knn_embs_arr = numpy.load(shared_embs, allow_pickle=True)
            shift_bank = list(zip(knn_embs_arr, knn_shifts_arr))

            # still save to run dir for get_results compatibility
            numpy.save(os.path.join(self.results_dir, "knn_shifts.npy"), knn_shifts_arr)
            numpy.save(os.path.join(self.results_dir, "knn_embs.npy"), knn_embs_arr)

            return knn_embs_arr, shift_bank

        # saved shift bank not found, build new shift bank
        print("Building shift bank...")
        os.makedirs(shared_bank_dir, exist_ok=True)

        clean_embs = []
        indices = numpy.random.choice(len(triplets), 1000, replace=False)
        anchors = [triplets[i][0] for i in indices]
        noiseless_weights = np.load("noiseless_models/noiseless_trained_6_fashion.npy", allow_pickle=True)

        # get clean embeddings by passing anchors through trained clean circuit
        for t in tqdm(anchors):
            emb = numpy.array([float(z) for z in self.qiskit_circuit(self, noiseless_weights, numpy.array(t))])
            clean_embs.append(emb)

        # fit KMeans on the clean embeddings to partition embedding space
        clean_embs = numpy.array(clean_embs)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(clean_embs)

        # select x number of representitive samples from each cluster
        calibration_indices = []
        for k in range(n_clusters):
            cluster_mask = kmeans.labels_ == k
            cluster_indices = numpy.where(cluster_mask)[0]
            selected = cluster_indices[:samples_per_cluster]
            calibration_indices.extend(selected)

        # measure the actual noise shifts of 5 randomly selected profiles for each representitive sample
        shift_bank = []
        for idx in tqdm(calibration_indices):
            sample = anchors[idx]
            clean_emb = clean_embs[idx]
            shifts_for_sample = []
            # LIMITATION: could investigate having these be consistent, was random for increased coverage
            sampled_profiles = numpy.random.choice(self.np_train, 5, replace=False)  # based on asusmption that profiles are temporally stable
            for prof in sampled_profiles:
                noisy_emb = numpy.array([float(z) for z in prof["circuit"](self, noiseless_weights, numpy.array(sample))])
                shifts_for_sample.append(noisy_emb - clean_emb)
            shift = numpy.mean(shifts_for_sample, axis=0)
            shift_bank.append((clean_emb, shift))

        knn_shifts_arr = numpy.array([s[1] for s in shift_bank])
        knn_embs_arr = numpy.array([s[0] for s in shift_bank])

        # save to shared location
        numpy.save(shared_shifts, knn_shifts_arr)
        numpy.save(shared_embs, knn_embs_arr)
        print(f"Shift bank saved to shared location: {shared_bank_dir}")

        # also save to run dir
        numpy.save(os.path.join(self.results_dir, "knn_shifts.npy"), knn_shifts_arr)
        numpy.save(os.path.join(self.results_dir, "knn_embs.npy"), knn_embs_arr)

        return clean_embs, shift_bank

    def train(self, triplets):
        if "ADAM" in self.optimiser:
            opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        if "GRAD" in self.optimiser:
            opt = qml.GradientDescentOptimizer(stepsize=self.learning_rate)
        if "SPSA" in self.optimiser:
            opt = qml.SPSAOptimizer(maxiter=self.epochs, a=self.learning_rate, c=self.perturbation_rate)

        self.loss_history = []
        if self.noise_train:
            # get/build shift bank before training
            _, shift_bank = self.build_clustered_shift_bank(triplets)
            self.knn_shifts = numpy.array([s[1] for s in shift_bank])
            knn_embs = numpy.array([s[0] for s in shift_bank])
            # fit KNN based on shift bank embeddings
            self.knn = NearestNeighbors(n_neighbors=self.neighbours)
            self.knn.fit(knn_embs)

        pbar = tqdm(range(self.epochs), desc="Training")
        for i in pbar:
            # select n random samples for epoch
            # LIMITATION: traditional full training set per epoch not possible with how slow the sims are
            batch_index = np.random.randint(0, len(triplets), self.batch_size)
            x_train_batch = [triplets[im] for im in batch_index]
            # loss forward pass
            self.weights = opt.step(lambda w: self.loss(w, x_train_batch), self.weights)

            # separate clean eval for logging
            curr_loss = float(self.loss(self.weights, x_train_batch))
            self.loss_history.append(curr_loss)
            pbar.set_postfix(loss=f"{curr_loss:.4f}")

            # track best weights
            if curr_loss < self.best_loss:
                self.best_loss = curr_loss
                self.best_weights = self.weights.copy()

            if i % 20 == 0 and i > 1:
                self.weights_list.append(self.weights)

    def loss(self, weights, features):
        loss = 0

        for im in features:
            # clean embeddings
            A = self.circuit(self, weights, np.array(im[0]))
            P = self.circuit(self, weights, np.array(im[1]))
            N = self.circuit(self, weights, np.array(im[2]))
            margin = 0.02
            if self.noise_train:
                # replace positive with noise shifted anchor if noise training
                P_arr = np.array(P)
                n_P = P_arr + self.sample_noise_knn(P_arr)
                P = n_P

            d_pos = sum(np.square(A[i] - P[i]) for i in range(len(A)))
            d_neg = sum(np.square(A[i] - N[i]) for i in range(len(A)))
            loss += np.maximum(0, d_pos - d_neg + margin)

        return loss / len(features)

    def generate_y_hats(self, y_hat, num_clusters):
        import itertools
        perms = list(itertools.permutations(range(num_clusters)))
        y_hats = []
        for perm in perms:
            this_yhat = [perm[i] for i in y_hat]
            y_hats.append(this_yhat)
        return y_hats

    def get_embeddings(self, triplets, circuit):
        embeddings = []
        for im in tqdm(triplets, desc="Generating embeddings"):
            z_out = circuit(self, self.weights, np.array(im[0]))
            embeddings.append(numpy.array([float(z) for z in z_out]))
        return numpy.array(embeddings)

    def evaluate_embeddings(self, embeddings, labels):
        """fit GMM on embeddings, save it, return accuracy"""
        num_classes = len(set(labels))
        gmm = GaussianMixture(n_components=num_classes, random_state=42, n_init=10)
        gmm.fit(embeddings)
        with open(os.path.join(self.results_dir, "gmm.pkl"), "wb") as f:
            pickle.dump(gmm, f)
        y_hat = gmm.predict(embeddings)
        return self._permutation_accuracy(y_hat, labels, num_classes)

    def evaluate_embeddings_test(self, embeddings, labels):
        """load saved GMM, predict on test embeddings, return accuracy"""
        with open(os.path.join(self.results_dir, "gmm.pkl"), "rb") as f:
            gmm = pickle.load(f)
        num_classes = len(set(labels))
        y_hat = gmm.predict(embeddings)
        return self._permutation_accuracy(y_hat, labels, num_classes)

    def _permutation_accuracy(self, y_hat, labels, num_classes):
        y_hats = self.generate_y_hats(y_hat, num_classes)
        max_cor = max(sum(1 for i, j in zip(yh, labels) if i == j) for yh in y_hats)
        return 100 * max_cor / len(labels)

    def load_weights(self, weights):
        self.weights = weights

    def predict_noisy_clustering(self, x_train, y_train, x_test, y_test, noise_profile=None, variance=None):
        """Predict the class of test samples under the effects of the provided noise profile"""
        test_profiles = self.noise_helper.load_calibration_data(noise_profile if noise_profile else self.holdout_profiles)

        train_labels = [int(l) for l in y_test]
        grid = GridPlotter(train_labels, self.results_dir)
        umap = UMAP(n_components=2, random_state=42)

        clean_emb = self.get_embeddings(x_test, self.qiskit_circuit)
        gmm_clean = self.evaluate_embeddings(clean_emb, y_test)  # refit gmm in eval environment
        print(f"Clean: {gmm_clean}")

        test_results = [{"filename": "clean", "backend": "clean", "csc": 1.0, "accuracy": gmm_clean}]

        for prof in test_profiles:
            circuit = self.build_noisy_circuit(prof["noise_model"])
            test_emb = self.get_embeddings(triplets=x_test, circuit=circuit)
            gmm_noisy = self.evaluate_embeddings_test(test_emb, y_test)
            short_name = prof["filename"].replace("hist_", "").replace(".json", "")
            print(f"Noisy: {gmm_noisy}")

            if noise_profile:
                # joint UMAP so clean and noisy are in the same space
                combined_emb = numpy.vstack([clean_emb, test_emb])
                combined_umap = umap.fit_transform(combined_emb)
                grid.add("Clean", combined_umap[: len(clean_emb)], y_test, gmm_clean, 1.0)
                grid.add(
                    f"Noisy: {short_name}",
                    combined_umap[len(clean_emb) :],
                    y_test,
                    gmm_noisy,
                    prof["csc"],
                )
            else:
                # holdout profiles — independent UMAP per profile
                umap_noisy = umap.fit_transform(test_emb)
                grid.add(f"Noisy: {short_name}", umap_noisy, y_test, gmm_noisy, prof["csc"])

            test_results.append(
                {
                    "filename": prof["filename"],
                    "backend": prof.get("backend", ""),
                    "csc": prof["csc"],
                    "accuracy": gmm_noisy,
                }
            )

        grid.render("noise_profile_grid.png")

        conf_changes = {}
        with open(os.path.join(self.results_dir, "run_info.json")) as f:
            run_info = json.load(f)
        conf = run_info["config"]
        for key in conf:
            if self.params[key] != conf[key]:
                conf_changes[key] = self.params[key]

        conf_changes["results"] = test_results
        with open(
            os.path.join(self.results_dir, f"{self.backend_name}_noisy_eval_results.json"),
            "w",
        ) as f:
            json.dump(conf_changes, f, indent=4)

        mean_acc = numpy.mean([r["accuracy"] for r in test_results])
        print(f"Mean holdout accuracy: {mean_acc:.2f}%")
        return test_results

    def predict_clustering(self, triplets, labels, test_triplets=None, test_labels=None, save_path=None):
        """Predict the class of the test samples under noiseless conditions"""
        anchor_labels = [int(l) for l in labels]
        has_test = test_triplets is not None

        train_emb = self.get_embeddings(triplets, self.circuit)
        # NOTE: GMM is always refitted on training embeddings since there are differences in processing between Pennylane and Qiskit simulators
        # so there are small shifts in embedding space that can cause the pennylane GMM to become unreliable.
        gmm_train = self.evaluate_embeddings(train_emb, anchor_labels)

        grid = GridPlotter(anchor_labels, self.results_dir)
        umap = UMAP(n_components=2, random_state=42)
        train_umap = umap.fit_transform(train_emb)
        grid.add("Train", train_umap, anchor_labels, gmm_train, 1.0)

        gmm_test = None
        if has_test:
            grid = GridPlotter(anchor_labels, self.results_dir)
            test_anchor_labels = [int(l) for l in test_labels]
            test_emb = self.get_embeddings(test_triplets, self.circuit)
            gmm_test = self.evaluate_embeddings_test(test_emb, test_anchor_labels)

            combined_emb = numpy.vstack([train_emb, test_emb])
            umap_model = UMAP(n_components=2, random_state=42)
            combined_umap = umap_model.fit_transform(combined_emb)
            train_umap = combined_umap[: len(train_emb)]
            test_umap = combined_umap[len(train_emb) :]

            grid.add("Train", train_umap, anchor_labels, gmm_train, 1.0)
            grid.add("Test", test_umap, test_anchor_labels, gmm_test, 1.0)

        filename = os.path.basename(save_path) if save_path else "embeddings.png"
        grid.render(filename)
        return gmm_train, gmm_test

    def predict_clustering_variance(self, x_test, y_test, variance=None):
        train_labels = [int(l) for l in y_test]
        grid = GridPlotter(train_labels, self.results_dir)

        for i in range(variance):
            test_emb = self.get_embeddings(triplets=x_test, circuit=self.circuit)
            gmm_acc = self.evaluate_embeddings_test(test_emb, y_test)
            umap_model = UMAP(n_components=2, random_state=42)
            test_umap = umap_model.fit_transform(test_emb)
            grid.add(f"Run: {i}", test_umap, y_test, gmm_acc, 1.00)

        grid.render("variance_plots.png")

    def save_experiment(self, triplets, labels, test_triplets=None, test_labels=None):
        self._ensure_results_dir()

        numpy.save(os.path.join(self.results_dir, "loss_history.npy"), self.loss_history)
        numpy.save(os.path.join(self.results_dir, "weights.npy"), self.weights_list)
        numpy.save(os.path.join(self.results_dir, "best_weights.npy"), self.best_weights)

        # plot loss via visualiser
        vis = Visualiser(self.results_dir)
        vis.plot_loss(
            self.loss_history,
            self.dataset,
        )

        self.weights = self.best_weights
        gmm_train = self.predict_clustering(
            triplets,
            labels,
            save_path=os.path.join(self.results_dir, "embeddings_train.png"),
        )
        gmm_test = None, None
        if test_triplets is not None:
            gmm_test = self.predict_clustering(
                triplets,
                labels,
                test_triplets,
                test_labels,
                save_path=os.path.join(self.results_dir, "embeddings_test.png"),
            )

        self.params["results"]["gmm_accuracy_train"] = gmm_train
        self.params["results"]["gmm_accuracy_test"] = gmm_test
        self.params["results"]["final_loss"] = float(self.loss_history[-1]) if self.loss_history else None
        self.params["results"]["min_loss"] = float(min(self.loss_history)) if self.loss_history else None

        run_info = {
            "config": {
                **{k: v for k, v in self.params.items() if k not in ("noise_profiles", "holdout_profiles", "results")},
                "noise_profiles": (
                    list({p["filename"]: {"filename": p["filename"], "csc": p["csc"]} for p in self.np_train}.values()) if self.noise_train else []
                ),
                "holdout_profiles": self.holdout_profiles,
            },
            "results": self.params["results"],
        }
        with open(os.path.join(self.results_dir, "run_info.json"), "w") as f:
            json.dump(run_info, f, indent=4)
        print(f"RESULTS_DIR: {self.results_dir}")

    def _ensure_results_dir(self):
        """create the results directory only when we actually need it"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir, exist_ok=True)
            print(f"Created results dir: {self.results_dir}")
