import numpy
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import random
from PIL import Image
from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import itertools
import pandas as pd
import os, json
import datetime
import pickle # pickle time
from umap import UMAP


#files
from noise.noise import noise
from gridplotter import GridPlotter

# QML imports
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AmplitudeEmbedding
from qiskit_aer import AerSimulator

DEBUGGING = False

class Triplet:
    def __init__(self, params, testing=False, results_dir=None):
        self.noise_helper = noise(fake=params['fake'], hist_count=params['historic_load'])
        self.params = params
        self.weights_list = []
        self.loss_history = []
        self.clean_loss_history = []
        self.noisy_loss_history = []
        self.losses = []
        self.holdout_profiles = self.noise_helper.holdout_profiles
        self.best_loss = float('inf')
        self.current_epoch = 0
        self.ss_samples = 0

        # unpack params
        self.num_wires = params['num_qubits']
        self.num_layers = params['layers']
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']
        self.embed_dims = params['embed_dims']
        self.shots = params['shots']
        self.learning_rate = params['learning_rate']
        self.cooldown_lr = params['cooldown_lr']
        self.perturbation_rate = params['perturbation_rate']
        self.backend = params['backend']
        self.noise_train = params['noise_train']
        self.noise_samp_per_batch = params['noise_samp_per_batch']
        self.historic_load = params['historic_load']
        self.fake = params['fake']
        self.optimiser = params['optimiser']
        self.pca_dims = params['PCA_dims']
        self.label_space = params['label_space']
        self.num_triplets = params['num_triplets']
        self.dataset = params['dataset']
        self.testing = testing
        self.sim = 'density_matrix' if testing else params.get('sim', 'statevector')
        print(f"Sim: {self.sim}, Shots: {self.shots}")
        self.threshold = params["threshold"]
        self.variance_samples = params['variance_samples']
        self.epoch_variance = params['epoch_variance']
        self.variance = None # initialise to none for non-NT runs
        self.staged_epochs = params.get('staged_epochs', 50)
        self.ramp = params.get('ramp', 50)
        print(f"RAMP: {self.ramp}")
        self.cluster_weight = params.get('cluster_weight', 5)
        self.backend_name = params.get('backend_name', None)
        print(f"Loading {self.backend_name} only...")

        # randomly initialise the weights
        self.weights = 0.01 * np.random.randn(self.num_layers, self.num_wires, 3) # move here to allow for retraining. Otherwise it overwrites the loaded weights when training starts
        self._rng = random.Random(42) # seed for variances kept consistent between seeds

        # file management
        # only create a new file when training, save stuff to the provided filename when testing
        if testing and results_dir:
            self.results_dir = results_dir
        elif testing and results_dir == None:
            print("ERROR: Test directory not supplied")
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

            self.results_dir = os.path.join('Results', run_name)

        # Load noise data
        # only load the noise profiles during training, the testing function loads its own profiles for testing
        if (self.noise_train and (testing == False)) or True:
            print("\nLoading Calibration Data:")
            self.noise_profiles = self.noise_helper.load_calibration_data(limit_backends=self.backend_name)
            for prof in tqdm(self.noise_profiles, desc="Building Encoders"):
                prof["circuit"] = self.build_noisy_circuit(prof["noise_model"])
            
            # create noise sets
            self.np_train = [prof for prof in self.noise_profiles if prof['filename'] not in self.holdout_profiles] # load training noise profiles
            self.np_test = [prof for prof in self.noise_profiles if prof['filename'] in self.holdout_profiles] # load training noise profiles
            print(f"Train profiles: {len(self.np_train)} | Test profiles: {len(self.np_test)}")
            print(f"Train backends: {list(set(p['backend'] for p in self.np_train))}")

        # init circuit backends
        qiskit = self.init_qiskit()
        pennylane = qml.device('lightning.qubit', wires=self.num_wires)


        
        @qml.qnode(pennylane, shots=self.shots)
        def circuit(self, weights, features=None):
            AmplitudeEmbedding(features=features.astype('float64'), wires=range(self.num_wires), normalize=True, pad_with=0)
            #AmplitudeEmbedding(features=features.astype('float64'), wires=range(2), normalize=True, pad_with=0)
            for W in weights:
                self.layer(W)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.embed_dims)]
        
        @qml.qnode(qiskit, shots=1000)
        def qiskit_circuit(self, weights, features=None):
            AmplitudeEmbedding(features=features.astype('float64'), wires=range(self.num_wires), normalize=True, pad_with=0)
            #AmplitudeEmbedding(features=features.astype('float64'), wires=range(2), normalize=True, pad_with=0)
            for W in weights:
                self.layer(W)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.embed_dims)]
                    
        self.circuit = circuit
        self.qiskit_circuit = qiskit_circuit
        
    def train(self, triplets, labels):
        if "ADAM" in self.optimiser:
            opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        if "GRAD" in self.optimiser:
            opt = qml.GradientDescentOptimizer(stepsize=self.learning_rate)
        if "SPSA" in self.optimiser:
            opt = qml.SPSAOptimizer(maxiter=self.epochs, a=self.learning_rate, c=self.perturbation_rate)

        self.loss_history = []
        # if self.noise_train:    self.fit_noise_distribution(triplets=triplets)
        # self.fit_noise_distribution(triplets=triplets, labels=labels)
        if self.noise_train:
            clean_embs, shift_bank, _ = self.build_clustered_shift_bank(triplets)
            self.knn_shifts = numpy.array([s[1] for s in shift_bank])
            knn_embs = numpy.array([s[0] for s in shift_bank])
            from sklearn.neighbors import NearestNeighbors
            self.knn = NearestNeighbors(n_neighbors=1)
            self.knn.fit(knn_embs)

        pbar = tqdm(range(self.epochs), desc="Training")
        for i in pbar:
            batch_index = np.random.randint(0, len(triplets), self.batch_size) # select batch_size random triplets to include in epoch
            x_train_batch = [triplets[im] for im in batch_index]
            self.current_epoch = i
            self.weights = opt.step(lambda w: self.loss(w, x_train_batch)[0], self.weights)

            # separate clean eval for logging
            total, clean, noisy = self.loss(self.weights, x_train_batch)
            current_loss = float(total)
            self.loss_history.append(current_loss)
            self.clean_loss_history.append(float(clean))
            self.noisy_loss_history.append(float(noisy))

            pbar.set_postfix(loss=f"{current_loss:.4f}")

            # track best weights
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_weights = self.weights.copy()
            
            if i % 20 == 0 and i > 1:
                self.weights_list.append(self.weights)

    def hypersphere_random(self, embedding):
        # sample direction uniformly on unit hypersphere
        direction = numpy.random.randn(len(embedding))
        direction = direction / numpy.linalg.norm(direction)

        return direction
    
    # def sample_noise(self):
    #     """sample from fitted noise distribution"""
    #     return numpy.random.multivariate_normal(self.noise_mean, self.noise_cov)
    
    def sample_noise_knn(self, clean_embedding):
        from autograd.tracer import getval
        emb_np = numpy.array(getval(clean_embedding).tolist())
        _, neighbour_idx = self.knn.kneighbors(emb_np.reshape(1, -1))
        return self.knn_shifts[neighbour_idx[0][0]]

    def loss(self, weights, features):
        clean_loss = 0
        cluster_loss = 0
        consistency_loss = 0
        total_loss = 0
        embeddings = []
        # margin = 0.35 # 0.21
        for im in features:
            # clean embeddings
            A = self.circuit(self, weights, np.array(im[0]))
            P = self.circuit(self, weights, np.array(im[1]))
            N = self.circuit(self, weights, np.array(im[2]))
            # save embeddings to evaluate inter-cluster seperation
            embeddings.append(np.array(A))
            embeddings.append(np.array(P))
            embeddings.append(np.array(N))


            # clean_loss += (np.abs(A[0]-P[0]) + np.abs(A[1]-P[1])) - (np.abs(A[0]-N[0]) + np.abs(A[1]-N[1]))
            d_pos = sum(np.square(A[i]-P[i]) for i in range(len(A)))
            d_neg = sum(np.square(A[i]-N[i]) for i in range(len(A)))
            clean_loss += d_pos - d_neg

            # noisy embeddings
            if self.noise_train and self.np_train:
                # TODO: try noisy negative maybe. Encourage good seperation from other results? Could this be what drives samples apart better
                A_arr = np.array(A)
                P_arr = np.array(P)
                N_arr = np.array(N)

                if self.current_epoch < self.staged_epochs:
                    noisy_weight = 0.0
                else:
                    noisy_weight = min(1.0, (self.current_epoch - self.staged_epochs) / max(self.ramp, 1))
                    
                # shift_prob = self.noise_gmm.score_samples(noise_vec.reshape(1,-1))[0]
                # noisy_weight_sample = numpy.clip(numpy.exp(shift_prob), 0.0, 1.0)

                # apply a shift vector sampled from the shift vector
                n_A = A_arr + self.sample_noise_knn(A_arr)
                n_P = P_arr + self.sample_noise_knn(P_arr)
                n_N = N_arr + self.sample_noise_knn(N_arr)

                # save embeddings to evaluate inter-cluster seperation
                embeddings.append(n_A)
                embeddings.append(n_P)
                embeddings.append(n_N)

                consistency_loss += sum(np.square(n_A[i] - A_arr[i]) for i in range(len(A)))
                consistency_loss += sum(np.square(n_P[i] - P_arr[i]) for i in range(len(P)))
                consistency_loss += sum(np.square(n_N[i] - N_arr[i]) for i in range(len(N)))
        
        # dimensionality test
        # fit GMM on clean + noisy embeddings
        # if self.noise_train:
        #     from autograd.tracer import getval
        #     emb_np = numpy.array([getval(e).tolist() for e in embeddings], dtype=float)
        #     # print(f"Dimensionality: {numpy.linalg.matrix_rank(numpy.cov(emb_np.T))}")
        #     train_gmm = GaussianMixture(n_components=3, random_state=42, n_init=5)
        #     assignments = train_gmm.fit_predict(emb_np) # we dont care for correct assignments just clusters we can use to find the median of each cluster
        #     emb_pl = np.array(embeddings)  # back into pennylane numpy
        #     separation_loss = 0
        #     centres = []
        #     for k in range(3):
        #         mask = (assignments == k)
        #         if mask.sum() < 2:  # guard against empty clusters
        #             continue
        #         centre_k = np.mean(emb_pl[mask], axis=0)  # differentiable
        #         centres.append(centre_k)

        #     for i in range(len(centres)):
        #         for j in range(i+1, len(centres)):
        #             dist = np.sum(np.square(centres[i] - centres[j]))
        #             margin = 0.5  # minimum desired separation
        #             separation_loss += np.maximum(0, margin - dist)

        #     cluster_loss = (self.cluster_weight * separation_loss)


        total_loss = clean_loss + consistency_loss
        # else:
            # total_loss = clean_loss
        return total_loss / len(features), clean_loss / len(features), consistency_loss / len(features)

    # def layer(self, W):
    #     for i in range(self.num_wires):
    #         qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
    #     for wire in range(self.num_wires-1):
    #         qml.CNOT(wires=[wire, self.num_wires-1])

    # ring
    def layer(self, W):
        for i in range(self.num_wires):
            qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
        for wire in range(self.num_wires - 1):
            qml.CNOT(wires=[wire, wire + 1])  # each qubit talks to its neighbour
            qml.CNOT(wires=[self.num_wires - 1, 0])  # close the ring
    # def layer(self, W):
    #     for i in range(self.num_wires):
    #         qml.RX(W[i, 0], wires=i)  # X rotation
    #         qml.RY(W[i, 1], wires=i)  # Y rotation  
    #         qml.RZ(W[i, 2], wires=i)  # Z rotation
    #     for wire in range(self.num_wires - 1):
    #         qml.CNOT(wires=[wire, wire + 1])
    #     qml.CNOT(wires=[self.num_wires - 1, 0])

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
        """fit GMM on train embeddings, save it, return accuracy"""
        num_classes = len(set(labels))
        gmm = GaussianMixture(n_components=num_classes, random_state=42, n_init=10)
        gmm.fit(embeddings)
        with open(os.path.join(self.results_dir, 'gmm.pkl'), 'wb') as f:
            pickle.dump(gmm, f)
        y_hat = gmm.predict(embeddings)
        return self._permutation_accuracy(y_hat, labels, num_classes)

    def evaluate_embeddings_test(self, embeddings, labels):
        """load saved GMM, predict on test embeddings, return accuracy"""
        with open(os.path.join(self.results_dir, 'gmm.pkl'), 'rb') as f:
            gmm = pickle.load(f)
        num_classes = len(set(labels))
        y_hat = gmm.predict(embeddings)
        return self._permutation_accuracy(y_hat, labels, num_classes)

    def _permutation_accuracy(self, y_hat, labels, num_classes):
        y_hats = self.generate_y_hats(y_hat, num_classes)
        max_cor = max(sum(1 for i, j in zip(yh, labels) if i == j) for yh in y_hats)
        return 100 * max_cor / len(labels)
        

    def init_qiskit(self, noise_model=None):
        """Clean or noisy device initialisation"""
        sim = AerSimulator(
            noise_model=noise_model,
            method='density_matrix', # maybe change back to density_matrix
            seed_simulator=42,
            max_parallel_threads=5, # I have 5 super cores so no use heating up the whole laptop for no increase in speed
            max_parallel_experiments=5
        )
        circuit  = qml.device('qiskit.aer', wires=self.num_wires, backend=sim)
        return circuit
    
    def build_noisy_circuit(self, noise_model):
        noise_sim = AerSimulator(
            noise_model=noise_model,
            method='density_matrix',
            seed_simulator=42,
            max_parallel_threads=5, # I have 5 super cores so no use heating up the whole laptop for no increase in speed
            max_parallel_experiments=5,
            basis_gates=['cx', 'u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'id', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'swap', 'cx', 'ccx']
        )
        noisy_dev = qml.device(
            'qiskit.aer',
            wires=self.num_wires,
            backend=noise_sim,
        )

        @qml.qnode(noisy_dev, shots=5000)
        def noisy_circuit(self, weights, features):
            AmplitudeEmbedding(
                features=features.astype('float64'),
                wires=range(self.num_wires),
                normalize=True, pad_with=0
            )
            for W in weights:
                self.layer(W)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.embed_dims)]

        return noisy_circuit
        

    """
    PLOTTING
    """
    def save_experiment(self, triplets, labels, test_triplets=None, test_labels=None):
        self._ensure_results_dir()
        # self.shots = 1000 # increase shots for fitting the GMM
        # save loss history and weights
        np.save(os.path.join(self.results_dir, 'loss_history.npy'), self.loss_history)
        numpy.save(os.path.join(self.results_dir, 'weights.npy'), self.weights_list)
        # save best weights separately
        numpy.save(os.path.join(self.results_dir, 'best_weights.npy'), self.best_weights)

        # save loss plot
        self.plot_loss(save_path=os.path.join(self.results_dir, 'loss.png'))
        self.weights = self.best_weights
        # run embedding eval and save plots
        gmm_train = self.predict_clustering(
            triplets, labels,
            save_path=os.path.join(self.results_dir, 'embeddings_train.png')
        )
        gmm_test = None, None
        if test_triplets is not None:
            gmm_test = self.predict_clustering(
                triplets, labels, test_triplets, test_labels,
                save_path=os.path.join(self.results_dir, 'embeddings_test.png')
            )

        # build results
        self.params['results']['gmm_accuracy_train'] = gmm_train
        self.params['results']['gmm_accuracy_test'] = gmm_test
        # self.params['results']['kmeans_accuracy_train'] = kmeans_train
        # self.params['results']['kmeans_accuracy_test'] = kmeans_test
        self.params['results']['final_loss'] = float(self.loss_history[-1]) if self.loss_history else None
        self.params['results']['min_loss'] = float(min(self.loss_history)) if self.loss_history else None

        run_info = {
            "config": {
                **{k: v for k, v in self.params.items() if k not in ('noise_profiles', 'holdout_profiles', 'results')},
                "noise_profiles": list(
                    {p['filename']: {'filename': p['filename'], 'csc': p['csc']}
                    for p in self.np_train}.values()
                ) if self.noise_train else [],
                "holdout_profiles": self.holdout_profiles,
            },
            "results": self.params['results']
        }

        with open(os.path.join(self.results_dir, 'run_info.json'), 'w') as f:
            json.dump(run_info, f, indent=4)

        print(f"RESULTS_DIR: {self.results_dir}")
    
    # TODO: add noisy and clean loss components to graph, integral style
    def plot_loss(self, save_path=None):
        epochs = range(len(self.loss_history))
        smoothed_total = pd.Series(self.loss_history).rolling(window=10, min_periods=1).mean()
        has_components = len(self.clean_loss_history) == len(self.loss_history) and len(self.loss_history) > 0

        fig, ax = plt.subplots(figsize=(10, 5))

        if has_components:
            smoothed_clean = pd.Series(self.clean_loss_history).rolling(window=10, min_periods=1).mean()
            smoothed_noisy = pd.Series(self.noisy_loss_history).rolling(window=10, min_periods=1).mean()

            # stacked fills — noisy sits on top of clean
            ax.fill_between(epochs, smoothed_clean, alpha=0.25, color='steelblue', label='Clean component')
            ax.fill_between(epochs, smoothed_noisy, alpha=0.25, color='coral', label='Noisy component')

        # total loss on top
        ax.plot(self.loss_history, alpha=0.2, color='steelblue')
        ax.plot(smoothed_total, color='steelblue', linewidth=2, label='Total loss (smoothed)')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Triplet Loss')
        ax.set_title(f'SLIQ Training Loss ({self.dataset})')
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path if save_path else 'loss.png', dpi=150)
        plt.close()

    # TODO: change to only plot pre-embedded embeddings
    # TODO: create a plotting function that will plot the test embeddings for each noise profile in a giant grid
    def predict_clustering(self, triplets, labels, test_triplets=None, test_labels=None, save_path=None):
        anchor_labels = [int(l) for l in labels]
        has_test = test_triplets is not None

        # get embeddings
        train_emb = self.get_embeddings(triplets, self.circuit)
        
        gmm_train = self.evaluate_embeddings(train_emb, anchor_labels)

        # training grid
        grid = GridPlotter(anchor_labels, self.results_dir)
        # grid.add('Train', train_emb, anchor_labels, gmm_train, 1.0)
        umap = UMAP(n_components=2, random_state=42)
        train_umap = umap.fit_transform(train_emb)
        
        # use vis_embs for plotting, train_emb for GMM
        grid.add('Train', train_umap, anchor_labels, gmm_train, 1.0)
        
        gmm_test = None
        if has_test:
            # redefine for test so we can use the same vstack UMAP for both plots
            grid = GridPlotter(anchor_labels, self.results_dir)
            test_anchor_labels = [int(l) for l in test_labels]
            test_emb = self.get_embeddings(test_triplets, self.circuit)
            gmm_test = self.evaluate_embeddings_test(test_emb, test_anchor_labels)

            # Test umap for both training and test sets
            combined_emb = numpy.vstack([train_emb, test_emb])
            combined_labels = anchor_labels + test_anchor_labels

            umap_model = UMAP(n_components=2, random_state=42)
            combined_umap = umap_model.fit_transform(combined_emb)

            # split back
            train_umap = combined_umap[:len(train_emb)]
            test_umap = combined_umap[len(train_emb):]

            grid.add('Train', train_umap, anchor_labels, gmm_train, 1.0)
            grid.add('Test', test_umap, test_anchor_labels, gmm_test, 1.0)

        filename = os.path.basename(save_path) if save_path else 'embeddings.png'
        grid.render(filename)

        return gmm_train, gmm_test

    def load_weights(self, weights):
        self.weights = weights

    def predict_noisy_clustering(self, x_train, y_train, x_test, y_test, noise_profile=None, variance=None):
        # select the holdout profile noise cirucits from circuit storage
        # OR select the provided noise profiles
        test_profiles = self.noise_helper.load_calibration_data(noise_profile if noise_profile else self.holdout_profiles)

        # init the grid plotter
        train_labels = [int(l) for l in y_test]
        grid = GridPlotter(train_labels, self.results_dir)
        umap = UMAP(n_components=2, random_state=42)

        # train gmm again on density_matrix
        # train_emb = self.get_embeddings(x_test, self.circuit)
        # train_gmm = self.evaluate_embeddings(train_emb, [int(l) for l in y_train])

        clean_emb = self.get_embeddings(x_test, self.qiskit_circuit)
        gmm_clean = self.evaluate_embeddings(clean_emb, y_test) # refit gmm in eval environment
        # gmm_clean = self.evaluate_embeddings(clean_emb, [int(l) for l in y_test])
        print(f"Clean: {gmm_clean}")
        test_results = []
        test_results.append({
                'filename': "clean",
                'backend': "clean",
                'csc': 1.0,
                'accuracy': gmm_clean
            })
        for prof in test_profiles:
            circuit = self.build_noisy_circuit(prof["noise_model"])
            test_emb = self.get_embeddings(triplets=x_test, circuit=circuit)
            gmm_noisy = self.evaluate_embeddings_test(test_emb, y_test)
            short_name = prof['filename'].replace('hist_', '').replace('.json', '')
            print(f"Noisy: {gmm_noisy}")

            if noise_profile:
                # joint UMAP so clean and noisy are in the same space
                combined_emb = numpy.vstack([clean_emb, test_emb])
                combined_umap = umap.fit_transform(combined_emb)
                train_umap = combined_umap[:len(clean_emb)]
                test_umap = combined_umap[len(clean_emb):]
                grid.add('Clean', train_umap, y_test, gmm_clean, 1.0)
                grid.add(f'Noisy: {short_name}', test_umap, y_test, gmm_noisy, prof['csc'])
            else:
                # holdout profiles — independent UMAP per profile
                umap_noisy = umap.fit_transform(test_emb)
                grid.add(f'Noisy: {short_name}', umap_noisy, y_test, gmm_noisy, prof['csc'])

            test_results.append({
                'filename': prof['filename'],
                'backend': prof.get('backend', ''),
                'csc': prof['csc'],
                'accuracy': gmm_noisy
            })

        # render the full grid
        grid.render('noise_profile_grid.png')

        # find changes in how the model is run during training vs inference
        conf_changes = {}
        with open(os.path.join(self.results_dir, "run_info.json")) as f:
            run_info = json.load(f)
        conf = run_info["config"]
        for key in conf:
            if self.params[key] != conf[key]:
                conf_changes[key] = self.params[key] # append the changed field to the object that is returned at the end of inference

        conf_changes['results'] = test_results
        # save results json
        with open(os.path.join(self.results_dir, 'noisy_eval_results.json'), 'w') as f:
            json.dump(conf_changes, f, indent=4)

        mean_acc = numpy.mean([r['accuracy'] for r in test_results])
        print(f"Mean holdout accuracy: {mean_acc:.2f}%")
        return test_results
    
    def predict_clustering_variance(self, x_test, y_test, variance=None):

        # init the grid plotter
        train_labels = [int(l) for l in y_test]
        grid = GridPlotter(train_labels, self.results_dir)

        for i in range(variance):
            test_emb = self.get_embeddings(triplets=x_test, circuit=self.circuit)
            gmm_acc = self.evaluate_embeddings_test(test_emb, y_test)

            umap_model = UMAP(n_components=2, random_state=42)
            test_umap = umap_model.fit_transform(test_emb)


            grid.add(f'Run: {i}', test_umap, y_test, gmm_acc, 1.00)

        # render the full grid
        grid.render('variance_plots.png')


    def _ensure_results_dir(self):
        """create the results directory only when we actually need it"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir, exist_ok=True)
            print(f"Created results dir: {self.results_dir}")


    def get_variance(self, triplets):
        """
        it takes an insane amount of time to perform training on a bunch of noisy circuits. Furthermore, QuST assumes all gates are relevant,
        computing the output variance tunes results to the gates that matters for our circuit
        alternatively, we measure the circuits variance before training and use the variance to perturb the embeddings
        """
        # zero weights for identity circuit variance measurements
        zero_weights = numpy.zeros_like(self.weights)
        sample = triplets[numpy.random.randint(0, len(triplets))][0]
        # clean embeddings
        clean_embs = np.array([float(z) for z in self.circuit(self, zero_weights, np.array(sample))])

        # per-profile shift distributions
        all_shifts = []

        for prof in tqdm(self.np_train, desc="Getting variance embeddings"):
            for _ in range(self.variance_samples):  # run same sample N times
                noisy_embs = np.array([float(z) for z in prof["circuit"](self, zero_weights, np.array(sample))])
                all_shifts.append(clean_embs - noisy_embs)

        # TODO: add filtering of extremely noisy profiles based on their variance. How do i decide this variance though
        # REMOVED FILTERING IN PLACE OF MEDIAN
        
        all_shifts = numpy.array(all_shifts)
        # sigma = float(filtered_shifts.mean())
        sigma = float(numpy.median(all_shifts))
        sigma_std = float(all_shifts.std())

        return sigma, sigma_std, all_shifts

    def fit_noise_distribution(self, triplets, labels, n_samples=30, before_training=True, save_title=None):
        """fit multivariate Gaussian to observed embedding shifts"""
        from sklearn.mixture import GaussianMixture
        
        self.random_index = self.ss_samples
        # if before_training:
        #     # only zero weights for first training
        #     # weights = numpy.zeros_like(self.weights)
        #     # save embeddings used for initial comparison for comparison after training
        #     # self.random_index = numpy.random.choice(len(triplets), 7, replace=False)
        # else:
        #     l = 1
        #     # weights = self.weights
        # # random_index = numpy.random.randint(len(triplets))
        
        all_shift_vectors = []
        emb = []
        shift_backends = []
        for r in self.random_index:
            for idx in tqdm(range(n_samples), desc="Getting variance embeddings"):
                sample = triplets[r][0]
                clean_emb = numpy.array([float(z) for z in self.qiskit_circuit(self, self.weights, numpy.array(sample))]) # TODO: this is wrong, needs to be qiskit circuit wtf
                emb.append((clean_emb, "Noiseless"))
                
                for prof in self.np_train:
                    noisy_emb = numpy.array([float(z) for z in prof["circuit"](self, self.weights, numpy.array(sample))])
                    emb.append((noisy_emb, prof['backend']))
                    shift_vector = noisy_emb - clean_emb  # full vector not magnitude
                    all_shift_vectors.append(shift_vector)
                    # shift_backends.append(prof['backend'])
                    shift_backends.append(r)



                # also embed holdout profiles to see where they fall
                # for prof in self.np_test:
                #     noisy_emb = numpy.array([float(z) for z in prof["circuit"](self, zero_weights, numpy.array(sample))])
                #     emb.append((noisy_emb, f"{prof['backend']}_holdout"))
                #     shift_vector = noisy_emb - clean_emb  # full vector not magnitude
                #     all_shift_vectors.append(shift_vector)
                #     shift_backends.append(f"{prof['backend']}_holdout")
            
        all_shift_vectors = numpy.array(all_shift_vectors)
            
            # self.plot_embedding_spread(emb, f"noise_spread_{r}.png")
            # self.plot_embedding_spread_umap(emb, f"noise_spread_umap_{r}")
        if save_title is None:
            save_title = f"shift_spread_before{before_training}.png"
        self.plot_embedding_spread(list(zip(all_shift_vectors, shift_backends)), save_name=save_title)
            # self.plot_embedding_shift_magnitude(list(zip(all_shift_vectors, shift_backends)))

        # fit multivariate Gaussian to shift vectors
        # mean and covariance of the noise distribution
        self.noise_gmm = GaussianMixture(n_components=3, random_state=42) # for 3 backend profiles 
        self.noise_gmm.fit(all_shift_vectors)
        self._ensure_results_dir() # save pickled noise GMM
        with open(os.path.join(self.results_dir, 'noise_gmm.pkl'), 'wb') as f:
            pickle.dump(self.noise_gmm, f)
        self.noise_mean = numpy.mean(all_shift_vectors, axis=0)
        self.noise_cov  = numpy.cov(all_shift_vectors.T)
        
        # print(f"Noise distribution mean: {self.noise_mean}")
        # print(f"Noise covariance diagonal: {numpy.diag(self.noise_cov)}")

    def build_clustered_shift_bank(self, triplets, n_clusters=10, samples_per_cluster=5, n_noise_samples=10):
        """get clean embeddings for all triplets, cluster them, select representatives, measure real shifts"""
        self._ensure_results_dir()
        # get clean embeddings for all training samples
        print("Getting clean embeddings...")
        clean_embs = []
        indices = numpy.random.choice(len(triplets), 500, replace=False)
        anchors = [triplets[i][0] for i in indices]
        noiseless_weights = np.load("Results/2026-05-22/13-02-17__NT0_e150_shotsNone_lr0.1_cNone_histTrue__fashionMNIST_l3/best_weights.npy", allow_pickle=True)
        for t in tqdm(anchors):
            emb = numpy.array([float(z) for z in self.qiskit_circuit(self, noiseless_weights, numpy.array(t))])
            clean_embs.append(emb)
        
        # cluster clean embeddings
        clean_embs = numpy.array(clean_embs)  # add this after the embedding loop
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(clean_embs)

        # select representative samples per cluster
        calibration_indices = []
        for k in range(n_clusters):
            cluster_mask = kmeans.labels_ == k
            cluster_indices = numpy.where(cluster_mask)[0]
            selected = cluster_indices[:samples_per_cluster]
            calibration_indices.extend(selected)
        
        # measure real shifts for calibration samples
        shift_bank = []  # list of (clean_emb, shift_vector) tuples
        for idx in tqdm(calibration_indices):
            sample = anchors[idx]
            clean_emb = clean_embs[idx]
            shifts_for_sample = []
            for prof in numpy.random.choice(self.np_train, min(5, len(self.np_train)), replace=False):
                noisy_emb = numpy.array([float(z) for z in prof["circuit"](self, noiseless_weights, numpy.array(sample))])
                shifts_for_sample.append(noisy_emb - clean_emb)
            shift = numpy.mean(shifts_for_sample, axis=0)
            shift_bank.append((clean_emb, shift))

        knn_shifts_arr = numpy.array([s[1] for s in shift_bank])
        knn_embs_arr   = numpy.array([s[0] for s in shift_bank])
        self._ensure_results_dir()
        numpy.save(os.path.join(self.results_dir, 'knn_shifts.npy'), knn_shifts_arr)
        numpy.save(os.path.join(self.results_dir, 'knn_embs.npy'),   knn_embs_arr)
        return clean_embs, shift_bank, kmeans

    def evaluate_embedding_space(self, triplets, labels, save_name="embedding_dims_variance.png"):
        emb = []
        for i, triplet in enumerate(tqdm(triplets, desc="Getting Clean Embeddings")):
            sample = triplet[0]  # anchor
            clean_emb = numpy.array([float(z) for z in self.qiskit_circuit(self, self.weights, numpy.array(sample))])
            emb.append((clean_emb, str(labels[i])))
        
        self.plot_embedding_spread(emb, save_name=save_name, title="Embedding distribution by dimension", y_label="Expectation value")

    def find_variance(self, triplets, n_samples=500):
        self._ensure_results_dir()
        sigma, sigma_std, all_shifts = self.get_variance(triplets=triplets, n_samples=n_samples)

        # plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # left — shift distribution across all profiles
        axes[0].hist(all_shifts, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
        axes[0].axvline(sigma, color='red', linestyle='--', linewidth=2, label=f'mean σ={sigma:.4f}')
        axes[0].axvline(sigma + sigma_std, color='coral', linestyle=':', linewidth=1.5, label=f'±1 std')
        axes[0].axvline(sigma - sigma_std, color='coral', linestyle=':', linewidth=1.5)
        axes[0].set_xlabel('Embedding shift magnitude (L2)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Distribution of embedding shifts under noise')
        axes[0].legend()


        plt.suptitle('Noise calibration: embedding shift analysis', fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'noise_calibration.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")

        return sigma, sigma_std
    
    def plot_embedding_spread_umap(self, embeddings, save_name='embedding_spread_umap.png'):
        """
        embeddings: list of (numpy_array, backend_str) tuples
                    OR plain numpy array (no colour coding)
        """
        self._ensure_results_dir()

        # handle both formats
        emb_array = numpy.array([e[0] for e in embeddings])
        backends  = [e[1] for e in embeddings]

        umap_model = UMAP(n_components=2, random_state=42)
        embs_2d = umap_model.fit_transform(emb_array)
        fig, ax = plt.subplots(figsize=(9, 7))

        unique_backends = sorted(set(backends))
        colors = ['steelblue', 'coral', 'mediumseagreen', 'gold', 'orchid']
        for i, backend in enumerate(unique_backends):
            mask = numpy.array([b == backend for b in unique_backends])
            pts  = embs_2d[mask]
            if backend == "Noiseless":
                ax.scatter(pts[:,0], pts[:,1], s=30, alpha=1, color=colors[i % len(colors)], 
                        label=backend, marker="D", zorder=5)  
            elif "holdout" in backend:
                ax.scatter(pts[:,0], pts[:,1], s=20, alpha=1, color=colors[i % len(colors)], 
                        label=backend, marker="X", zorder=1)  
            else:
                ax.scatter(pts[:,0], pts[:,1], s=12, alpha=0.55, color=colors[i % len(colors)], 
                        label=backend, zorder=1)  

        ax.legend(fontsize=8, title='Backend')
        ax.set_title(f'Embedding spread (n={len(emb_array)})')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')

        save_path = os.path.join(self.results_dir, save_name)
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"Saved: {save_path}")


    def plot_embedding_spread(self, embeddings, save_name='embedding_spread.png', title='Per-dimension shift spread by sample', y_label="Shift Magnitude"):
        self._ensure_results_dir()

        emb_array = numpy.array([e[0] for e in embeddings])
        label  = [e[1] for e in embeddings]

        unique_labels = sorted(set(label))
        if 'Noiseless' in unique_labels:
            unique_labels = ['Noiseless'] + [b for b in unique_labels if b != 'Noiseless']

        n_labels = len(unique_labels)
        n_dims = emb_array.shape[1]
        colors = ['steelblue', 'coral', 'mediumseagreen', 'gold', 'orchid']

        # fixed candle width — image grows with more dims/labels
        candle_width = 0.12
        group_width = candle_width * n_labels  # total width per dim group
        fig_width = max(6, n_dims * (group_width + 0.4) + 1)  # +0.4 gap, +1 margin
        
        fig, ax = plt.subplots(figsize=(fig_width, 4))

        for i, l in enumerate(unique_labels):
            mask = numpy.array([b == l for b in label])
            pts = emb_array[mask]
            positions = numpy.arange(n_dims) + (i - n_labels/2 + 0.5) * candle_width
            ax.boxplot(
                [pts[:, d] for d in range(n_dims)],
                positions=positions,
                widths=candle_width * 0.8,
                patch_artist=True,
                boxprops=dict(facecolor=colors[i % len(colors)], alpha=0.6),
                medianprops=dict(color='black', linewidth=1.5),
                whiskerprops=dict(color=colors[i % len(colors)]),
                capprops=dict(color=colors[i % len(colors)]),
                flierprops=dict(marker='o', color=colors[i % len(colors)], alpha=0.3, markersize=3),
                label=f"{l}"
            )

        ax.set_xticks(numpy.arange(n_dims))
        ax.set_xticklabels([f'Z{d}' for d in range(n_dims)])
        ax.set_xlim(-0.5, n_dims - 0.5)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_ylabel(y_label)
        ax.set_xlabel('Embedding dimension')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8, title='Sample')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, save_name)
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"Saved: {save_path}")

    def plot_embedding_shift_magnitude(self, embeddings, save_name="embedding_shift_mag.png"):
        self._ensure_results_dir()

        emb_array = numpy.array([e[0] for e in embeddings])
        backends  = [e[1] for e in embeddings]
        magnitudes = numpy.linalg.norm(emb_array, axis=1)

        unique_backends = sorted(set(backends))
        data = [magnitudes[[b == backend for b in backends]] for backend in unique_backends]

        fig, ax = plt.subplots(figsize=(10, 6))
        parts = ax.violinplot(data, positions=range(len(unique_backends)), showmedians=True)

        ax.set_xticks(range(len(unique_backends)))
        ax.set_xticklabels(unique_backends, rotation=20, ha='right')
        ax.set_ylabel('Shift magnitude (L2)')
        ax.set_title('Embedding shift magnitude by backend')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, save_name)
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"Saved: {save_path}")