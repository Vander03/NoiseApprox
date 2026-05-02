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

        # unpack params
        self.num_wires = params['num_qubits']
        self.num_layers = params['layers']
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']
        self.embed_dims = params['embed_dims']
        self.shots = params['shots']
        self.learning_rate = params['learning_rate']
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
        if self.noise_train and (testing == False):
            print("\nLoading Calibration Data:")
            self.noise_profiles = self.noise_helper.load_calibration_data()
            for prof in tqdm(self.noise_profiles, desc="Building Encoders"):
                prof["circuit"] = self.build_noisy_circuit(prof["noise_model"])
            
            # create noise sets
            self.np_train = [prof for prof in self.noise_profiles if prof['filename'] not in self.holdout_profiles] # load training noise profiles
            self.np_test = [prof for prof in self.noise_profiles if prof['filename'] in self.holdout_profiles] # load training noise profiles
            print(f"Train profiles: {len(self.np_train)} | Test profiles: {len(self.np_test)}")
            print(f"Train backends: {list(set(p['backend'] for p in self.np_train))}")

        if 'aer' in self.backend:        
            circuit = self.init_qiskit()
        elif 'lightning' in self.backend:
            circuit = qml.device('lightning.qubit', wires=self.num_wires)


        
        @qml.qnode(circuit, shots=self.shots)
        def circuit(self, weights, features=None):
            AmplitudeEmbedding(features=features.astype('float64'), wires=range(self.num_wires), normalize=True, pad_with=0)
            #AmplitudeEmbedding(features=features.astype('float64'), wires=range(2), normalize=True, pad_with=0)
            for W in weights:
                self.layer(W)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.embed_dims)]
        
        self.circuit = circuit
        
    def train(self, triplets):
        if "ADAM" in self.optimiser:
            opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        if "GRAD" in self.optimiser:
            opt = qml.GradientDescentOptimizer(stepsize=self.learning_rate)
        if "SPSA" in self.optimiser:
            opt = qml.SPSAOptimizer(maxiter=self.epochs, a=self.learning_rate, c=self.perturbation_rate)

        # randomly initialise the weights
        self.weights = 0.01 * np.random.randn(self.num_layers, self.num_wires, 3)
        self.loss_history = []

        pbar = tqdm(range(self.epochs), desc="Training")
        for i in pbar:
            batch_index = np.random.randint(0, len(triplets), (self.batch_size,)) # select batch_size random triplets to include in epoch
            x_train_batch = [triplets[im] for im in batch_index]

            # replace lambda and save the previous loss to avoid re-evaluation
            # also allows me to save the individual contributions for plotting
            self.weights = opt.step(lambda w: self.loss(w, x_train_batch)[0], self.weights)

            # separate clean eval for logging
            total, clean, noisy = self.loss(self.weights, x_train_batch)
            current_loss = float(total)
            self.loss_history.append(current_loss)
            self.clean_loss_history.append(float(clean))
            self.noisy_loss_history.append(float(noisy))

            pbar.set_postfix(loss=f"{current_loss:.4f}")

            if i % 20 == 0 and i > 1:
                self.weights_list.append(self.weights)
                numpy.save('base_mnist', self.weights_list)


    def loss(self, weights, features):
        clean_loss = 0
        noisy_loss = 0
        total_loss = 0
        for im in features:
            # clean embeddings
            A = self.circuit(self, weights, np.array(im[0]))
            P = self.circuit(self, weights, np.array(im[1]))
            N = self.circuit(self, weights, np.array(im[2]))

            # clean_loss += (np.abs(A[0]-P[0]) + np.abs(A[1]-P[1])) - (np.abs(A[0]-N[0]) + np.abs(A[1]-N[1]))
            clean_loss += (np.square(A[0]-P[0]) + np.square(A[1]-P[1])) - (np.square(A[0]-N[0]) + np.square(A[1]-N[1]))

            # noisy embeddings
            if self.noise_train and self.np_train:
                n_loss = []
                # hold out some noise profiles for generialisation testing
                trainable = [p for p in self.np_train if p['filename'] not in self.holdout_profiles]
                Cn = random.sample(trainable, min(self.noise_samp_per_batch, len(trainable))) # select 10 random samples from the collected noise profiles
                for prof in Cn:
                    n_A = prof["circuit"](self, weights, np.array(im[0]))
                    # l = (np.abs(A[0]-n_A[0]) + np.abs(A[1]-n_A[1])) # embed the noisy anchor close to clean anchor
                    # l = (np.square(A[0]-n_A[0]) + np.square(A[1]-n_A[1])) # square loss noisy anchor should be close to clean anchor
                    l = (np.square(n_A[0]-P[0]) + np.square(n_A[1]-P[1])) - (np.square(n_A[0]-N[0]) + np.square(n_A[1]-N[1])) # noisy anchor needs to be closer to the positive than the negative
                    n_loss.append(l)
                    #n_loss.append(prof["csc"] * l) 
                # accumulate noise into loss function
                noisy_loss += (1/(self.noise_samp_per_batch) * sum(n_loss))

            # loss += (np.square(A[0]-P[0]) + np.square(A[1]-P[1])) - (np.square(A[0]-N[0]) + np.square(A[1]-N[1])) # l2 loss
        total_loss = noisy_loss + clean_loss
        return total_loss / len(features), clean_loss / len(features), noisy_loss / len(features)

    # def layer(self, W):
    #     for i in range(self.num_wires):
    #         qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
    #     for wire in range(self.num_wires-1):
    #         qml.CNOT(wires=[wire, self.num_wires-1])

    def layer(self, W):
        for i in range(self.num_wires):
            qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
        for wire in range(self.num_wires - 1):
            qml.CNOT(wires=[wire, wire + 1])  # each qubit talks to its neighbour
        qml.CNOT(wires=[self.num_wires - 1, 0])  # close the ring


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
            method=self.sim, # maybe change back to density_matrix
            max_parallel_threads=5, # I have 5 super cores so no use heating up the whole laptop for no increase in speed
            max_parallel_experiments=5
        )
        circuit  = qml.device('qiskit.aer', wires=self.num_wires, backend=sim)
        return circuit
    
    def build_noisy_circuit(self, noise_model):
        noise_sim = AerSimulator(
            noise_model=noise_model,
            method=self.sim,
            basis_gates=['cx', 'u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'id', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'swap', 'cx', 'ccx']
        )
        noisy_dev = qml.device(
            'qiskit.aer',
            wires=self.num_wires,
            backend=noise_sim,
        )

        @qml.qnode(noisy_dev, shots=self.shots)
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

        # save loss plot
        self.plot_loss(save_path=os.path.join(self.results_dir, 'loss.png'))

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

    def predict_noisy_clustering(self, x_test, y_test, noise_profile=None, variance=None):
        # select the holdout profile noise cirucits from circuit storage
        # OR select the provided noise profiles
        test_profiles = self.noise_helper.load_calibration_data(noise_profile if noise_profile else self.holdout_profiles)

        # init the grid plotter
        train_labels = [int(l) for l in y_test]
        grid = GridPlotter(train_labels, self.results_dir)
        umap = UMAP(n_components=2, random_state=42)

        clean_emb = self.get_embeddings(x_test, self.circuit)
        gmm_clean = self.evaluate_embeddings_test(clean_emb, [int(l) for l in y_test])
        print(f"Clean: {gmm_clean}")
        test_results = []
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