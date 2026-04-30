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

#files
from noise.noise import noise

# QML imports
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AmplitudeEmbedding
from qiskit_aer import AerSimulator

DEBUGGING = False

class Triplet:
    def __init__(self, params, testing=False):
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


        # Load noise data
        print("\nLoading Calibration Data:")
        # only load the noise profiles during training, the testing function loads its own profiles for testing
        if self.noise_train and (testing == False):
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
        elif 'mixed' in self.backend:
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
            # also allows me to save the individual contributions
            last_loss = None
            def loss_with_capture(weights):
                nonlocal last_loss
                total, clean, noisy = self.loss(weights, x_train_batch)
                last_loss = (total, clean, noisy)
                return total

            self.weights = opt.step(loss_with_capture, self.weights)

            if last_loss is not None:
                total, clean, noisy = last_loss
                current_loss = float(total)
                self.loss_history.append(current_loss)
                self.clean_loss_history.append(float(clean))
                self.noisy_loss_history.append(float(noisy))
            else:
                current_loss = 0.0

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

            clean_loss += (np.abs(A[0]-P[0]) + np.abs(A[1]-P[1])) - (np.abs(A[0]-N[0]) + np.abs(A[1]-N[1]))

            # noisy embeddings
            if self.noise_train and self.np_train:
                n_loss = []
                # hold out some noise profiles for generialisation testing
                trainable = [p for p in self.np_train if p['filename'] not in self.holdout_profiles]
                Cn = random.sample(trainable, min(self.noise_samp_per_batch, len(trainable))) # select 10 random samples from the collected noise profiles
                self.used_profiles = Cn
                for prof in Cn:
                    n_A = prof["circuit"](weights, np.array(im[0]))
                    l = (np.abs(A[0]-n_A[0]) + np.abs(A[1]-n_A[1])) # embed the noisy anchor close to clean anchor
                    n_loss.append(prof["csc"] * l)
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

    # def evaluate_embeddings(embeddings, labels):


    def init_qiskit(self, noise_model=None):
        """Clean or noisy device initialisation"""
        sim = AerSimulator(
            noise_model=noise_model,
            method='statevector',
            max_parallel_threads=5, # I have 5 super cores so no use heating up the whole laptop for no increase in speed
            max_parallel_experiments=5
        )
        circuit  = qml.device('qiskit.aer', wires=self.num_wires, backend=sim)
        return circuit
    
    def build_noisy_circuit(self, noise_model):
        noise_sim = AerSimulator(
            noise_model=noise_model,
            method='statevector',
            basis_gates=['cx', 'u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'id', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'swap', 'cx', 'ccx']
        )
        noisy_dev = qml.device(
            'qiskit.aer',
            wires=self.num_wires,
            backend=noise_sim,
        )

        @qml.qnode(noisy_dev, shots=self.shots)
        def noisy_circuit(weights, features):
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
    def save_experiment(self, triplets, labels, test_triplets=None, test_labels=None, log_dir='Results'):
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

        results_dir = os.path.join(log_dir, run_name)
        os.makedirs(results_dir, exist_ok=True)

        # save loss history and weights
        np.save(os.path.join(results_dir, 'loss_history.npy'), self.loss_history)
        numpy.save(os.path.join(results_dir, 'weights.npy'), self.weights_list)

        # save loss plot
        self.plot_loss(save_path=os.path.join(results_dir, 'loss.png'))

        # run embedding eval and save plots
        kmeans_train, gmm_train = self.plot_embeddings(
            triplets, labels,
            save_path=os.path.join(results_dir, 'embeddings_train.png')
        )
        kmeans_test, gmm_test = None, None
        if test_triplets is not None:
            kmeans_test, gmm_test = self.plot_embeddings(
                triplets, labels, test_triplets, test_labels,
                save_path=os.path.join(results_dir, 'embeddings_test.png')
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

        with open(os.path.join(results_dir, 'run_info.json'), 'w') as f:
            json.dump(run_info, f, indent=4)

        print(f"RESULTS_DIR: {results_dir}")
        return results_dir
    
    # TODO: add noisy and clean loss components to graph, integral style
    def plot_loss(self, save_path=None):
        smoothed = pd.Series(self.loss_history).rolling(window=10, min_periods=1).mean()
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, alpha=0.3, color='steelblue', label='Raw loss')
        plt.plot(smoothed, color='steelblue', linewidth=2, label='Smoothed (10-epoch)')
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Triplet Loss')
        plt.title('SLIQ Baseline Training Loss (MNIST)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path if save_path else 'loss.png', dpi=150)
        # plt.show()

    # TODO: change to only plot pre-embedded embeddings
    # TODO: create a plotting function that will plot the test embeddings for each noise profile in a giant grid
    def plot_embeddings(self, triplets, labels, test_triplets=None, test_labels=None, weights=None, save_path=None):
        embeddings = self.get_embeddings(triplets, weights)
        anchor_labels = [int(l) for l in labels]
        num_classes = len(set(anchor_labels))

        has_test = test_triplets is not None
        if has_test:
            test_embeddings = self.get_embeddings(test_triplets, weights)
            test_anchor_labels = [int(l) for l in test_labels]

        eval_embeddings = test_embeddings if has_test else embeddings
        eval_labels = test_anchor_labels if has_test else anchor_labels

        # GMM - fit on train, evaluate on eval set
        gmm = GaussianMixture(n_components=num_classes, random_state=42, n_init=10)
        gmm.fit(embeddings)
        y_hat = gmm.predict(eval_embeddings)

        # permutation search
        y_hats = self.generate_y_hats(y_hat, num_classes)
        max_cor, best_yhat = 0, y_hat
        for yh in y_hats:
            num_cor = sum(1 for i, j in zip(yh, eval_labels) if i == j)
            if num_cor > max_cor:
                max_cor = num_cor
                best_yhat = yh

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_classes, init='k-means++', random_state=42)
        kmeans.fit(embeddings, anchor_labels)
        y_hat = kmeans.predict(test_embeddings if has_test else embeddings)
        kmeans_accuracy = 100 * sum(i == j for i, j in zip(y_hat, eval_labels)) / len(eval_labels)
        print(f"Kmeans Clustering Accuracy ({'test' if has_test else 'train'}): {kmeans_accuracy:.2f}%")


        gmm_accuracy = 100 * max_cor / len(eval_labels)
        print(f"GMM Clustering Accuracy ({'test' if has_test else 'train'}): {gmm_accuracy:.2f}%")

        unique_labels = sorted(set(anchor_labels))
        colors = plt.cm.tab10(numpy.linspace(0, 1, len(unique_labels)))

        n_plots = 2 if has_test else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6))
        if not has_test:
            axes = [axes]

        def scatter_plot(ax, embs, lbls, title):
            unique = sorted(set(lbls))
            for label, color in zip(unique, colors):
                mask = [i for i, l in enumerate(lbls) if l == label]
                ax.scatter(
                    embs[mask, 0], embs[mask, 1],
                    label=f'Class {label}',
                    color=color, alpha=0.7, s=30, edgecolors='none'
                )
            ax.set_xlabel('Z₀ expectation')
            ax.set_ylabel('Z₁ expectation')
            ax.set_title(title)
            ax.legend()
        
        scatter_plot(axes[0], embeddings, anchor_labels, 'Train Embeddings')
        if has_test:
            scatter_plot(axes[1], test_embeddings, test_anchor_labels, f'Test Embeddings\nGMM Accuracy: {gmm_accuracy:.2f}%')
        else:
            axes[0].set_title(f'SLIQ Baseline Embeddings - MNIST\nGMM Accuracy: {gmm_accuracy:.2f}%')

        plt.tight_layout()
        plt.savefig(save_path if save_path else 'embeddings.png', dpi=150)
        # plt.show()

        return kmeans_accuracy, gmm_accuracy

    def load_weights(self, weights):
        self.weights = weights

    def predict_noisy_clustering(self, x_train, y_train, x_test, y_test, noise_profile=None, variance=None):
        # select the holdout profile noise cirucits from circuit storage
        # OR select the provided noise profiles
        test_profiles = self.noise_helper.load_calibration_data(
            noise_profile if noise_profile else self.holdout_profiles
        )
        
        # for each noise profile, build noisy circuit and get embeddings
        test_results = []
        for prof in tqdm(test_profiles, desc="Building Encoders"):
                circuit = self.build_noisy_circuit(prof["noise_model"])
                test_emb = self.get_embeddings(triplets=x_test, circuit=circuit)
                results = self.evaluate_embeddings(test_emb, y_test)
                test_results.append({
                    'backend': prof['filename'],
                    'csc': prof['csc'],
                    'results': results
                })



