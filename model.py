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
import os

#files
import noise

# QML imports
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AmplitudeEmbedding
from qiskit_aer import AerSimulator

DEBUGGING = False

class Triplet:
    def __init__(self, num_qubits, backend, shots, noise_train=False):
        self.weights_list = []
        self.num_wires = num_qubits
        self.num_layers = 4 # defautlt 4
        self.batch_size = 30
        self.epochs = 150
        self.embed_dims = 5
        self.losses = []
        self.shots = shots
        self.learning_rate = 0.1
        self.perturbation_rate = 0.05

        # Load noise data
        # print("\nLoading Calibration Data:")
        # if noise_train:
        #     self.noise_profiles = noise.load_calibration_data(include_fake=fake, include_hist=(fake == False), num_hist=self.historic_load, holdouts=self.holdout_profiles, load_prof=self.load_prof) # load training noise profiles
        #     # build and cache noisy encoders
        #     for prof in tqdm(self.noise_profiles, desc="Building Encoders"):
        #         prof["noisy_encoder"] = self.build_noisy_circuit(prof["noise_model"], prof['backend'])
            
        #     # create noise sets
        #     self.np_train = [prof for prof in self.noise_profiles if prof['filename'] not in self.holdout_profiles] # load training noise profiles
        #     self.np_test = [prof for prof in self.noise_profiles if prof['filename'] in self.holdout_profiles] # load training noise profiles
        #     print(f"Train profiles: {len(self.np_train)} | Test profiles: {len(self.np_test)}")
        #     print(f"Train backends: {list(set(p['backend'] for p in self.np_train))}")

        if 'aer' in backend:
            circuit = self.init_qiskit()
        elif 'mixed' in backend:
            circuit = qml.device('lightning.qubit', wires=self.num_wires)
        
        @qml.qnode(circuit, shots=self.shots)
        def layers(self, weights, features=None):
            AmplitudeEmbedding(features=features.astype('float64'), wires=range(self.num_wires), normalize=True, pad_with=0)
            #AmplitudeEmbedding(features=features.astype('float64'), wires=range(2), normalize=True, pad_with=0)
            for W in weights:
                self.layer(W)
            return [qml.expval(qml.PauliZ(i)) for i in range(6)]
        
        self.layers = layers
        
    def train(self, triplets):
        # opt = qml.GradientDescentOptimizer(stepsize=0.1)
        # opt = qml.AdamOptimizer(stepsize=0.01)
        opt = qml.SPSAOptimizer(maxiter=self.epochs, a=self.learning_rate, c=self.perturbation_rate)
        self.weights = 0.01 * np.random.randn(self.num_layers, self.num_wires, 3)
        self.loss_history = []

        pbar = tqdm(range(self.epochs), desc="Training")
        for i in pbar:
            batch_index = np.random.randint(0, len(triplets), (self.batch_size,)) # select batch_size random triplets to include in epoch
            x_train_batch = [triplets[im] for im in batch_index]

            self.weights = opt.step(lambda v: self.loss(v, x_train_batch), self.weights)

            curr_loss = float(self.loss(self.weights, x_train_batch))
            self.loss_history.append(curr_loss)

            pbar.set_postfix(loss=f"{curr_loss:.4f}")

            if i % 20 == 0 and i > 1:
                self.weights_list.append(self.weights)
                numpy.save('base_mnist', self.weights_list)

        self.plot_loss()

    def plot_loss(self):
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
        plt.savefig('sliq_loss.png', dpi=150)
        plt.show()

    def loss(self, weights, features):
        loss = 0
        for im in features:
            A = self.layers(self, weights, np.array(im[0]))
            P = self.layers(self, weights, np.array(im[1]))
            N = self.layers(self, weights, np.array(im[2]))
            # loss += (np.square(A[0]-P[0]) + np.square(A[1]-P[1])) - (np.square(A[0]-N[0]) + np.square(A[1]-N[1]))
            loss += (np.abs(A[0]-P[0]) + np.abs(A[1]-P[1])) - (np.abs(A[0]-N[0]) + np.abs(A[1]-N[1]))
        #print(str(loss))
        #print(f'Epoch: {self.cur_epoch} Accuracy: {100* correct / (len(features))}')
        return loss / len(features)

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


    def get_embeddings(self, triplets, weights=None):
        if weights is None:
            # use model weights if no past weights are supplied
            weights = self.weights
        embeddings = []
        for im in tqdm(triplets, desc="Generating embeddings"):
            z_out = self.layers(self, weights, np.array(im[0]))
            embeddings.append(numpy.array([float(z) for z in z_out]))
        return numpy.array(embeddings)

    def plot_embeddings(self, triplets, labels, test_triplets=None, test_labels=None, weights=None):
        embeddings = self.get_embeddings(triplets, weights)
        anchor_labels = [int(l) for l in labels]

        has_test = test_triplets is not None
        if has_test:
            test_embeddings = self.get_embeddings(test_triplets, weights)
            test_anchor_labels = [int(l) for l in test_labels]

        eval_embeddings = test_embeddings if has_test else embeddings
        eval_labels = test_anchor_labels if has_test else anchor_labels

        # GMM - fit on train, evaluate on eval set
        num_classes = len(set(anchor_labels))
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

        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(embeddings, anchor_labels)  # supervised eval only
        y_hat = knn.predict(test_embeddings if has_test else embeddings)
        accuracy = 100 * sum(i == j for i, j in zip(y_hat, eval_labels)) / len(eval_labels)
        print(F"KNN Clustering Accuracy ({accuracy}%)")

        accuracy = 100 * max_cor / len(eval_labels)
        print(f"GMM Clustering Accuracy ({'test' if has_test else 'train'}): {accuracy:.2f}%")

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
            scatter_plot(axes[1], test_embeddings, test_anchor_labels, f'Test Embeddings\nGMM Accuracy: {accuracy:.2f}%')
        else:
            axes[0].set_title(f'SLIQ Baseline Embeddings - MNIST\nGMM Accuracy: {accuracy:.2f}%')

        plt.tight_layout()
        plt.savefig('sliq_embeddings.png', dpi=150)
        plt.show()

    def init_qiskit(self, noise_model=None):
        """Clean or noisy device initialisation"""
        sim = AerSimulator(
            noise_model=noise_model,
            method='density_matrix',
            max_parallel_threads=5, # I have 5 super cores so no use heating up the whole laptop for no increase in speed
            max_parallel_experiments=5
        )
        circuit  = qml.device('qiskit.aer', wires=self.num_wires, backend=sim)
        return circuit
    
    # def build_noisy_circuit(self, noise_model, backend):
    #     noise_sim = AerSimulator(
    #         noise_model=noise_model,
    #         method='density_matrix',
    #         basis_gates=['cx', 'u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'id', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'swap', 'cx', 'ccx']
    #     )
    #     if DEBUGGING: print(f"Built Noisy Encoder using {backend}")
    #     noisy_dev = qml.device(
    #         'qiskit.aer',
    #         wires=self.num_encoder_qubits,
    #         backend=noise_sim,
    #     )

    #     @qml.qnode(noisy_dev, shots=self.shots)
    #     def noisy_circuit(weights, features):
    #         AmplitudeEmbedding(
    #             features=features.astype('float64'),
    #             wires=self.num_wires,
    #             normalize=True, pad_with=0
    #         )
    #         for W in weights:
    #             self.layer(W, self.num_encoder_qubits)
    #         return [qml.expval(qml.PauliZ(i)) for i in range(self.embed_dims)]

    #     return noisy_circuit
