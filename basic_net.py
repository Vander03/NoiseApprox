import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AmplitudeEmbedding
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
class Triplet:
    def __init__(self, num_qubits):
        self.weights_list = []
        self.num_wires = num_qubits
        self.num_layers = 4
        self.batch_size = 30
        self.epochs = 100
        self.embed_dims = 5
        self.losses = []
        
    def train(self, triplets):
        opt = qml.GradientDescentOptimizer(stepsize=0.1)
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
            # Correct Order
            flattened = []
            for i, j, k in zip(im[0], im[1], im[2]):
                flattened.append(i)
            A = self.embed(self, weights, np.array(flattened))
            flattened = []
            for i, j, k in zip(im[0], im[1], im[2]):
                flattened.append(j)
            P = self.embed(self, weights, np.array(flattened))
            flattened = []
            for i, j, k in zip(im[0], im[1], im[2]):
                flattened.append(k)
            N = self.embed(self, weights, np.array(flattened))
            loss += (np.square(A[0]-P[0]) + np.square(A[1]-P[1])) - (np.square(A[0]-N[0]) + np.square(A[1]-N[1]))
        #print(str(loss))
        #print(f'Epoch: {self.cur_epoch} Accuracy: {100* correct / (len(features))}')
        return loss / len(features)

    @qml.qnode(qml.device(name='default.qubit', wires=13))
    def embed(self, weights, features=None):
        AmplitudeEmbedding(features=features.astype('float64'), wires=range(self.num_wires), normalize=True, pad_with=0)
        #AmplitudeEmbedding(features=features.astype('float64'), wires=range(2), normalize=True, pad_with=0)
        for W in weights:
            self.layer(W)
        return [qml.expval(qml.PauliZ(i)) for i in range(2)]

    def layer(self, W):
        for i in range(self.num_wires):
            qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
        for wire in range(self.num_wires-1):
            qml.CNOT(wires=[wire, self.num_wires-1])

    def eval_acc(self, triplets, labels, weights):
        all_weights = numpy.load(weights)
        for count, weights in enumerate(all_weights[-5:]):
            correct = 0
            embeddings = []
            for im in tqdm(triplets):
                flattened = []
                for i, j, k in zip(im[0], im[1], im[2]):
                    flattened.append(i)
                z_out1 = self.embed(self, weights, np.array(flattened))
                embeddings.append(z_out1)
            
            num_clusters = 3
            kmeans = KMeans(num_clusters)
            #kmeans = GaussianMixture(num_clusters)
            kmeans.fit(embeddings)
            #kmeans = GaussianMixture(2)
            y_hat = kmeans.predict(embeddings)
            y = [i[0] for i in labels]


            max_cor = 0
            for lab_set in [['CI', 'CA', 'CM'], ['CI', 'CM', 'CA'], ['CA', 'CM', 'CI'], ['CA', 'CI', 'CM'], ['CM', 'CA', 'CI'], ['CM', 'CI', 'CA']]:
                num_cor = 0
                lab_dict = {lab_set[0]: 0, lab_set[1]: 1, lab_set[2]: 2}
                for i, j in zip(y_hat, y):
                    if i == lab_dict[j]:
                        num_cor +=1
                max_cor = max(num_cor, max_cor)
            print(100*max_cor / len(triplets))
            y_hats = self.generate_y_hats(y_hat, num_clusters)
            for yh in y_hats:
                num_cor = 0
                for i, j in zip(yh, y):
                    if i == j:
                        num_cor +=1
                max_cor = max(num_cor, max_cor)
            print(100*max_cor / len(triplets))

    def generate_y_hats(self, y_hat, num_clusters):
        import itertools
        perms = list(itertools.permutations(range(num_clusters)))
        y_hats = []
        for perm in perms:
            this_yhat = [perm[i] for i in y_hat]
            y_hats.append(this_yhat)
        return y_hats


    