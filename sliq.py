import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AmplitudeEmbedding
import numpy
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
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
from collections import defaultdict


class Triplet:
    def __init__(self, num_qubits):
        self.weights_list = []
        self.num_wires = num_qubits
        self.num_layers = 4
        self.batch_size = 20
        self.epochs = 50
        self.embed_dims = 5
        self.losses = []
        
    def train(self, triplets):
        opt = qml.GradientDescentOptimizer(stepsize=0.1)
        self.weights = 0.01 * np.random.randn(self.num_layers, self.num_wires, 3)
        self.loss_history = []

        pbar = tqdm(range(self.epochs), desc="Training")
        for i in pbar:
            self.cur_epoch = i
            batch_index = np.random.randint(0, len(triplets), (self.batch_size,))
            x_train_batch = [triplets[im] for im in batch_index]
            self.weights = opt.step(lambda v: self.cost_embedding(v, x_train_batch), self.weights)

            curr_loss = float(self.cost_embedding(self.weights, x_train_batch))
            self.loss_history.append(curr_loss)
            self.weights_list.append(self.weights)

            if i%20 == 0:
                pbar.set_postfix(loss=f"{curr_loss:.4f}")

        self.plot_loss()


    def plot_loss(self):
        smoothed = pd.Series(self.loss_history).rolling(window=10, min_periods=1).mean()
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, alpha=0.3, color='steelblue', label='Raw loss')
        plt.plot(smoothed, color='steelblue', linewidth=2, label='Smoothed (10-epoch)')
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Triplet Loss')
        plt.title('SLIQ Training Loss (MNIST)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('sliq_loss.png', dpi=150)
        plt.show()


    def cost_embedding(self, weights, features):
        loss = 0
        for im in features:
            # Correct Order
            sign = 1
            flattened = []
            # #good
            for i, j, k in zip(im[0], im[1], im[2]):
                flattened.append(i)
                flattened.append(j)

            #bad ends also cycle loss
            z_out1 = self.embed(self, weights, np.array(flattened))
            z1_loss = self.triplet_loss(z_out1)
            # Reversed Order
            sign = -1
            flattened = []
            for i, j, k in zip(im[0], im[1], im[2]):
                flattened.append(k)
                flattened.append(i)
            z_out2 = self.embed(self, weights, np.array(flattened))
            z2_loss = self.triplet_loss(z_out2)
            siam_loss = .9*(z1_loss - z2_loss)
            consistancy_loss = .1*(np.abs((z_out1[0] - z_out2[2])) + np.abs((z_out1[1] - z_out2[3])))
            loss += .9*siam_loss
            loss += .1*consistancy_loss
        return loss / len(features)

    @qml.qnode(qml.device(name='lightning.qubit', wires=11))
    def embed(self, weights, features=None):
        AmplitudeEmbedding(features=features.astype('float64'), wires=range(self.num_wires), normalize=True, pad_with=0)
        #AmplitudeEmbedding(features=features.astype('float64'), wires=range(3), normalize=True, pad_with=0)
        for W in weights:
            self.layer(W)
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]
    


    def layer(self, W):
        for i in range(self.num_wires):
            qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
        for wire in range(self.num_wires-1):
            qml.CNOT(wires=[wire, self.num_wires-1])

    def triplet_loss(self, z_out):
        return np.abs(z_out[0] - z_out[2]) + np.abs(z_out[1] - z_out[3])


    def eval_acc(self, triplets, labels, weights_file):
        x = [i[0] for i in triplets]
        y = [i[0] for i in labels]
        all_weights = numpy.load(weights_file)
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        x = [i[1] for i in triplets]
        y = [i[1] for i in labels]
        clf.fit(x, y)
        print(clf.score(x, y))
        accuracies = []
        print(len(all_weights))
        for count, weights in enumerate(all_weights[-5:]):
            correct = 0
            embeddings = []
            for im in tqdm(triplets):
                flattened = []
                for i, j, k in zip(im[0], im[1], im[2]):
                    flattened.append(i)
                    flattened.append(j)
                z_out1 = self.embed(self, weights, np.array(flattened))
                embeddings.append(np.reshape(z_out1, [-1]))

            else:
                num_clusters = 2
                if '4_classes' in weights_file:
                    num_clusters = 4
                kmeans = GaussianMixture(num_clusters)
                kmeans.fit(embeddings)
                y_hat = kmeans.predict(embeddings)
                y = [i[0] for i in labels]
                max_cor = 0
                y_hats = self.generate_y_hats(y_hat, num_clusters)
                for yh in y_hats:
                    num_cor = 0
                    for i, j in zip(yh, y):
                        if i == j:
                            num_cor +=1
                    max_cor = max(num_cor, max_cor)
                print(100*max_cor / len(triplets))
                accuracies.append(100*max_cor / len(triplets))
        print(accuracies)

    def generate_y_hats(self, y_hat, num_clusters):
        import itertools
        perms = list(itertools.permutations(range(num_clusters)))
        y_hats = []
        for perm in perms:
            this_yhat = [perm[i] for i in y_hat]
            y_hats.append(this_yhat)
        return y_hats
        
    def eval_consistancy(self, triplets, weights):
        embeddings = []
        weights_name=weights
        weights = np.load(weights)[-1]
        reversed_embeddings = []
        for im in tqdm(triplets):
            flattened = []
            for i, j, k in zip(im[0], im[1], im[2]):
                flattened.append(i)
                flattened.append(j)
            z_out = self.embed(self, weights, np.array(flattened))
            embeddings.append(z_out)
            flattened = []
            for i, j, k in zip(im[0], im[1], im[2]):
                flattened.append(j)
                flattened.append(i)
            z_out = self.embed(self, weights, np.array(flattened))
            reversed_embeddings.append(z_out)
        losses = []
        for i, j in zip(embeddings, reversed_embeddings):
            #print(i,j)
            #losses.append(float(np.abs(i[0] - j[2]) + float(np.abs(i[1] - j[3]))))
            losses.append(float(np.abs(i[0] - j[2]) + float(np.abs(i[1] - j[3])) + float(np.abs(i[2] - j[0])) + float(np.abs(i[3] - j[1]))))
        print(f"Losses: {np.mean(losses)}")
        return losses
        