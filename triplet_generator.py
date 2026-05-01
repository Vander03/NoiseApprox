import os
from sklearn.decomposition import PCA
import numpy as np
from sklearn import preprocessing
import random
from PIL import Image
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
def generate_pca_triplets(dataset, label_space=10, num_triplets=5000, testing=False, pca_dims=32):
    """
    Generates PCA triplet training examples from the specified dataset with specified qubit and label space sizes.
    :param dataset: String name of folder in datasets/[dataset] containing training and testing .npy files.
    :param num_qubits: Integer number of qubits 'q' defining 2^q PCA dimensions
    :param label_space: Integer number of labels to consider, filtering examples outside the label space.
    For MNIST, this corresponds to the lowest digit being filtered out.
    :return: The triplets used to train the Triplet Model - the list of n x 3 images, indices, and labels, where 'n'
    is the number of examples used. For these 3 return values, the first column is the anchor,
    the second the positive example, and the third the negative example. 'indices[j][1]' gives the index of the positive
    example of the jth triplet in the pre-triplet data, after filtering for the label space and performing PCA
    """
    x, y = load_data(dataset, testing)
    x, y = filter_labels(x, y, label_space)
    # x = scale_data(preprocessing.normalize(x))
    x = perform_pca(x=x, pca_dims=pca_dims)
    return generate_augmented_triplets(x, y, num_triplets)


def load_data(dataset, testing):
    """
    Loads training and testing data from datasets/[dataset]/x_train.npy and datasets/[dataset]/y_train.npy
    :param dataset: the folder within the 'datasets' folder containing the npy values.
    :return: The pair of loaded Numpy arrays for training features 'x' and labels 'y'
    """
    data_path = os.path.join("datasets", dataset)
    x_path = os.path.join(data_path, "x_train.npy")
    y_path = os.path.join(data_path, "y_train.npy")
    if testing:
        x_path = os.path.join(data_path, "x_test.npy")
        y_path = os.path.join(data_path, "y_test.npy")
    x = np.load(x_path)
    y = np.load(y_path)


    return x, y


def filter_labels(images, labels, label_space):
    """
    Filters examples if their label (integer) is not below the specified label space.
    :param images: Numpy array of image data, each row is a single image example.
    :param labels: Numpy array of example labels, corresponding 1:1 with the images.
    :param label_space: The # of labels to allow, resulting in keeping examples with labels [0, label_space - 1]
    :return: The list of remaining images and list of remaining labels.
    """
    filtered_images, filtered_labels = [], []
    for im, lab in zip(images, labels):
        if lab < label_space:
            filtered_images.append(im)
            filtered_labels.append(lab)
    return filtered_images, filtered_labels


def perform_pca(x, pca_dims=32):
    """
    Performs PCA on the given training example data with the specified # of dimensions. L2 normalizes data for
    PCA fit and transformation, and returns resulting features scaled to [0, 1] range.
    :param x: Numpy array containing rows of image data training examples.
    :param pca_dims: the number of dimensions (features) to reduce each example to via PCA.
    :return: Numpy array with [0, 1] scaled result of PCA dimensionality reduction.
    """
    pca = PCA(pca_dims)
    # Normalize image data so its pythagorean sum is 1
    pca.fit(preprocessing.normalize(x))
    return scale_data(pca.transform(preprocessing.normalize(x)))


def generate_triplets(x, y, size=5000):
    """
    Collects the specified number of triplet training examples from the given training data. Triplets
    have the first two elements in the same class (label) and the third in a separate class.
    :param x: Numpy array containing rows of training examples.
    :param y: Numpy array containing labels 1:1 with training examples.
    :param size: Number of triplets to collect.
    :return: 3 lists of 3-tiples, each corresponding to a triplet.
    The first is "triplets" containing the 3 examples of training data in the triplet.
    The second is "image_indices" giving the indices within the original data given for those examples.
    The third is "labels" giving the labels corresponding to each of the 3 examples such that:
    labels[n][0] == labels[n][1] != labels[n][2]
    """
    # List of 3-tuples picked from training examples.
    
    triplets = []
    image_indices = []
    labels = []
    for _ in range(size):
        index = p_index = n_index = int(np.floor(random.random() * len(x)))
        label = y[index]
        # Find a "positive" example, one in the SAME class as "index"
        while p_index == index or y[p_index] != label:
            p_index = int(np.floor(random.random() * len(x)))
        # Find a "negative" example, one in a DIFFERENT class as "index"
        while n_index == index or y[n_index] == label:
            n_index = int(np.floor(random.random() * len(x)))
        # Once 3 examples have been found, add examples, their indices, and their labels to output lists
        triplets.append((x[index], x[p_index], x[n_index]))
        image_indices.append((index, p_index, n_index))
        labels.append((y[index], y[p_index], y[n_index]))
    return triplets, image_indices, labels

def generate_augmented_triplets(x, y, num_triplets=5000):
    triplets = []
    labels = []
    for _ in range(num_triplets):
        idx = random.randint(0, len(x) - 1)
        idy = y[idx] # save its label only for GMM evaluation
        anchor = x[idx]
        positive = augment(anchor)
        neg_idx = idx
        while neg_idx == idx:
            neg_idx = random.randint(0, len(x) - 1)
        negative = x[neg_idx]
        
        triplets.append((anchor, positive, negative))
        labels.append(idy)
    return triplets, labels

def augment(image, pca_dims=32):
    # image is already PCA-reduced, shape (32,)
    # add small gaussian noise
    noise = np.random.normal(0, 0.05, image.shape)
    augmented = np.clip(image + noise, 0, 1)
    return augmented.astype(np.float32)


def scale_data(data, scale=None, dtype=np.float32):
    """
    Scales every element in data linearly such that its minimum value is at the bottom of the specified scale,
    and the maximum value is at the top.
    :param data: Numpy array of data to scale.
    :param scale: A 2-element array, whose first value gives lower scale, second gives upper.
    :param dtype: Data type to transform scaled data to.
    :return: Data scaled as specified in the given type.
    """
    if scale is None:
        scale = [0, 1]
    min_data, max_data = [float(np.min(data)), float(np.max(data))]
    min_scale, max_scale = [float(scale[0]), float(scale[1])]
    data = ((max_scale - min_scale) * (data - min_data) / (max_data - min_data)) + min_scale
    return data.astype(dtype)

def generate_pca_triplets_batch(dataset, num_qubits, label_space=10, batch_size=2):
    x, y = load_data(dataset)
    x, y = filter_labels(x, y, label_space)
    triplets = []
    image_indices = []
    labels = []
    size = 4000
    for _ in range(size):
        index = p_index = n_index = int(np.floor(random.random() * len(x)))
        label = y[index]
        anchors_in = [index for i in range(batch_size)]
        pos_in = []
        negs_in = []
        while n_index == index or y[n_index] != label:
            n_index = int(np.floor(random.random() * len(x)))
        neg_label = y[n_index]
        # Find a "positive" example, one in the SAME class as "index"
        for i in range(batch_size):
            while p_index == index or y[p_index] != label:
                p_index = int(np.floor(random.random() * len(x)))
            pos_in.append(p_index)
        # Find a "negative" example, one in a DIFFERENT class as "index"
            while n_index == index or y[n_index] != neg_label:
                n_index = int(np.floor(random.random() * len(x)))
            negs_in.append(n_index)   
        anchors = [x[i] for i in anchors_in]
        negs = [x[i] for i in negs_in]
        pos = [x[i] for i in pos_in]

        triplets.append((anchors, negs, pos))
        image_indices.append((anchors_in, pos_in, negs_in))

        labels.append(([y[index] for i in range(batch_size)], [y[index] for i in range(batch_size)], [y[i] for i in negs_in]))
    return triplets, image_indices, labels
