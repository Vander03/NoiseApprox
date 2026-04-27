import triplet_generator
import sliq
import basic_net
import numpy as np
if __name__ == '__main__':
    dataset='MNIST'
    num_qubits = 6
    # Preprocessing
    triplets, labels = triplet_generator.generate_pca_triplets(dataset, label_space=4, num_triplets=1000, testing=False)
    network = basic_net.Triplet(num_qubits)
    network.train(triplets)

    # network.plot_loss()
    network.plot_embeddings(triplets, labels)
    # load test data
    t_triplets, t_lebals = triplet_generator.generate_pca_triplets(dataset, label_space=2, num_triplets=1000, testing=True)
    network.plot_embeddings(triplets, labels, t_triplets, t_lebals)