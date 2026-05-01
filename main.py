import triplet_generator
import sliq
import model
import numpy as np
import argparse

pennylane = "lightning.qubit"
qiskit = "qiskit.aer"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--message', type=str, required=True, help='describe the purpose of this run')
    args = parser.parse_args()
    
    params = ({
        "dataset": "MNIST", # Dataset the run was trained on
        "epochs": 150, # number of epochs the model was trained on
        "num_qubits": 5, # number of qubits in the circuit, width of the circuit
        "PCA_dims": 16, # number of dimensions the data was reduced to
        "backend": qiskit, # backend the circuits were simulated on
        "sim": "statevector", # statevector / density_matrix
        "shots": 300, # number of shots used during circuit eval (N/A for pennylane)
        "num_triplets": 5000, # number of triplets to generate
        "label_space": 3, # number of labels selected from the class
        "layers": 4, # depth of the circuit
        "batch_size": 64, # number of samples collected per batch
        "max_train_samples": 1000, # maximum training samples
        "embed_dims": 4, # number of qubits to measure at the end of the circuit
        "learning_rate": 0.3, # learning rate param applicable to SPSA, Grad Descent and Adam
        "perturbation_rate": 0.1, # noisy variance to perturb the parameters by, Applicable to SPSA
        "optimiser": "SPSA", # selected optimiser
        "noise_train": False, # if the model made use of Noise Training
        "noise_samp_per_batch": 2, # number of noise samples the model was exposed to per batch
        "historic_load": 10, # number of historical noise profiles to select from each backend
        "fake": False, # if the model made use of fake noise profiles
        "message": args.message, # required param, details the purpose for running the model
        "noise_profiles": [], # array to store the noise profiles seen during training
        "holdout_profiles": [], # save the holdout profiles juuust in case they change
        "results": {} # results dictionary for storing the final results
    })

    # preprocessing
    triplets, labels = triplet_generator.generate_pca_triplets(
        params['dataset'],
        label_space=params['label_space'],
        num_triplets=params['num_triplets'],
        testing=False,
        pca_dims=params['PCA_dims']
    )
    t_triplets, t_labels = triplet_generator.generate_pca_triplets(
        dataset=params['dataset'],
        label_space=params['label_space'],
        num_triplets=params['num_triplets'],
        testing=True,
        pca_dims=params['PCA_dims']
    )

    network = model.Triplet(params)
    network.train(triplets)
    network.save_experiment(triplets, labels, t_triplets, t_labels)