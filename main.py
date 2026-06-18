import triplet_generator
import sliq
import model
import numpy as np
import argparse
from noise import noise_test
import sys


pennylane = "lightning.qubit"
qiskit = "qiskit.aer"

# arams = ({
#         "dataset": "MNIST", # Dataset the run was trained on
#         "epochs": 250, # number of epochs the model was trained on
#         "num_qubits": 5, # number of qubits in the circuit, width of the circuit
#         "PCA_dims": 32, # number of dimensions the data was reduced to
#         "backend": pennylane, # backend the circuits were simulated on
#         "sim": "statevector", # statevector / density_matrix
#         "shots": None, # number of shots used during circuit eval (N/A for pennylane)
#         "num_triplets": 5000, # number of triplets to generate
#         "label_space": 3, # number of labels selected from the class
#         "layers": 6, # depth of the circuit
#         "batch_size": 64, # number of samples collected per batch
#         "max_train_samples": 1000, # maximum training samples
#         "embed_dims": 5, # number of qubits to measure at the end of the circuit
#         "learning_rate": args.learning_rate, # learning rate param applicable to SPSA, Grad Descent and Adam
#         "cooldown_lr": None,
#         "perturbation_rate": None, # noisy variance to perturb the parameters by, Applicable to SPSA
#         "optimiser": "ADAM", # selected optimiser
#         "noise_train": args.noise_train, # if the model made use of Noise Training
#         "noise_samp_per_batch": 2, # number of noise samples the model was exposed to per batch
#         "historic_load": 20, # number of historical noise profiles to select from each backend
#         "fake": False, # if the model made use of fake noise profiles
#         "message": args.message, # required param, details the purpose for running the model
#         "noise_profiles": [], # array to store the noise profiles seen during training
#         "holdout_profiles": [], # save the holdout profiles juuust in case they change
#         "results": {}, # results dictionary for storing the final results
#         "epoch_variance": 5, # how often the variance gets updated
#         "variance_samples": 10, # number of samples to calculate variance per profile
#         "threshold": 0.10, # threshold of variance to allow during training
#         "seed": args.seed
#     })


try:
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--message', type=str, required=True, help='describe the purpose of this run')
        parser.add_argument('--noise_train', type=lambda x: x.lower() == 'true', default=True)
        parser.add_argument('--seed', type=int, default=1)
        parser.add_argument('--learning_rate', type=float, default=0.1)
        parser.add_argument('--ramp', type=float, default=50)
        parser.add_argument('--staged', type=float, default=50)
        parser.add_argument('--backend', type=str, default="fez")
        parser.add_argument('--results_dir', type=str, default=None)
        args = parser.parse_args()
        
        params = ({
            "dataset": "fashionMNIST", # Dataset the run was trained on
            "epochs": 50, # number of epochs the model was trained on
            "num_qubits": 6, # number of qubits in the circuit, width of the circuit # TODO try 11 no PCA
            "PCA_dims": 64, # number of dimensions the data was reduced to
            "backend": pennylane, # backend the circuits were simulated on
            "sim": "statevector", # statevector / density_matrix
            "shots": None, # number of shots used during circuit eval (N/A for pennylane)
            "num_triplets": 5000, # number of triplets to generate
            "label_space": 3, # number of labels selected from the class
            "layers": 6, # depth of the circuit
            "batch_size": 128, # number of samples collected per batch
            "max_train_samples": 1000, # maximum training samples
            "embed_dims": 6, # number of qubits to measure at the end of the circuit
            "learning_rate": args.learning_rate, # learning rate param applicable to SPSA, Grad Descent and Adam
            "cooldown_lr": None,
            "perturbation_rate": None, # noisy variance to perturb the parameters by, Applicable to SPSA
            "optimiser": "ADAM", # selected optimiser
            "noise_train": args.noise_train, # if the model made use of Noise Training
            "noise_samp_per_batch": 2, # number of noise samples the model was exposed to per batch
            "historic_load": 20, # number of historical noise profiles to select from each backend
            "fake": False, # if the model made use of fake noise profiles
            "message": args.message, # required param, details the purpose for running the model
            "noise_profiles": [], # array to store the noise profiles seen during training
            "holdout_profiles": [], # save the holdout profiles juuust in case they change
            "results": {}, # results dictionary for storing the final results
            "epoch_variance": 5, # how often the variance gets updated
            "variance_samples": 1000, # number of samples to calculate variance per profile
            "threshold": 0.10, # threshold of variance to allow during training
            "seed": args.seed,
            "staged_epochs": args.staged, # number of epochs before the noise training starts
            "ramp": args.ramp, # number of epochs before the noise training gets to full strength
            "metric_learning": False,
            "cluster_weight": 10,
            "backend_name": args.backend, # filter the backends to this computer
            "neighbours": 1,
            "results_dir": args.results_dir
        })

        import numpy
        numpy.random.seed(args.seed)
        np.random.seed(args.seed)
        import random
        random.seed(args.seed)

        # preprocessing

        triplets, labels = triplet_generator.generate_pca_triplets(
            params['dataset'],
            label_space=params['label_space'],
            num_triplets=params['num_triplets'],
            testing=False,
            pca_dims=params['PCA_dims'],
            metric_learning=params['metric_learning'],
            noise_train=params['noise_train']
        )
        t_triplets, t_labels = triplet_generator.generate_pca_triplets(
            dataset=params['dataset'],
            label_space=params['label_space'],
            num_triplets=params['num_triplets'],
            testing=True,
            pca_dims=params['PCA_dims'],
            metric_learning=params['metric_learning'],
            noise_train=params['noise_train']
        )


        network = model.Triplet(params)
        network.train(triplets, labels) # LABELS NOT USED IN MODEL. USED FOR PLOTS TESTING VARIANCE IN DIMENSIONS
        network.save_experiment(triplets, labels, t_triplets, t_labels)
        from visualiser import Visualiser
        vis = Visualiser(network.results_dir)
        vis.plot_loss(network.loss_history, network.clean_loss_history, network.noisy_loss_history)
        vis.evaluate_embedding_space(triplets[:500], labels[:500], network)
        # vis.measure_shift_distribution(triplets, network, ss_samples=[63, 550, 1755])
        noise_test.analyse_model(network.results_dir)
except KeyboardInterrupt:
    # Exits silently without printing the long traceback message
    sys.exit(0)