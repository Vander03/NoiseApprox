import sys, os, pickle, argparse
import numpy
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import triplet_generator
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()

    with open(os.path.join(args.results_dir, 'run_info.json')) as f:
        run_info = json.load(f)
    config = run_info['config']

    triplets, labels = triplet_generator.generate_pca_triplets(
        config['dataset'],
        label_space=config['label_space'],
        num_triplets=config['num_triplets'],
        testing=False
    )

    from model import Triplet
    model = Triplet(config, testing=True, results_dir=args.results_dir)
    weights = numpy.load(os.path.join(args.results_dir, 'weights.npy'), allow_pickle=True)
    model.weights = weights[-1]

    embeddings = model.get_embeddings(triplets, model.circuit)
    model.evaluate_embeddings(embeddings=embeddings, labels=labels)