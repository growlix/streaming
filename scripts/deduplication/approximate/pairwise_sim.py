import argparse
import numpy as np
from tqdm import tqdm
import os
import torch
import pickle
import torch.multiprocessing as mp
from functools import partial
from typing import Mapping


def compute_similarities(
    centroid_data: Mapping,
    embedding_path: str,
    n_samples: int,
    dim: int,
    reduction: str="max"
    ):
    embeddings = np.memmap(embedding_path, dtype='float32', mode='r', shape=(n_samples, dim))
    centroid_inds = centroid_data['inds']
    centroid_embeddings = torch.tensor(embeddings[centroid_inds,:])
    centroid_size = centroid_embeddings.shape[0]
    similarity_vector = torch.tensor([])
    if centroid_size > 2:
        similarity_matrix = (centroid_embeddings @ (centroid_embeddings.T))
        similarity_matrix.fill_diagonal_(0.0)
        assert similarity_matrix.shape[0]==similarity_matrix.shape[1]        
        triu = torch.triu(similarity_matrix, diagonal=1)
        # TODO: Choose between mean or max
        similarity_vector = triu[:,1:].max(dim=0)[0]
    return similarity_vector.numpy(), centroid_data['labels']


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='k-means on embeddings')
    parser.add_argument('--emb_path', type=str, help='Path to embeddings file, a n x d numpy memmap where n = n samples and d = dimensionality')
    parser.add_argument('--centroid_path', type=str, help='Path to centroids file, the output of kmeans.py')
    parser.add_argument('--sample_ids', type=str, default='index', help='Path to file containing index:label mapping, or "index" to just use index in embedding array')
    parser.add_argument('--n_samples', type=int, help='Number of samples in embedding array')
    parser.add_argument('--dim', type=int, help='Embedding dimensionality')
    parser.add_argument('--save_directory', type=str, default='/tmp/similarities/')
    parser.add_argument('--filename_base', type=str, default='')
    parser.add_argument('--quantiles', nargs='+', default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], help="Quantiles to compute")
    parser.add_argument('--subsample', type=int, default=-1, help="Number of clusters to subsample. Useful if you want a faster distribution estimate and don't need to compute distances for all samples.")
    args = parser.parse_args()

    # The Pile
    # train
    # n_samples = 210607728
    # val
    # args.n_samples = 214670

    # C4
    # train
    # n_samples = 364868892
    # val
    # n_samples = 364608

    # args.emb_path = "/tmp/pile_val_embeddings.npy"
    # args.centroid_path = "/tmp/pile_val_clusters.pkl"
    # args.n_samples = 210607728
    # args.dim = 768

    # emb_array = np.memmap(args.emb_path, dtype='float32', mode='r', shape=(args.n_samples, args.dim))
    with open(args.centroid_path, 'rb') as handle:
        centroids = pickle.load(handle)
    
    if not os.path.exists(args.save_directory):
        os.makedirs(args.save_directory)    

    # TODO: Attempt to infer n_samples from dim if n_samples not provided (and vice versa)
    sample_ids = np.array([])
    if args.sample_ids == "index":
        # TODO: Read sample IDs from file
        sample_ids = np.arange(args.n_samples)
    
    if args.subsample == -1:
        clusters_to_sample = len(centroids)
    else:
        clusters_to_sample = args.subsample
    quantiles = args.quantiles

    # TODO: Parallelize
    all_similarities = []
    similarities_map = {}
    mapfun = partial(compute_similarities, embedding_path=args.emb_path, n_samples=args.n_samples, dim= args.dim)
    with mp.Pool() as pool:
        for cluster_similarity, cluster_labels in tqdm(pool.imap_unordered(mapfun, centroids[:clusters_to_sample]), total=clusters_to_sample):
            all_similarities.append(cluster_similarity)
            # TODO: Check for collisions
            for label, similarity in zip(cluster_labels, cluster_similarity):
                similarities_map[label] = similarity
    
    all_similarities = np.concatenate(all_similarities, axis=0)
    similarity_quantiles = np.quantile(all_similarities, quantiles)
    quantile_map = {}
    for computed, q in zip(similarity_quantiles, quantiles):
        quantile_map[q] = computed
    save_file = {
        "quantiles": quantile_map,
        "similarities": similarities_map
    }
    savename = os.path.join(args.save_directory, f'{args.filename_base}_max_similarity.pkl')
    with open(savename, 'wb') as handle:
        pickle.dump(save_file, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved similarity data to {savename}')
