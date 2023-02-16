import faiss
import torch
import time
import numpy as np
from torch.nn.functional import normalize
import logging
from tqdm import tqdm
import pandas as pd
import os
import pickle

# get all the emb - use "pretrained" openclip/ or the Kushal used - store them as np memmap - don't forget to (layer) normalize each single emb (use torch.normalize)

# kmeans clustering with faiss, choose the number of cluster, 50K, 20K

# emb_array = np.memmap('EMB_MEMORY.npy', dtype='float32', mode='w+', shape=SHAPE)
# emb_array[0] = .....
# emb_array = np.memmap('EMB_MEMORY.npy', dtype='float32', mode='r+', shape=SHAPE)
# emb_array[0]

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

def kmeans_clustering(data: np.ndarray, paths_list: np.ndarray , ncentroids: int=1000, niter: int=100, seed: int=1234,
                      verbose: bool=True, use_clusters_bal: bool=True, sim_metric: str='l2', keep_hard: bool=True,
                      use_supervised_prototypes: bool=False, Kmeans_with_cos_dist: bool=False, kmeans_obj_file="") -> list:
    '''
    Runs Kmeans clustering using "faiss", and ranks each cluster/class items using the 
    distance to the cluster centorid.
    args:
        data (embs): numpy array of shape [dataset_size x d], d is th representation vector size.
        ncentroids: number of centroids.
        niter: Kmeans clustering iterations.

    returns:
        sorted_centroids: list of list, each list represents a cluster/class examples sorted by the distance to
                        the cluster centroid. Each example is represented by tuple (img id, dist, cluster id)
    '''
    # Step 1) Compute Kmeans centroids
    # -- if kmeans_obj is not created and saved before - > create and train faiss Kmeans clustering object
    if not os.path.exists(kmeans_obj_file):
        ## -- data shoud be normalized
        d = data.shape[1]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'clustering on {device} ....')
        
        spherical = True  # spherical=True when Kmeans_with_cos_dist is True
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, seed=seed, spherical= spherical, gpu=True) # faiss.Kmeans "gpu" argument: bool or int, optional. False: don't use GPU, True: use all GPUs, number: use this many GPUs.
        st = time.time()
        kmeans.train(data)
        logger.info(f'time for clustering (mins): {(time.time()-st)/(60)}')
        if kmeans_obj_file != "":
            with open(kmeans_obj_file, 'wb') as file:
                pickle.dump(kmeans, file)
    elif kmeans_obj_file != "":
        # -- Else, load stored kmeans object
        logger.info(f'Loading faiss Kmeans pickle file from {kmeans_obj_file}')
        with open(kmeans_obj_file, 'rb') as file:
            kmeans = pickle.load(file)

    # Step 2) Find the nearest centroid for each data point, l2 distance search
    logger.info('nearest centroids...')
    st = time.time()
    dist_to_cent, nearest_cent = kmeans.index.search(data, 1) # nearest_cent: the nearest centroid for each example in data. dist_to_cent: contains the squared L2 distances.
    dist_to_cent, nearest_cent = dist_to_cent.squeeze(1), nearest_cent.squeeze(1)
    logger.info(f"time to find nearest centroids: {(time.time()-st)/60}")

    # Step 3) sort each class/cluster
    logger.info('Ranking...')
    st = time.time()
    if use_clusters_bal: # for cluster balancing
        assert use_supervised_prototypes is False, "use_clusters_bal requires use_supervised_prototypes=False"
        df = pd.DataFrame({'paths_list': paths_list, 'nearest_cent':nearest_cent, 'dist_to_cent':dist_to_cent})
        sorted_clusters = rank_within_cluster_df(data, df, kmeans.centroids, sim_metric, keep_hard, spherical)
        # sorted_clusters = rank_within_cluster(data, paths_list, kmeans.centroids, nearest_cent, dist_to_cent, sim_metric, keep_hard, spherical)
        logger.info(f'time for ranking {(time.time()-st)/60} mins')
        return sorted_clusters


def rank_within_cluster_df(data, df, centroids: np.ndarray, sim_metric: str, keep_hard: bool=True, spherical: bool=False) -> list:
    """
    Sorts each cluster items by the distance to the cluster centroid
    """

    assert sim_metric in ['cosine', 'l2'], "sim_metric should be one of ['cosine', 'l2']"

    sorted_clusters = []
    for cluster_c in tqdm(range(len(centroids))): 
        # ids = (nearest_cent==cluster_c)  # boolean array: True for the examples in cluster c
        cluster_df = df.loc[df['nearest_cent'] == cluster_c]

        cluster_items = list(cluster_df.index) #np.where(ids)[0] # ids of examples in cluster c
        if sim_metric=='cosine':
            if spherical:
                cluster_dists_to_cent = list(1 - cluster_df['dist_to_cent'])
            else:
                cluster_c_centroid = torch.Tensor(centroids[cluster_c])
                sim_to_cent = torch.nn.CosineSimilarity(dim=1)(torch.Tensor(data[cluster_items]), cluster_c_centroid)
                cluster_dists_to_cent = (1-sim_to_cent).tolist()

        elif sim_metric=='l2': # get the l2 distance from "dist_to_cent" array
            cluster_dists_to_cent = list(cluster_df['dist_to_cent'])

        cluster_label = np.full((len(cluster_df)), cluster_c).tolist()
        # in_labels = [img_name.split('_')[0] for img_name in paths_list[ids]]
        images_names = list(cluster_df['paths_list'])
        sort_descending = keep_hard
        cluster_sorted = sorted(zip(images_names, cluster_items, cluster_dists_to_cent, cluster_label), key=lambda x: x[2], reverse=sort_descending) # sort_descending = True for descending sort
            
        sorted_clusters.append(cluster_sorted) #  Descending dists. list of of list of tuples (example, dist). The ith list of tuples corresponds to cluster i
    
    return sorted_clusters


file = "/tmp/EMB_ARRAY.npy"
file = "/tmp/VAL_ARRAY.npy"
dim = 768
n_samples = 210607728
n_samples = 214670

emb_array = np.memmap(file, dtype='float32', mode="r", shape=(n_samples, dim))

# Should be able to pass sample IDs as text file
sample_ids = np.arange(n_samples)
kmeans_clustering(emb_array, sample_ids, ncentroids=50)