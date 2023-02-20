import argparse
import numpy as np
from tqdm import tqdm
import os
import torch
import pickle
import pandas as pd
import random



def _encode_shard(self, shard, sorted_clusters_path):
    def init_memmap_laion440_images_embs():
        STR_DTYPE = "S24"
        EMB_MEMORY = "/checkpoints/amroabbas/datapruning/open-clip-encodings/laion440m/laion440m_ViT-B-16_embs.npy"
        dataset_size = 437444349
        SHAPE = (dataset_size, 512)
        embs = np.memmap(EMB_MEMORY, dtype='float32', mode='r', shape=SHAPE)

        return embs
    
    embs = init_memmap_laion440_images_embs()
    

    

    end = min(50000, shard+self.args.clusters_per_job)

    def contains_duplicates(arr):
        return len(np.unique(arr)) != len(arr)

    for cluster_id in tqdm(range(shard, end)):
        big_clusters = [24224, 31847] 
        if cluster_id in big_clusters:
            continue     
        dict_file_loc = os.path.join(self.args.save_loc, f"dicts/cluster_{cluster_id}.pt")
        df_file_loc = os.path.join(self.args.save_loc, f"dataframes/cluster_{cluster_id}.pkl")

        if os.path.exists(dict_file_loc) and os.path.exists(df_file_loc):
            print(f"{dict_file_loc} exists, moving on")
            continue
        ## -- load cluster i representations
        cluster_i = np.load(os.path.join(sorted_clusters_path, f'cluster_{cluster_id}.npy'))
        # 1) store cluster size
        cluster_size = cluster_i.shape[0]

        ## -- indices for cluster items in the dataset
        cluster_ids = cluster_i[:, 1].astype('int32')

        cluster_reps = embs[cluster_ids]
        cluster_reps = torch.tensor(cluster_reps)

        ## -- compute pairwise cos sim between cluster items, then replace to diagonal with zeros to ignore self similarity
        pair_w_sim_matrix = (cluster_reps @ (cluster_reps.T))
        pair_w_sim_matrix.fill_diagonal_(0.0)
        assert pair_w_sim_matrix.shape[0]==pair_w_sim_matrix.shape[1]

        if True:
            ## -- get paths to cluster i images
            image_urls = cluster_i[:, 0]

            ## -- make sure all the paths are unique this ensure that the duplicates are really stored many time times on memory
            assert not contains_duplicates(image_urls)
            
            ## -- 2) compute the sum of all pairwise sim values exept the diagonal (diagonal items = 1.0)
            # sim_sum = torch.sum(pair_w_sim_matrix)-(cluster_size*1.0) # torch.sum(pair_w_sim_matrix.flatten()[1:].view(n-1, n+1)[:,:-1])
            # avg_pair_w_sim = sim_sum/((cluster_size**2)-cluster_size)
            # sim_sum = torch.sum(triu)
            # avg_pair_w_sim = sim_sum/(cluster_size*(cluster_size-1)/2)
            avg_sim_to_others_list = (1/(cluster_size-1))*(torch.sum(pair_w_sim_matrix, dim=0)) # -- array of shape (cluster_size x 1)

            ##-- 3) compute max pairwise similarity
            max_pair_w_sim_list = torch.max(pair_w_sim_matrix, dim=0)[0] # -- array of shape (cluster_size x 1)
            min_pair_w_sim_list = torch.min(pair_w_sim_matrix, dim=0)[0] # -- array of shape (cluster_size x 1)
            std_pair_w_sim = pair_w_sim_matrix.std()

            ## -- 4) average value of cos similarity to cluster centroid
            avg_sim_to_cent = (1-cluster_i[:, 2].astype('float32')).mean()
            std_sim_to_cent = (1-cluster_i[:, 2].astype('float32')).std()

        ## -- We need upper tringular matrix because (1)we don't need to look at self sim (always=1) (2)we need the compinations not permutations
        triu = torch.triu(pair_w_sim_matrix, diagonal=1)
                    
        ## -- if the max sim between one example and any other example is > 1-eps, remove this example
        M = torch.max(triu, dim=0)[0]

        points_to_remove_df = pd.DataFrame()

        values_for_eps = {}

        for eps in self.args.eps_list:
            ## -- 5) We need to remove a point from the dataset when its pairwise similarity to other point is > 1-ebs
            eps_points_to_remove = M>1-eps
            points_to_remove_df[f'eps={eps}'] = eps_points_to_remove
            
            if True:
                ## -- 6) num duplicates in this cluster
                eps_num_duplicates = sum(eps_points_to_remove).item()

                ## -- 7) duplicates ratio %
                eps_duplicates_ratio = 100*eps_num_duplicates/cluster_size 
                ## -- 8) number of similar points to each point
                eps_num_duplicates_for_each_point = 1 + torch.sum(pair_w_sim_matrix>1-eps, dim=0) # -- array of shape (cluster_size x 1)
                ## -- store all the value computed for this eps
                values_for_eps = {'duplicates_ratio': eps_duplicates_ratio, 'num_duplicates': eps_num_duplicates,
                                    'eps_num_duplicates_for_each_point': eps_num_duplicates_for_each_point}
                                                
        num_duplicates_in_cluster_i = {
                                        'values_for_eps': values_for_eps,
                                        'cluster_size': cluster_size,
                                        'cluster_id': cluster_id,
                                        'avg_sim_to_cent': avg_sim_to_cent,
                                        'std_sim_to_cent': std_sim_to_cent,
                                        'std_pair_w_sim': std_pair_w_sim,
                                        'avg_sim_to_others_list': avg_sim_to_others_list,
                                        'max_pair_w_sim_list': max_pair_w_sim_list,
                                        'min_pair_w_sim_list': min_pair_w_sim_list
                                        }

        if self.args.save_loc != "":
            ## -- save dict
            torch.save(num_duplicates_in_cluster_i, dict_file_loc)
            ## --save df
            with open(df_file_loc, 'wb') as file:
                pickle.dump(points_to_remove_df, file)


        print('DONE cluster_id ', cluster_id)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='k-means on embeddings')
    parser.add_argument('--emb_path', type=str, help='Path to embeddings file, a n x d numpy memmap where n = n samples and d = dimensionality')
    parser.add_argument('--centroid_path', type=str, help='Path to centroids file, the output of kmeans.py')    
    parser.add_argument('--n_samples', type=int, help='Number of samples in embedding array')
    parser.add_argument('--dim', type=int, help='Embedding dimensionality')
    parser.add_argument('--save_directory', type=str, default='/tmp/centroids/')
    parser.add_argument('--filename_base', type=str, default='')
    parser.add_argument('--n_iter', type=int, default=40, help='Number of kmeans iterations')
    args = parser.parse_args()

    # The Pile
    # train
    # n_samples = 210607728
    # val
    # n_samples = 214670

    # C4
    # train
    # n_samples = 364868892
    # val
    # n_samples = 364608

    args.emb_path = "/tmp/pile_val_e5base_embeddings.npy"
    args.centroid_path = "/tmp/pile_val_clusters.pkl"
    args.n_samples = 214670
    args.dim = 768

    emb_array = np.memmap(args.emb_path, dtype='float32', mode='r', shape=(args.n_samples, args.dim))
    with open(args.centroid_path, 'rb') as handle:
        centroids = pickle.load(handle)
    

    # TODO: Attempt to infer n_samples from dim if n_samples not provided (and vice versa)
    sample_ids = np.array([])
    if args.sample_ids == "index":
        # TODO: Read sample IDs from file
        sample_ids = np.arange(args.n_samples)