import argparse
import os
import pickle

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from streaming.base import MDSWriter, StreamingDataset
from get_embeddings import StreamingDatasetIndexed


def running_average(avg: float, new_val: float, count: int):
    return avg*(count-1)/count + new_val/count


def write_and_prune(
    streaming_remote: str,
    streaming_local: str,
    split: str,
    save_dir: str,
    name: str,
    similarity_file: str,
    keep_fraction: float,
    source_key: str,
):

    with open(similarity_file, 'rb') as handle:
        similarities = pickle.load(handle)

    try:
        keep_threshold = similarities['quantiles'][keep_fraction]
    except KeyError as e:
        print(f'{keep_fraction} quantile not present in {similarity_file}. Please use one of:\n')
        for q in similarities['quantiles'].keys():
            print(q)
        return e

    def collate_fn(batch):
        return batch

    # Instantiate old dataset
    old_dataset = StreamingDatasetIndexed(
        local=streaming_local,
        remote=streaming_remote,
        split=split,
        shuffle=False
        )

    dataloader = DataLoader(old_dataset, batch_size=64, num_workers=8, collate_fn=collate_fn)

    # Get params for new dataset from old dataset
    columns = {k: v for k, v in zip(old_dataset.shards[0].column_names, old_dataset.shards[0].column_encodings)}
    compression = old_dataset.shards[0].compression
    hashes = old_dataset.shards[0].hashes
    size_limit = old_dataset.shards[0].size_limit

    # TODO: Use UIDs instead of index
    # Write deduplicated samples
    removed = 0
    data_stats = {
        'source': {},
        'n_samples': 0,
        'mean_length': 0,
    }
<<<<<<< HEAD
    i = 0
    with MDSWriter(dirname=save_dir, columns=columns, compression=compression, hashes=hashes, size_limit=size_limit) as out:
        for batch in tqdm(dataloader):
            for sample, index in batch:
            # for i, sample in tqdm(enumerate(old_dataset), total=len(old_dataset)):
                if similarities['similarities'].get(index, 0) < keep_threshold:
                    out.write(sample)
                    data_stats['n_samples'] += 1
                    sample_len = len(sample['text'])
                    data_stats['mean_length'] = running_average(data_stats['mean_length'], sample_len, i+1)
                    if source_key in sample.keys():
                        source = sample[source_key]
                        source_stats = data_stats['source'].get(source, {'n_samples': 0, 'mean_length': 0})
                        source_stats['n_samples'] += 1
                        source_stats['mean_length'] = running_average(source_stats['mean_length'], sample_len, i+1)
                        data_stats['source'][source] = source_stats
                else:
                    removed += 1
                if i%2000000 == 0:
                    print(f'\nRemoved {removed} of {i+1} samples {removed/(i+1):.4f}')
                i += 1
    print(f'\nRemoved {removed} of {len(old_dataset)} samples  {removed/len(old_dataset):.4f}')
=======
    n_iterated = 0
    with MDSWriter(dirname=save_dir, columns=columns, compression=compression, hashes=hashes, size_limit=size_limit) as out:
        for i, sample in tqdm(enumerate(old_dataset), total=len(old_dataset)):
            n_iterated += 1
            if similarities['similarities'].get(i, 0) < keep_threshold:
                out.write(sample)
                data_stats['n_samples'] += 1
                sample_len = len(sample['text'])
                data_stats['mean_length'] = running_average(data_stats['mean_length'], sample_len, i+1)
                if source_key in sample.keys():
                    source = sample[source_key]
                    source_stats = data_stats['source'].get(source, {'n_samples': 0, 'mean_length': 0})
                    source_stats['n_samples'] += 1
                    source_stats['mean_length'] = running_average(source_stats['mean_length'], sample_len, i+1)
                    data_stats['source'][source] = source_stats
            else:
                removed += 1
            if i%2000000 == 0:
                print(f'\nRemoved {removed} of {i+1} samples {removed/(i+1):.4f}')
    print(f'\nRemoved {removed} of {n_iterated} samples  {removed/(n_iterated):.4f}')
>>>>>>> 2a3f0685f54fe798ab2f03c132af1340f85621ca

    savename = os.path.join(save_dir, 'data_stats.pkl')
    with open(savename, 'wb') as handle:
        pickle.dump(data_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\nSaved data stats to {savename}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write and prune dataset')
    parser.add_argument('--streaming_remote', type=str, default="None")
    parser.add_argument('--streaming_local', type=str, default="/tmp/streaming_dataset")
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--similarity_file', type=str)
    parser.add_argument('--keep_fraction', type=float)
    parser.add_argument('--source_key', type=str, default='pile_set_name', help='Key indicating data source of sample')
    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, args.split)
    write_and_prune(**vars(args))