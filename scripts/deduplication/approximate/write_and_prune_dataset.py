import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm

from streaming.base import MDSWriter, StreamingDataset


def write_and_prune(
    streaming_remote: str,
    streaming_local: str,
    split: str,
    save_dir: str,
    name: str,
    similarity_file: str,
    keep_fraction: float,
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

    # Instantiate old dataset
    old_dataset = StreamingDataset(
        local=streaming_local,
        remote=streaming_remote,
        split=split,
        shuffle=False
        )

    # Get params for new dataset from old dataset
    columns = {k: v for k, v in zip(old_dataset.shards[0].column_names, old_dataset.shards[0].column_encodings)}
    compression = old_dataset.shards[0].compression
    hashes = old_dataset.shards[0].hashes
    size_limit = old_dataset.shards[0].size_limit

    # TODO: Use UIDs instead of index
    # Write deduplicated samples
    with MDSWriter(dirname=save_dir, columns=columns, compression=compression, hashes=hashes, size_limit=size_limit) as out:
        for i, sample in tqdm(enumerate(old_dataset), total=len(old_dataset)):
            if similarities['similarities'].get(i, 0) < keep_threshold:
                out.write(sample)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write and prune dataset')
    parser.add_argument('--streaming_remote', type=str, default="None")
    parser.add_argument('--streaming_local', type=str, default="/tmp/streaming_dataset")
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--similarity_file', type=str)
    parser.add_argument('--keep_fraction', type=float)
    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, args.split)
    write_and_prune(**vars(args))