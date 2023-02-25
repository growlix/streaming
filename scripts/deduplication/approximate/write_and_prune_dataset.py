import argparse
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

    # Write deduplicated samples
    with MDSWriter( dirname=save_dir, columns=columns, compression=compression, hashes=hashes, size_limit=size_limit) as out:
        for i, sample in tqdm(enumerate(old_dataset)):
            if i in remove_ex.keys():
                text = sample['text']
                for start,end in remove_ex[i][::-1]:
                    text = text[:start] + text[end:]
                sample['text'] = text
            out.write(sample)

if __name__ == 'main':
    parser = argparse.ArgumentParser(description='Write and prune dataset')
    parser.add_argument('--streaming_remote', type=str, default="None")
    parser.add_argument('--streaming_local', type=str, default="/tmp/streaming_dataset")
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--similarity_file', type=str)
    parser.add_argument('--keep_fraction', type=float)
    args = parser.parse_args()

    print("brap")

    write_and_prune(**vars(args))