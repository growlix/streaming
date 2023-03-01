import argparse
import os
import pickle

from tqdm import tqdm

from streaming.base import MDSWriter, StreamingDataset

SUBSET_MAP = {
    'pubmed_abstracts': 'PubMed Abstracts',
    'arxiv': 'ArXiv',
    'pile-cc': 'Pile-CC',
    'nih': 'NIH ExPorter',
    'hackernews': 'HackerNews',
    'hn': 'HackerNews',
    'ubuntu': 'Ubuntu IRC',
    'github': 'Github',
    'youtube': 'YoutubeSubtitles',
    'wikipedia': 'Wikipedia (en)',
    'europarl': 'EuroParl',
    'opensubtitles': 'OpenSubtitles',
    'opensubs': 'OpenSubtitles',
    'books3': 'Books3',
    'dm_math': 'DM Mathematics',
    'dmmath': 'DM Mathematics',
    'openwebtext2': 'OpenWebText2',
    'owt2': 'OpenWebText2',
    'stackexchange': 'StackExchange',
    'enron': 'Enron Emails',
    'philpapers': 'PhilPapers',
    'uspto': 'USPTO Backgrounds',
    'pubmed_central': 'PubMed Central',
    'pmc': 'PubMed Central',
    'gutenberg': 'Gutenberg (PG-19)',
    'bookcorpus2': 'BookCorpus2',
    'freelaw': 'FreeLaw'
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write and prune dataset')
    parser.add_argument('--streaming_remote', type=str, default="None")
    parser.add_argument('--streaming_local', type=str, default="/tmp/streaming_dataset")
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--subsets', nargs='+', default=[], help="Subset(s) to collect. Subsets are: f{SUBSET_MAP}")
    parser.add_argument('--split', type=str)
    args = parser.parse_args()

    subsets = []
    for subset in args.subsets:
        try:    
            subsets.append(SUBSET_MAP[subset.lower()])
        except KeyError:
            print(f'Subset {subset} not recognized. Please use one or more of the following subsets:')
            for k, v in SUBSET_MAP.items():
                print(f'\n{k}: {v}')

    args.save_dir = os.path.join(args.save_dir, args.split)
    
    full_dataset = StreamingDataset(
        local=args.streaming_local,
        remote=args.streaming_remote,
        split=args.split,
        shuffle=False
        )
    
    # Get params for new dataset from old dataset
    columns = {k: v for k, v in zip(full_dataset.shards[0].column_names, full_dataset.shards[0].column_encodings)}
    compression = full_dataset.shards[0].compression
    hashes = full_dataset.shards[0].hashes
    size_limit = full_dataset.shards[0].size_limit

    with MDSWriter(dirname=args.save_dir, columns=columns, compression=compression, hashes=hashes, size_limit=size_limit) as out:
        for sample in tqdm(full_dataset, total=len(full_dataset)):
            if sample['pile_set_name'] in subsets:
                out.write(sample)
