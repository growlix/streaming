import argparse
import os
import pickle

from torch.utils.data import DataLoader
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
    parser.add_argument('--include_list', nargs='+', default=[], help=f"Subset(s) to use. Subsets are: {SUBSET_MAP}")
    parser.add_argument('--block_list', nargs='+', default=[], help=f"Subset(s) to block. Subsets are : {SUBSET_MAP}")
    parser.add_argument('--split', type=str)
    args = parser.parse_args()

    if args.include_list and args.block_list:
        raise ValueError('Arguments --include_list and --block_list are mutually exclusive. Please pick one.')

    use_block_list = False
    passed_subsets = []
    if args.include_list:
        passed_subsets = args.include_list
    elif args.block_list:
        passed_subsets = args.block_list
        use_block_list = True
    subsets = []
    for subset in passed_subsets:
        try:    
            subsets.append(SUBSET_MAP[subset.lower()])
        except KeyError:
            print(f'Subset {subset} not recognized. Please use one or more of the following subsets:')
            for k, v in SUBSET_MAP.items():
                print(f'\n{k}: {v}')
    
    args.save_dir = os.path.join(args.save_dir, args.split)
    
    def collate_fn(batch):
        return batch    
    
    full_dataset = StreamingDataset(
        local=args.streaming_local,
        remote=args.streaming_remote,
        split=args.split,
        shuffle=False
        )    
    dataloader = DataLoader(full_dataset, batch_size=64, num_workers=8, collate_fn=collate_fn)    

    # Get params for new dataset from old dataset
    columns = {k: v for k, v in zip(full_dataset.shards[0].column_names, full_dataset.shards[0].column_encodings)}
    compression = full_dataset.shards[0].compression
    hashes = full_dataset.shards[0].hashes
    size_limit = full_dataset.shards[0].size_limit

    written_samples = 0
    with MDSWriter(out=args.save_dir, columns=columns, compression=compression, hashes=hashes, size_limit=size_limit) as out:
        for batch in tqdm(dataloader):
            for sample in batch:
                if use_block_list and sample['pile_set_name'] not in subsets:
                    out.write(sample)
                    written_samples += 1
                elif not use_block_list and sample['pile_set_name'] in subsets:
                    out.write(sample)
                    written_samples += 1
    print(f'Previous dataset size: {len(full_dataset)}. Wrote {written_samples} samples, excluded {len(full_dataset)-written_samples} samples.')

    # savename = os.path.join(args.save_dir, 'data_stats.pkl')
    # with open(savename, 'wb') as handle:
    #     pickle.dump(data_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # print(f'\nSaved data stats to {savename}')    
