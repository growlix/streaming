import ast

from tqdm import tqdm
from streaming import MDSWriter
from datasets import load_dataset


save_dir = "/nfs/streaming_datasets/pile_v2/train"
dataset = load_dataset("CarperAI/pilev2-dev", split="train", streaming=True)

columns = {'pile_set_name': 'str', 'text': 'str'}
compression = 'zstd:16'
hashes = ['sha1', 'xxh64']
size_limit = 134217728

written_samples = 0
failed_samples = 0
with MDSWriter(out=save_dir, columns=columns, compression=compression, hashes=hashes, size_limit=size_limit) as out:
    for i, sample in enumerate(tqdm(dataset)):
        try:
            meta = ast.literal_eval(sample['meta'])
            new_sample = {'pile_set_name': meta['source'], 'text': sample['text']}
            out.write(new_sample)
            written_samples += 1
        except:
           failed_samples += 1
        if i % 100000 == 0:
            print(f'Written {written_samples}. Unable to write {failed_samples}.')

print(f'Total dataset size: {written_samples + failed_samples}. Wrote {written_samples} samples, unable to write {failed_samples}.')