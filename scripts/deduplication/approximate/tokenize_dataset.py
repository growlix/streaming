import argparse
from tqdm import tqdm
from get_embeddings import chunk_tokens_bert
from streaming.base import MDSWriter, StreamingDataset
from transformers import AutoTokenizer, AutoModel

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Load a dataset.')
    parser.add_argument('--streaming_remote', type=str, default="s3://mosaicml-internal-dataset-the-pile/mds/2/")
    parser.add_argument('--streaming_local', type=str, default="/tmp/streaming_dataset")
    parser.add_argument('--save_dir', default="~/tmp/streaming_dataset_tokenized", type=str)
    parser.add_argument('--file_name', default='', type=str)
    parser.add_argument('--split', default='train', type=str)
    parser.add_argument('--instruction', default='query: ', type=str)
    parser.add_argument('--tokenizer', default='intfloat/e5-base', type=str)
    parser.add_argument('--max_seq_length', default=512, type=int, help="Model's maximum accepted sequence length")
    args = parser.parse_args()

    save_dir = args.save_dir
    split = args.split
    dataset_name = args.name
    if args.streaming_remote.lower() == "none":
        args.streaming_remote = None
    remote = args.streaming_remote
    local = args.streaming_local

    tokenizer_name = args.tokenizer
    chunk_size = args.chunk_size
    instruction = args.instruction

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    old_dataset = StreamingDataset(
    local=local,
    remote=remote,
    split=split,
    shuffle=False
    )

    # Get params for new dataset from old dataset
    columns = {k: v for k, v in zip(old_dataset.shards[0].column_names, old_dataset.shards[0].column_encodings)}
    compression = old_dataset.shards[0].compression
    hashes = old_dataset.shards[0].hashes
    size_limit = old_dataset.shards[0].size_limit

    # Write deduplicated samples
    with MDSWriter(out=save_dir, columns=columns, compression=compression, hashes=hashes, size_limit=size_limit) as out:
        for i, sample in tqdm(enumerate(old_dataset)):
            text = sample["text"]
            tokenized = chunk_tokens_bert(text=text, tokenizer=tokenizer, chunk_size=chunk_size, instruction=instruction)
            
            out.write(sample)

