import argparse
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
    # parser.add_argument('--model_name', default='intfloat/e5-base', type=str)
    parser.add_argument('--tokenizer', default='intfloat/e5-base', type=str)
    parser.add_argument('--max_seq_length', default=512, type=int, help="Model's maximum accepted sequence length")
    # parser.add_argument('--embedding_dim', default=768, type=int)
    # parser.add_argument('--batch_size_dataloader', default=256, type=int)
    # parser.add_argument('--batch_size_inference', default=512, type=int)
    # parser.add_argument('--world_size', default=torch.cuda.device_count(), type=int)
    # parser.add_argument('--post_processing_fxn', default='post_processing_bert', type=str)
    # parser.add_argument('--collator', default="e5", type=str)
    args = parser.parse_args()

    save_dir = args.save_dir
    split = args.split
    dataset_name = args.name
    if args.streaming_remote.lower() == "none":
        args.streaming_remote = None
    remote = args.streaming_remote
    local = args.streaming_local

    instruction = args.instruction
    tokenizer_name = args.tokenizer

    old_dataset = StreamingDataset(
    local=local,
    remote=remote,
    split=split,
    shuffle=False
    )

