import os
from streaming.base.dataset import StreamingDataset
import torch
from torch import device, Tensor
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import argparse
import numpy as np
from typing import Any, Callable, Dict, List, Mapping, Tuple, Union
from transformers import AutoTokenizer, AutoModel

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = f'{world_size}'
    os.environ['RANK'] = f'{rank}'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

# Need dataset to return index, because we need to associate embedding w/ sample index
class StreamingDatasetIndexed(StreamingDataset):
    def __getitem__(self, index: int) -> Tuple[Dict[str, Any], int]:
        """Get sample by global index.
        Args:
            index (int): Sample index.
        Returns:
            Dict[str, Any]: Column name with sample data.
        """
        shard, index_in_shard = self.index.find_sample(index)
        reader = self.shards[shard]
        return reader[index_in_shard], index


# Embedding model's max sequence length in 512, so we need a function that will divide
# each document into chunks of length ≤512
def chunk_text(text: str, chunk_size: int=512, instruction: str="Represent the document for clustering:"):
    chunks = []
    for start in range(0, len(text), chunk_size):
        end = start + chunk_size
        if len(text) < end:
            curr_seq = text[start:]
        else:
            curr_seq = text[start:end]
        chunks.append([instruction, curr_seq])
    return chunks


def batch_to_device(batch, target_device: device):
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

# Custom collate function that will get our data into the appropriate input format for the
# encoder
class E5Collator:
    def __init__(self, tokenizer=None, chunk_size: int=512, instruction: Union[str, None]=None) -> None:
        
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.instruction = instruction

    def _chunk_tokens(self, text: List[str], chunk_size=None, instruction=None):
        # Make sure there's a tokenizer
        assert self.tokenizer
        if chunk_size is None:
            chunk_size = self.chunk_size
            # If we never set a chunk size, default to using the tokenizer's maximum sequence length
            if chunk_size is None:
                chunk_size = self.tokenizer.model_max_length
        if instruction is None:
            instruction = self.instruction
        # Tokenize instruction (if we have one)
        if instruction is not None:
            instruction_tokenized = self.tokenizer(instruction, truncation=False, max_length=False, return_tensors='pt')
            # Strip the beginning ([CLS]) and final ([SEP]) tokens.
            instruction_tokenized = {k: v[:,1:-1] for k, v in instruction_tokenized.items()}
            instruction_length = instruction_tokenized['input_ids'].shape[1]
        else:
            instruction_tokenized = None
            instruction_length = 0
        # Account for the size of the instruction when chunking - reduce chunk_size by
        # instruction_length
        max_length = chunk_size - instruction_length
        # Tokenize
        input_tokenized = self.tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt', return_overflowing_tokens=True, padding=True)
        # Insert the instruction into the tokenized input at index = 1 (after the [CLS]
        # token and before the input text)
        if instruction is not None:
            assert type(instruction_tokenized) is dict
            for k, instruction_value in instruction_tokenized.items():
                input_value = input_tokenized[k]
                input_tokenized[k] = torch.cat([
                    input_value[:,0:1],
                    instruction_value.repeat(input_value.shape[0], 1),
                    input_value[:, 1:]
                    ], 1)
        return input_tokenized

    def __call__(self, samples: List) -> Tuple[Dict[str, Tensor], Tensor]:
        sample_indices = []
        texts = []
        for sample in samples:
            texts.append(sample[0]['text'])
            sample_indices.append(sample[1])
        samples_tokenized = self._chunk_tokens(texts)
        _, counts = samples_tokenized['overflow_to_sample_mapping'].unique(return_counts=True)
        sample_indices = torch.tensor(sample_indices).repeat_interleave(repeats=counts)
        samples_tokenized.pop('overflow_to_sample_mapping')
        return samples_tokenized, sample_indices

def avg_pool_tokens(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def avg_sequences(seq_embeddings: Tensor, sample_indices: Tensor):
    curr_device = seq_embeddings.device
    sample_indices = sample_indices.to(curr_device)
    uniques, inverse = sample_indices.unique(return_inverse=True)
    reduce_inds = inverse.view(inverse.size(0), 1).expand(-1, seq_embeddings.size(1))
    mean_embeddings = torch.zeros(uniques.size(0), seq_embeddings.size(1), device=curr_device)
    # print(f"Device rank: {rank}. {reduce_inds} {embeddings}")
    mean_embeddings.scatter_reduce_(dim=0, index=reduce_inds, src=seq_embeddings, reduce='mean', include_self=False)
    return mean_embeddings

def post_process_bert(
    last_hidden_state: Tensor,
    attention_mask: Tensor,
    sample_indices: Tensor,
    **kwargs
    ):

    seq_embeddings = avg_pool_tokens(last_hidden_state, attention_mask)
    doc_embeddings = avg_sequences(seq_embeddings, sample_indices)
    doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)

    return doc_embeddings

def subdivide_batch(batch: Mapping, batch_size: int):
    n_samples = batch['input_ids'].shape[0]
    batches = []
    for i in range(0, n_samples, batch_size):
        if i + batch_size > n_samples:
            inds = (i, n_samples)
        else:
            inds = (i, i + batch_size)
        batches.append({k: v[inds[0]:inds[1], :] for k, v in batch.items()})
    return batches


def do_the_thing(
    rank: int,
    world_size: int,
    streaming_remote: str,
    streaming_local: str,
    save_dir: str,
    split: str,
    file_name: str,
    model_name: str,
    embedding_dim: int,
    batch_size_dataloader: int,
    batch_size_inference: int,
    collator: Callable,
    post_processing: Callable,
    ):

    setup(rank, world_size)

    dataset = StreamingDatasetIndexed(
        local=streaming_local,
        remote=streaming_remote,
        split=split,
        shuffle=False,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size_dataloader,
        shuffle=False,
        collate_fn=collator,
        num_workers=8
    )

    if rank == 0:
        model = AutoModel.from_pretrained(model_name).to(rank)
    dist.barrier()
    if rank > 0:
        model = AutoModel.from_pretrained(model_name).to(rank)
    dist.barrier()

    model.eval()

    dataset_len = torch.tensor(len(dataset), device=rank)
    dist.reduce(dataset_len,dst=0)
    if rank == 0:
        # File that we write to that contains embeddings
        emb_array = np.memmap(file_name, dtype='float32', mode='w+', shape=(int(dataset_len.item()), embedding_dim))

    if rank == 0:
        pbar = tqdm(total=len(dataloader))
    else:
        pbar = None

    for batch, sample_indices in dataloader:
        batch = batch_to_device(batch, model.device)
        if isinstance(batch, Mapping) and batch['input_ids'].shape[0] > batch_size_inference:
            microbatches = subdivide_batch(batch, batch_size_inference)
            microbatch_out = []
            with torch.no_grad():
                for microbatch in microbatches:
                    microbatch_out.append(model(**microbatch))
            del microbatches
            # Aggregate microbatches
            out = microbatch_out[0]
            for i, microbatch_out in enumerate(microbatch_out[1:], 1):
                for k, v in microbatch_out.items():
                    out[k] = torch.cat([out[k], v], 0)
                    microbatch_out[i] = []
            del microbatch_out
        else:
            with torch.no_grad():
                out = model(**batch)
        embeddings = post_processing(**out, **batch, sample_indices=sample_indices)
        sample_indices_unique = sample_indices.unique()

        if rank > 0:
            dist.gather(embeddings)
            dist.gather(sample_indices_unique.to(rank))
        if rank == 0 and world_size > 1:
            uniques = sample_indices_unique.to(rank)
            embeddings_list = [torch.zeros_like(embeddings) for _ in range(world_size)]
            inds_list = [torch.zeros_like(uniques) for _ in range(world_size)]
            dist.gather(embeddings, embeddings_list)
            dist.gather(uniques, inds_list)
            embeddings_gathered = torch.cat(embeddings_list, dim=0).cpu().numpy()
            inds_gathered = torch.cat(inds_list, dim=0).cpu().numpy()
            emb_array[inds_gathered,:] = embeddings_gathered
        elif rank == 0:
            emb_array[sample_indices_unique.cpu().numpy(),:] = embeddings.cpu().numpy()

        if rank == 0:
            assert type(pbar) is tqdm
            pbar.update(len(sample_indices_unique))

    if rank == 0:
        assert type(pbar) is tqdm
        emb_array.flush()
        pbar.close()

    cleanup()

POST_PROCESSING_FXNS = {
    'post_processing_bert': post_process_bert,
    }

COLLATORS = {
    'e5': E5Collator,
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Load a dataset.')
    parser.add_argument('--streaming_remote', type=str, default="s3://mosaicml-internal-dataset-the-pile/mds/2/")
    parser.add_argument('--streaming_local', type=str, default="/tmp/streaming_dataset")
    parser.add_argument('--save_dir', default="~/data_embeddings", type=str)
    parser.add_argument('--file_name', default='EMB_MEMORY.npy', type=str)
    parser.add_argument('--split', default='train', type=str)
    parser.add_argument('--instruction', default='query: ', type=str)
    parser.add_argument('--model_name', default='intfloat/e5-base', type=str)
    parser.add_argument('--tokenizer', default='intfloat/e5-base', type=str)
    parser.add_argument('--max_seq_length', default=512, type=int, help="Model's maximum accepted sequence length")
    parser.add_argument('--embedding_dim', default=768, type=int)
    parser.add_argument('--batch_size_dataloader', default=64, type=int)
    parser.add_argument('--batch_size_inference', default=128, type=int)
    parser.add_argument('--world_size', default=torch.cuda.device_count(), type=int)
    parser.add_argument('--post_processing_fxn', default='post_processing_bert', type=str)
    parser.add_argument('--collator', default="e5", type=str)
    args = parser.parse_args()

    if args.streaming_remote.lower() == "none":
        args.streaming_remote = None

    embedding_dim = args.embedding_dim
    instruction = args.instruction
    model_name = args.model_name
    tokenizer_name = args.tokenizer

    if tokenizer_name.lower() != "none":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = None
    
    if args.instruction.lower == "none":
        args.instruction = None

    collator = COLLATORS[args.collator.lower()](tokenizer=tokenizer, chunk_size=args.max_seq_length, instruction=args.instruction)
    post_processing_fxn = POST_PROCESSING_FXNS[args.post_processing_fxn]

    world_size = args.world_size
    # world_size = 1
    mp.spawn(
        do_the_thing,
        args=[
            world_size,
            args.streaming_remote,
            args.streaming_local,
            args.save_dir,
            args.split,
            args.file_name,
            args.model_name,
            args.embedding_dim,
            args.batch_size_dataloader,
            args.batch_size_inference,
            collator,
            post_processing_fxn,
        ],
        nprocs=world_size
    )    
