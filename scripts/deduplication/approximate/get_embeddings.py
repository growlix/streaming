import math
import os
from collections import defaultdict
# from InstructorEmbedding import INSTRUCTOR
from streaming.base.dataset import StreamingDataset
# from sentence_transformers import SentenceTransformer
import torch
from torch import Tensor, device
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from tqdm import tqdm
from tqdm.autonotebook import trange
import argparse
import numpy as np
from typing import Any, Callable, Dict, List, Tuple, Union
from torch.nn.parallel import DistributedDataParallel as DDP
# from lorem.text import TextLorem
# import logging
# from torchvision.models import resnet50
from transformers import AutoTokenizer, AutoModel
from functools import partial

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = f'{world_size}'
    os.environ['RANK'] = f'{rank}'
    # os.environ['LOCAL_WORLD_SIZE'] = f'{world_size}'
    # os.environ['LOCAL_RANK'] = f'{rank}'

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


# Synthetic data for testing
class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, size=10000):
        super(SyntheticDataset).__init__()
        self.size = size
        self.samples = []
        lorem = TextLorem(trange=(3, 30))
        # example = lorem.text()
        for _ in range(0, self.size):
            example = lorem.text()
            self.samples.append({'text': example})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index], index


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


def chunk_tokens(tokens: List[int], chunk_size: int=512, instruction: str="Represent the document for clustering:"):
    chunks = []
    for start in range(0, len(tokens), chunk_size):
        end = start + chunk_size
        if len(tokens) < end:
            curr_seq = tokens[start:]
        else:
            curr_seq = tokens[start:end]
        chunks.append(curr_seq)
    return chunks


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
            instruction_length = 0
        # Account for the size of the instruction when chunking - reduce chunk_size by
        # instruction_length
        max_length = chunk_size - instruction_length
        # Tokenize
        input_tokenized = self.tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt', return_overflowing_tokens=True, padding=True)
        # Insert the instruction into the tokenized input at index = 1 (after the [CLS]
        # token and before the input text)
        if instruction is not None:
            for k, instruction_value in instruction_tokenized.items():
                input_value = input_tokenized[k]
                input_tokenized[k] = torch.cat([
                    input_value[:,0:1],
                    instruction_value.repeat(input_value.shape[0], 1),
                    input_value[:, 1:]
                    ], 1)
        return input_tokenized

    def __call__(self, samples):
        sample_indices = []
        texts = []
        for sample in samples:
            texts.append(sample[0]['text'])
            sample_indices.append(sample[1])
        samples_tokenized = self._chunk_tokens(texts)

        # texts = [i[0]['text'] for i in samples]
        # sample_indices = [i ]
        for sample in samples:
            index = sample[1]
            text = sample[0]['text']
            if self.tokenizer is not None:
                chunks = self._chunk_tokens(text)
                n_chunks = len(chunks['token_ids'])
                for k, v in chunks.items():
                    collated_samples[k].append(v)
                # collated_samples.extend(chunks)
            else:
                chunks = self._chunk_text(text)
                n_chunks = len(chunks)
                collated_samples.extend(chunks)
            sample_indices = np.append(sample_indices, np.repeat(index, n_chunks))
        return collated_samples, sample_indices

# Custom collate function that will chunk up our documents
def collate_samples(samples):
    collated_samples = []
    sample_indices = np.array([],dtype='uint32')
    for sample in samples:
        index = sample[1]
        text = sample[0]['text']
        chunks = chunk_text(text)
        n_chunks = len(chunks)
        sample_indices = np.append(sample_indices, np.repeat(index, n_chunks))
        collated_samples.extend(chunks)
    return collated_samples, sample_indices


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
    batch_size: int,
    collator: Callable,
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
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=8
    )

    if rank == 0:
        # model = SentenceTransformer('sentence-transformers/sentence-t5-large').to(rank)
        # model = INSTRUCTOR('hkunlp/instructor-base').to(rank)
        model = AutoModel.from_pretrained(model_name).to(rank)
        # model = resnet50().to(rank)
        # print(f'rank: {rank}')
        # print(model._target_device)
    dist.barrier()
    if rank > 0:
        # model = SentenceTransformer('sentence-transformers/sentence-t5-large').to(rank)
        # model = INSTRUCTOR('hkunlp/instructor-base').to(rank)
        model = AutoModel.from_pretrained(model_name).to(rank)
        # model = resnet50().to(rank)
        # print(f'rank: {rank}')
        # print(model._target_device)
    dist.barrier()

    if rank == 0:
        # File that we write to that contains embeddings
        emb_array = np.memmap('EMB_MEMORY.npy', dtype='float32', mode='w+', shape=(len(dataset), embedding_dim))

    if rank == 0:
        pbar = tqdm(total=len(dataloader))

    for samples, sample_indices in dataloader:
        # print(f'rank: {rank}\n{sample_indices}')
        embeddings = model.encode(samples, device=f'cuda:{rank}', convert_to_tensor=True, batch_size=512)

        # print(f"rank {rank} successful forward")
        uniques, inverse = torch.tensor(sample_indices).unique(return_inverse=True)
        reduce_inds = inverse.view(inverse.size(0), 1).expand(-1, embeddings.size(1)).to(rank)
        mean_embeddings = torch.zeros(uniques.size(0), embeddings.size(1)).to(rank)
        # print(f"Device rank: {rank}. {reduce_inds} {embeddings}")
        mean_embeddings.scatter_reduce_(dim=0, index=reduce_inds, src=embeddings, reduce='mean', include_self=False)
        embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)

        # if rank > 0:
        #     dist.gather(embeddings)
        #     dist.gather(uniques.to(rank))
        # if rank == 0 and world_size > 1:
        #     uniques = uniques.to(rank)
        #     embeddings_list = [torch.zeros_like(embeddings) for _ in range(world_size)]
        #     inds_list = [torch.zeros_like(uniques) for _ in range(world_size)]
        #     dist.gather(embeddings, embeddings_list)
        #     dist.gather(uniques, inds_list)
        #     embeddings_gathered = torch.cat(embeddings_list, dim=0).cpu().numpy()
        #     inds_gathered = torch.cat(inds_list, dim=0).cpu().numpy()
        #     emb_array[inds_gathered,:] = embeddings_gathered
        # elif rank == 0:
        #     emb_array[uniques.cpu().numpy(),:] = embeddings.cpu().numpy()

        if rank == 0:
            pbar.update(len(uniques))

    if rank == 0:
        emb_array.flush()
        pbar.close()

    cleanup()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Load a dataset.')
    parser.add_argument('--streaming_remote', type=str, default="s3://mosaicml-internal-dataset-the-pile/mds/2/")
    parser.add_argument('--streaming_local', type=str, default="/tmp/streaming_dataset")
    parser.add_argument('--save_dir', default="~/data_embeddings", type=str)
    parser.add_argument('--name', default='pile', type=str)
    parser.add_argument('--split', default='train', type=str)
    parser.add_argument('--instruction', default='query: ', type=str)
    parser.add_argument('--model_name', default='intfloat/e5-base', type=str)
    parser.add_argument('--tokenizer', default='intfloat/e5-base', type=str)
    parser.add_argument('--max_seq_length', default=512, type=int, help="Model's maximum accepted sequence length")
    parser.add_argument('--embedding_dim', default=768, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--world_size', default=torch.cuda.device_count(), type=int)
    args = parser.parse_args()

    # split = args.split
    # save_dir = args.save_dir
    # dataset_name = args.name
    if args.streaming_remote.lower() == "none":
        args.streaming_remote = None
    # remote = args.streaming_remote
    # local = args.streaming_local

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

    collator = E5Collator(tokenizer=tokenizer, chunk_size=args.max_seq_length, instruction=args.instruction)

    world_size = args.world_size
    world_size = 1
    mp.spawn(
        do_the_thing,
        args=[
            world_size,
            args.streaming_remote,
            args.streaming_local,
            args.save_dir,
            args.split,
            args.name,
            args.model_name,
            args.embedding_dim,
            args.batch_size,
            collator,
        ],
        nprocs=world_size
    )    
