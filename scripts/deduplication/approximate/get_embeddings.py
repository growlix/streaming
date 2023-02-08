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
from torch.profiler import profile, record_function, ProfilerActivity
import datetime
import logging

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

def chunk_tokens_bert(text: List[str], tokenizer: Callable, chunk_size: int, instruction: Union[str, None]=None):
    # Tokenize instruction (if we have one)
    if instruction is not None:
        instruction_tokenized = tokenizer(instruction, truncation=False, max_length=False, return_tensors='pt')
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
    input_tokenized = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt', return_overflowing_tokens=True, padding=True)
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

class E5Collator:
    def __init__(self, tokenizer=None, chunk_size: int=512, instruction: Union[str, None]=None) -> None:
        
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.instruction = instruction

    def __call__(self, samples: List) -> Tuple[Dict[str, Tensor], Tensor]:
        sample_indices = []
        texts = []
        for sample in samples:
            texts.append(sample[0]['text'])
            sample_indices.append(sample[1])
        samples_tokenized = chunk_tokens_bert(text=texts, tokenizer=self.tokenizer, chunk_size=self.chunk_size, instruction=self.instruction)
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
    try:
        sample_indices = sample_indices.to(curr_device)
    except Exception as e:
        print(f'sample_indices: {sample_indices}]\nwith shape {sample_indices.shape}]\non device {sample_indices.device}')
        print(f'curr device: {curr_device}')
        raise e
    uniques, inverse = sample_indices.unique(return_inverse=True)
    reduce_inds = inverse.view(inverse.size(0), 1).expand(-1, seq_embeddings.size(1))
    mean_embeddings = torch.zeros(uniques.size(0), seq_embeddings.size(1), device=curr_device)
    # print(f"Device rank: {rank}. {reduce_inds} {embeddings}")
    mean_embeddings.scatter_reduce_(dim=0, index=reduce_inds, src=seq_embeddings, reduce='mean', include_self=False)
    return mean_embeddings

def post_process_bert(
    last_hidden_state: Tensor,
    # attention_mask: Tensor,
    sample_indices: Tensor,
    **kwargs
    ):

    # seq_embeddings = avg_pool_tokens(last_hidden_state, attention_mask)
    doc_embeddings = avg_sequences(last_hidden_state, sample_indices)
    doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)

    return doc_embeddings

def subdivide_batch(batch: Mapping, batch_size: int):
    n_samples = batch['input_ids'].shape[0]
    # batches = []
    for i in range(0, n_samples, batch_size):
        if i + batch_size > n_samples:
            inds = (i, n_samples)
        else:
            inds = (i, i + batch_size)
        yield {k: v[inds[0]:inds[1], :] for k, v in batch.items()}
    # return batches


class WrappedE5(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

    @property
    def device(self):
        return next(self.parameters()).device

    def avg_pool_tokens(self, last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        attention_mask = attention_mask.to(last_hidden_state.device)
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        out = self.backbone(*args, **kwds)
        out['pooler_output'] = None
        out['last_hidden_state'] = self.avg_pool_tokens(out['last_hidden_state'], kwds['attention_mask'])
        return out


def merge_memmaps(filename: str, rank0_filename: str, destination_memmap: np.memmap) -> None:
    memmap_files = [f for f in os.listdir() if filename in f]
    memmap_files.remove(rank0_filename)
    for memmap_file in memmap_files:
        source_memmap = np.memmap(memmap_file, dtype='float32', mode="r", shape=destination_memmap.shape)
        source_inds = source_memmap[:,0].nonzero()[0]
        destination_memmap[source_inds,:] = source_memmap[source_inds,:]
        destination_memmap.flush()
        del source_memmap
        os.remove(memmap_file)
        

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
    device_ids: Union[List, None]=None,
    parallel_strategy: str='dp',
    **kwargs,
    ):

    if parallel_strategy == 'mp':
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
        model = WrappedE5(model_name).to(rank)
    if parallel_strategy == 'mp':
        dist.barrier()
    if rank > 0:
        model = WrappedE5(model_name).to(rank)
    if parallel_strategy == 'mp':
        dist.barrier()

    if parallel_strategy == 'dp':
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()

    dataset_len = dataset.index.total_samples
    if rank == 0:
        logging.basicConfig(filename='rank0.log', encoding='utf-8', level=logging.DEBUG)
    # File that we write to that contains embeddings
    rank_filename = f'rank{rank}_{file_name}'
    emb_array = np.memmap(rank_filename, dtype='float32', mode='w+', shape=(int(dataset_len), embedding_dim))

    if rank == 0:
        pbar = tqdm(total=dataset_len)
        # total = dataset_len
    else:
        pbar = None
    for batch, sample_indices in dataloader:
        if parallel_strategy == 'mp':
            batch = batch_to_device(batch, model.device) # Move this to microbatch level
        if isinstance(batch, Mapping) and batch['input_ids'].shape[0] > batch_size_inference:
            # microbatches = subdivide_batch(batch, batch_size_inference)
            microbatches_out = []
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    for microbatch in subdivide_batch(batch, batch_size_inference):
                        microbatch_out = model(**microbatch)
                        # microbatch_out['last_hidden_state'] =
                        # avg_pool_tokens(microbatch_out['last_hidden_state'],
                        # microbatch['attention_mask'])
                        microbatch_out = batch_to_device(microbatch_out, torch.device('cpu'))
                        microbatches_out.append(microbatch_out)
            # del microbatches
            # Aggregate microbatches
            out = microbatches_out[0]
            for i, microbatch_out in enumerate(microbatches_out[1:], 1):
                for k, v in microbatch_out.items():
                    # new_shape = v.shape
                    # existing_shape = out[k].shape
                    # try:
                    if v is not None:
                        out[k] = torch.cat([out[k], v], 0)
                    # except torch.cuda.OutOfMemoryError as e:
                    #     print(f'Concating tensor of shapes {existing_shape} and {new_shape}\n')
                    #     raise e
                microbatches_out[i] = []
            del microbatches_out
        else:
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    out = model(**batch)
                    out = batch_to_device(out, torch.device('cpu'))
                # out['last_hidden_state'] = avg_pool_tokens(out['last_hidden_state'], batch['attention_mask'])
        embeddings = post_processing(**out, **batch, sample_indices=sample_indices)
        sample_indices_unique = sample_indices.unique()

        zero_embeddings = torch.where(embeddings[:,0] == 0)[0]
        if len(zero_embeddings) > 0:
            logging.warning(f"zero-value embeddings for samples {sample_indices_unique[zero_embeddings]}")
        

        # if rank > 0:
        #     dist.gather(embeddings.to)
        #     dist.gather(sample_indices_unique.to(rank))
        # if rank == 0 and parallel_strategy == 'mp':
        #     uniques = sample_indices_unique.to(rank)
        #     embeddings_list = [torch.zeros_like(embeddings) for _ in range(world_size)]
        #     inds_list = [torch.zeros_like(uniques) for _ in range(world_size)]
        #     dist.gather(embeddings, embeddings_list)
        #     dist.gather(uniques, inds_list)
        #     embeddings_gathered = torch.cat(embeddings_list, dim=0).cpu().numpy()
        #     inds_gathered = torch.cat(inds_list, dim=0).cpu().numpy()
        #     emb_array[inds_gathered,:] = embeddings_gathered
        # else:
        #     emb_array[sample_indices_unique.cpu().numpy(),:] = embeddings.cpu().numpy()
        emb_array[sample_indices_unique.numpy(),:] = embeddings.numpy()

        if rank == 0:
            assert type(pbar) is tqdm
            update_size = len(sample_indices_unique)
            if parallel_strategy == 'mp':
                update_size *= world_size
            pbar.update(update_size)
            current = pbar.format_dict['n']
            if current == update_size or current%(update_size * 10):
                total = pbar.format_dict['total'] 
                elapsed = str(datetime.timedelta(seconds=pbar.format_dict['elapsed'])).split('.')[0]
                est_total = str(datetime.timedelta(seconds=pbar.format_dict["total"]/pbar.format_dict["rate"])).split(".")[0]
                logging.info(f'{current} of {total} Samples ---- Elapsed: {elapsed} ---- Estimated Total: {est_total}')

    if rank == 0:
        assert type(pbar) is tqdm
        
        pbar.close()
    emb_array.flush()
    if parallel_strategy == 'mp':
        dist.barrier()

    if rank == 0:
        if parallel_strategy == 'mp':
            print("Merging arrays from different ranks")
            merge_memmaps(filename=file_name, rank0_filename=rank_filename, destination_memmap=emb_array)
        os.rename(rank_filename, file_name)
    if parallel_strategy == 'mp':
        dist.barrier()
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
    parser.add_argument('--batch_size_dataloader', default=320, type=int)
    parser.add_argument('--batch_size_inference', default=640, type=int)
    parser.add_argument('--world_size', default=torch.cuda.device_count(), type=int)
    parser.add_argument('--post_processing_fxn', default='post_processing_bert', type=str)
    parser.add_argument('--collator', default='e5', type=str)
    parser.add_argument('--parallel_strategy', default='dp', type=str, help="mp (multiprocessing) or dp (Data Parallel)")
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
    if args.parallel_strategy == 'mp':
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
                None,
                args.parallel_strategy,
            ],
            nprocs=world_size
        )    
    elif args.parallel_strategy == 'dp':
        device_ids = list(range(world_size))
        args.world_size = 1
        args.collator = collator
        do_the_thing(rank=0, **vars(args), post_processing=post_processing_fxn, device_ids=device_ids)
