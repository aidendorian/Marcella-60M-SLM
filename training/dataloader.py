from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
import torch
from src.tokenizer import Tokenizer

class TextData(IterableDataset):
    def __init__(self, dataset, tokenizer, block_size, max_samples=None, skip_chunks=0):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_samples = max_samples
        self.skip_chunks = skip_chunks
        self.chunks_yielded = 0

    def __iter__(self):
        buffer = []
        sample_count = 0
        chunks_skipped = 0

        for ex in self.dataset:
            if self.max_samples and sample_count >= self.max_samples:
                break

            text = ' '.join(ex['text']).replace(' \n', '').replace('\n', '')
            ids = self.tokenizer.encode(text)
            ids.append(self.tokenizer.eos_id)
            buffer.extend(ids)

            while len(buffer) >= self.block_size + 1:
                chunk = buffer[:self.block_size + 1]
                buffer = buffer[self.block_size + 1:]

                if chunks_skipped < self.skip_chunks:
                    chunks_skipped += 1
                    continue

                x = torch.tensor(chunk[:-1])
                y = torch.tensor(chunk[1:])
                yield x, y

                self.chunks_yielded += 1
                sample_count += 1

                if self.max_samples and sample_count >= self.max_samples:
                    return


def get_data(dataset_name: str = "draco976/wikipedia-bookcorpus",
             dataset_split: str = 'train',
             tkn_model: str = 'models/Marcella_vocab_32K.model',
             block_size: int = 512,
             batch_size: int = 2,
             num_workers: int = 0,
             pin_memory: bool = True,
             prefetch_factor: int | None = None,
             persistent_workers: bool = False,
             max_samples=None,
             resume_chunks: int = 0,
             avg_tokens_per_row: int = 520):

    dataset = load_dataset(
        dataset_name,
        split=dataset_split,
        streaming=True
    )

    if resume_chunks > 0:
        tokens_to_skip = resume_chunks * block_size
        rows_to_skip = int(tokens_to_skip / avg_tokens_per_row)

        safe_rows_to_skip = int(rows_to_skip * 0.98)
        dataset = dataset.skip(safe_rows_to_skip)

        tokens_covered_by_skip = safe_rows_to_skip * avg_tokens_per_row
        chunks_covered_by_skip = int(tokens_covered_by_skip / block_size)
        skip_chunks = resume_chunks - chunks_covered_by_skip
    else:
        skip_chunks = 0

    tokenizer = Tokenizer(tokenizer_model=tkn_model)
    text_data = TextData(
        dataset=dataset,
        tokenizer=tokenizer,
        block_size=block_size,
        max_samples=max_samples,
        skip_chunks=skip_chunks
    )

    data = DataLoader(
        text_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )

    return data, text_data