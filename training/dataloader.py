from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
import torch
from src.tokenizer import Tokenizer

class TextData(IterableDataset):
    def __init__(self, dataset, tokenizer, block_size, max_samples=None):
        super().__init__()
        
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_samples = max_samples
        
    def __iter__(self):
        sample_count = 0
        for ex in self.dataset:
            if self.max_samples and sample_count >= self.max_samples:
                break
                
            text = ' '.join(ex['text']).replace(' \n', '')
            text = text.replace('\n', '')
            ids = self.tokenizer.encode(text)
            
            for i in range(0, len(ids) - self.block_size, self.block_size):
                chunk = ids[i:i + self.block_size + 1]
                if len(chunk) == self.block_size + 1:
                    x = torch.tensor(chunk[:-1])
                    y = torch.tensor(chunk[1:])
                    yield x, y
                    sample_count += 1
                    if self.max_samples and sample_count >= self.max_samples:
                        return
        
def get_data(dataset_name:str="draco976/wikipedia-bookcorpus",
             dataset_split:str='train',
             tkn_model:str='models/Marcella_vocab_32K.model',
             block_size:int=512,
             batch_size:int=2,
             num_workers:int=4,
             pin_memory:bool=True,
             prefetch_factor:int=2,
             persistent_workers:bool=True,
             max_samples=None):
    
    dataset = load_dataset(
        dataset_name,
        split=dataset_split,
        streaming=True
    )
    
    tokenizer = Tokenizer(tokenizer_model=tkn_model)
    text_data = TextData(dataset=dataset,
                         tokenizer=tokenizer,
                         block_size=block_size,
                         max_samples=max_samples)
    
    data = DataLoader(text_data,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      prefetch_factor=prefetch_factor,
                      persistent_workers=persistent_workers)
    
    return data