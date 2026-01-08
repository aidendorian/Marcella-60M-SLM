from sentencepiece import SentencePieceProcessor
from typing import List

class Tokenizer:
    def __init__(self,
                 tokenizer_model:str='models/Marcella_vocab_32K.model'):
        
        super().__init__()
        
        self.tokenizer = SentencePieceProcessor()
        self.tokenizer.Load(model_file=tokenizer_model)
        self.num_words = self.tokenizer.vocab_size()
        self.bos_id = self.tokenizer.bos_id()
        self.eos_id = self.tokenizer.eos_id()
        self.unk_id = self.tokenizer.unk_id()
        
    def encode(self,
               sentence:str,
               add_bos:bool=False,
               add_eos:bool=False) -> List[int]:
        
        encoded = self.tokenizer.EncodeAsIds(sentence)
        if add_bos:
            encoded = [self.bos_id] + encoded
            
        if add_eos:
            encoded = encoded + [self.eos_id]
            
        return encoded
    
    def decode(self,
               encoded:List[int]) -> str:
        
        decoded = self.tokenizer.DecodeIds(encoded)
        return decoded
        