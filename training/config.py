class Config:
    def __init__(self):
        self.vocab_size = 32000
        self.embed_dim = 384
        self.num_transformer_layers = 32
        self.num_heads = 12
        self.attn_dropout = 0.05
        self.ffn_dropout = 0.1
        self.dataset_name = "draco976/wikipedia-bookcorpus"
        self.dataset_split = 'train'
        self.tkn_model = 'models/Marcella_vocab_32K.model'
        self.block_size = 512
        self.batch_size = 4
        self.num_workers = 4
        self.pin_memory = True
        self.prefetch_factor = 2
        self.persistent_workers = True
        self.max_samples = None
        self.validation_prompt = 'It was a terrible gruesome accident that killed Professor Jacob Grimes, and from a certain point of view it was her fault, and so for reasons of both moral obligation and self-interest—for without Professor Grimes she had no committee chair, and without a committee chair she could not defend her dissertation, graduate, or apply successfully for a tenure-track job in analytic magick—Alice found it necessary to beg for his life back from King Yama the Merciful, Ruler of the Underworld. This was no small undertaking. Over the past month she had become a self-taught expert in Tartarology, which was not one of her subfields. These days it was not anyone’s subfield, as Tartarologists rarely survived to publish their work. Since Professor Grimes’s demise she had spent her every waking moment reading every monograph, paper, and shred of correspondence she could find on the journey to Hell and back. At least a dozen scholars had made the trip and lived to credibly tell the tale, but very few in the past century. All existing sources were unreliable to different degrees and devilishly tricky to translate besides. Dante’s account was so distracted with spiteful potshots that the reportage got lost within. T. S. Eliot had supplied some of the more recent and detailed landscape descriptions on record, but The Waste Land was so self-referential that its status as a sojourner’s account was under serious dispute. Orpheus’s notes, already in archaic Greek, were largely in shreds like the rest of him. And Aeneas—well, that was all Roman propaganda. Possibly there were more accounts in lesser-known languages—Alice could have spent decades poring through the archives —but her funding clock could not wait. Her progress review loomed at the end of the term, and without a living and breathing advisor, the best Alice could hope for was an extension of funding sufficient to last until she transferred elsewhere and found a new advisor. But she didn’t want to transfer elsewhere, she wanted a Cambridge degree. She didn’t want any other advisor.'