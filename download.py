from datasets import load_dataset

dataset = load_dataset(
    "shahrukhx01/wikipedia-bookscorpus-en-preprocessed",
    split="train",
    streaming=True
)