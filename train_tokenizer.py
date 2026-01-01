import os
import subprocess
import threading
from datasets import load_dataset
from tqdm import tqdm

PIPE_PATH = "/tmp/spm_input.pipe"
MODEL_PREFIX = "Marcella_vocab_32K"
VOCAB_SIZE = 32000
INPUT_SENTENCE_SIZE = 1_000_000
MAX_LINES_TO_WRITE = 1_200_000

DATASET_NAME = "shahrukhx01/wikipedia-bookscorpus-en-preprocessed"
TEXT_FIELD = "text"

if not os.path.exists(PIPE_PATH):
    os.mkfifo(PIPE_PATH)

dataset = load_dataset(DATASET_NAME, split="train")

def writer():
    with open(PIPE_PATH, "w", encoding="utf-8") as f:
        for i, ex in enumerate(tqdm(dataset, desc="Streaming text")):
            if i >= MAX_LINES_TO_WRITE:
                break
            text = ex.get(TEXT_FIELD)
            if text:
                f.write(text.replace("\n", " ") + "\n")

t = threading.Thread(target=writer, daemon=True)
t.start()

cmd = [
    "spm_train",
    f"--input={PIPE_PATH}",
    "--input_format=text",
    f"--model_prefix={MODEL_PREFIX}",
    "--model_type=unigram",
    f"--vocab_size={VOCAB_SIZE}",
    "--character_coverage=0.9995",
    f"--input_sentence_size={INPUT_SENTENCE_SIZE}",
    "--shuffle_input_sentence=true",
    "--seed_sentencepiece_size=50000",
    "--num_threads=2",
    "--unk_id=0",
    "--bos_id=1",
    "--eos_id=2",
    "--pad_id=3",
]

subprocess.run(cmd, check=True)
t.join()