from datasets import load_dataset
import torch
from torch.utils.data import IterableDataset
from dotenv import load_dotenv



class TinyStoryData(IterableDataset):

    def __init__(self, split, tokenizer, hftok, max_len=256, limit=None, buffer_size=10000):

        self.dataset = load_dataset("roneneldan/TinyStories",split=split, streaming= True, cache_dir="./data", token=hftok)
        if split == "train":
            self.dataset = self.dataset.shuffle(seed=42, buffer_size=buffer_size)
        if limit:
            self.dataset = self.dataset.take(limit)
    
        self.tokenizer = tokenizer
        self.max_len = max_len

    
    def __iter__(self):

        for item in self.dataset:
            text = item["text"]
            tokens = self.tokenizer.encode(text)

            tokens = ([self.tokenizer.bos_id] + tokens + [self.tokenizer.eos_id])

            tokens = tokens[: self.max_len + 1]
            padding = (self.max_len+1) - len(tokens)
            tokens += [self.tokenizer.pad_id] * padding

            X = torch.tensor(tokens[:-1], dtype=torch.long)
            y = torch.tensor(tokens[1:], dtype=torch.long)

            yield X,y
        
