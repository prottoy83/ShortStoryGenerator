# GENERATED TEXT TOKENIZER USING AI 

import re
from collections import Counter

class StoryTokenizer:
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        
        # Define special tokens
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        
        # Dictionaries to map strings to ints, and ints to strings
        self.i2w = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}
        self.w2i = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        
    def _clean_text(self, text):
        # Basic cleanup: lowercase and separate punctuation
        text = text.lower()
        text = re.sub(r"([?.!,])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def train(self, dataset_stream, num_samples=50000):
        print(f"Training tokenizer on {num_samples} stories...")
        word_counts = Counter()
        
        # Read a chunk of the dataset to find the most common words
        count = 0
        for item in dataset_stream:
            if count >= num_samples:
                break
            
            text = self._clean_text(item["text"])
            word_counts.update(text.split())
            count += 1
            
        # Keep only the most common words, up to our max vocab size
        common_words = word_counts.most_common(self.max_vocab_size - 4) # -4 for special tokens
        
        for idx, (word, _) in enumerate(common_words, start=4):
            self.w2i[word] = idx
            self.i2w[idx] = word
            
        print(f"Tokenizer trained. Final vocab size: {len(self.w2i)}")

    def encode(self, text):
        # Convert text to a list of integer IDs
        text = self._clean_text(text)
        return [self.w2i.get(word, self.unk_id) for word in text.split()]

    def decode(self, token_ids):
        # Convert a list of integer IDs back to text
        words = [self.i2w.get(token, "<UNK>") for token in token_ids]
        # Filter out special tokens for clean reading
        words = [w for w in words if w not in ["<PAD>", "<BOS>", "<EOS>"]]
        return " ".join(words)
        
    def __len__(self):
        return len(self.w2i)