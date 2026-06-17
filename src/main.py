import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizer import StoryTokenizer
from dataset import TinyStoryData
from model import StoryLSTM
from train import train_model
from dotenv import load_dotenv
import os

def main():

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_lim = 500000

    tokenizer = StoryTokenizer(max_vocab_size=10000)
    raw_dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True, cache_dir="./data", token=hf_token)
    tokenizer.train(raw_dataset, num_samples=50000)

    print("Initializing dataset stream...")
    train_dataset = TinyStoryData(split="train", tokenizer=tokenizer, hftok=hf_token, max_len=256, limit=dataset_lim)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

    print("Initializing LSTM model...")
    vocab_size = len(tokenizer)
    
    model = StoryLSTM(
        embded_size=128, 
        hidden_size=256, 
        vocabs=vocab_size
    )

    epochs = 1
    print("Starting training...")
    checkpoint_path = "./checkpoints/tinystory_lstm_weights.pth"
    train_model(epochs, train_loader, model, tokenizer, device, checkpoint_path)

    

if __name__ == "__main__":
    main()