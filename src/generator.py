import torch
import torch.nn.functional as F

from model import StoryLSTM
from tokenizer import StoryTokenizer

def loadModel(checkpoint_path, device):

    checkpoint = torch.load(checkpoint_path, device)

    tokenizer = StoryTokenizer(max_vocab_size=checkpoint['vocab_size'])
    tokenizer.w2i = checkpoint['w2i']
    tokenizer.i2w = checkpoint['i2w']

    model = StoryLSTM(
        embded_size=checkpoint['embed_size'],
        hidden_size=checkpoint['hidden_size'],
        vocabs=checkpoint['vocab_size']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    model.eval()

    return model, tokenizer

def genStory(model, tokenizer, prompt, device, max_tok=30, temp=0.7):

    tokens = tokenizer.encode(prompt)

    tokens = [tokenizer.bos_id] + tokens

    print(f"\n[Prompt]: {prompt}")
    print("-" * 40)

    with torch.no_grad():
        for _ in range(max_tok):
            x=torch.tensor([tokens], dtype=torch.long).to(device=device)

            logits = model(x)
            next_token = logits[0, -1, :]

            nextTok = next_token / temp

            prob = F.softmax(nextTok, dim=-1)

            top_prob, top_ind = torch.topk(prob, k=100)
            choice = torch.multinomial(top_prob, num_samples=1)

            nextToken = top_ind[choice].item()
            if nextToken == tokenizer.eos_id:
                break

            tokens.append(nextToken)

        story = tokenizer.decode(tokens)
        return story

def main(promptx):
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint_path = "./checkpoints/tinystory_lstm_weights.pth"
    model, tokenizer = loadModel(checkpoint_path, device)

    #prompts = [
    #    "once upon a time",
    #    "the little boy wanted to",
    #    "one day, a magical cat"
    #]

    #for prompt in prompts:

    #    story = genStory(model, tokenizer, prompt, device, max_tok=150, temp=0.8)
    #    print(f"[Generated Story]:\n{story}\n")

    story = genStory(model, tokenizer, promptx, device, max_tok=1500, temp=0.8)
    print(f"[Generated Story]:\n{story}\n")

if __name__ == "__main__":
    prompt = input("Prompt: ")
    main(prompt)