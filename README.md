# Short Story Generator

An LSTM-based Sequential Short Story Generator built entirely from scratch in PyTorch. 

Taking a starting text prompt from the user, the model autoregressively predicts the most mathematically likely subsequent words to generate an original, coherent short story.

**Author:** Prottoy Roy

---

## 🧠 System Architecture

This project deliberately avoids pre-built Transformer libraries (like Hugging Face `transformers`) for the core logic, opting instead to build the entire natural language processing pipeline from text tokenization to model inference from the ground up.

The architecture is divided into four main pillars:

### 1. The Data Pipeline (Streaming & Tokenization)
To prevent Out-Of-Memory (OOM) crashes on standard hardware, the dataset pipeline is heavily optimized.
* **Dataset:** Uses the `TinyStories` dataset (~2.1 million syntactically simple stories).
* **Streaming `IterableDataset`:** Instead of loading the massive dataset into system RAM, the PyTorch `DataLoader` streams the data in chunks directly from the disk/network.
* **Custom Word Tokenizer:** A bespoke tokenizer that scans a subset of the data to build a frequency-based vocabulary dictionary (defaulting to the top 10,000 most common words). It automatically handles text cleaning, unknown word mapping (`<UNK>`), and sequence bounding (`<BOS>`, `<EOS>`, `<PAD>`).

### 2. The LSTM Model
The core neural network is a standard PyTorch implementation optimized for sequence modeling.
* **Embedding Layer:** Converts sparse integer token IDs into dense continuous vectors (adjustable embedding dimension).
* **LSTM Blocks:** Long Short-Term Memory layers process the sequence while maintaining a hidden state and cell state to carry context across time steps.
* **Linear Output:** A fully connected dense layer maps the final LSTM hidden states back to the vocabulary size, outputting raw logits for every possible word.

### 3. The Training Loop
* **Objective:** The model is trained using **Cross Entropy Loss**, predicting the next token at *every* time step simultaneously across the sequence.
* **Optimization:** Utilizes the Adam optimizer.
* **Stability:** Recurrent neural networks are notoriously prone to exploding gradients. This architecture implements strict **Gradient Clipping** (`max_norm=1.0`) during backpropagation to ensure mathematical stability.
* **Portability:** Checkpoints are saved as a master dictionary bundle containing not just the model weights (`state_dict`), but also the custom tokenizer mappings and dimension variables, ensuring the model can be loaded on any machine without the original dataset.

### 4. Autoregressive Generator
The inference script (`generator.py`) produces text one token at a time in a continuous feedback loop.
* **Temperature Scaling:** Before applying the Softmax function, logits are scaled by a `temperature` parameter. 
  * `T < 1.0`: Increases confidence, producing safer, more predictable text.
  * `T = 1.0`: Standard mathematical probability.
  * `T > 1.0`: Flattens the distribution, increasing randomness and creativity.
* **Multinomial Sampling:** Instead of simply picking the absolute highest probability word (`argmax`), the generator samples from the probability distribution, preventing the model from getting stuck in infinite repeating loops (e.g., "the the the").

---

## ⚙️ Hardware Support
This architecture is purely PyTorch-native and device-agnostic. It seamlessly supports:
* CPU inference
* NVIDIA CUDA environments
* AMD ROCm environments

---