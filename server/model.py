import torch
import torch.nn as nn

class StoryLSTM(nn.Module):
    def __init__(self, embded_size, hidden_size, vocabs):
        super().__init__()

        self.embedding = nn.Embedding(vocabs, embded_size)

        self.lstm = nn.LSTM(
                embded_size,
                hidden_size,
                num_layers=2,
                dropout=0.2,
                batch_first=True
            )
        self.fc = nn.Linear(hidden_size, vocabs)
    
    def forward(self, x):

        X = self.embedding(x)
        output, (hidden, cell) = self.lstm(X)
        logits = self.fc(output)

        return logits

