import torch.nn as nn

class ChordGenLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, lstm_units=512):
        super(ChordGenLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_units, num_layers=2, batch_first=True)
        self.fc = nn.Linear(lstm_units, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x