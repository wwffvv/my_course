import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderTransformer, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8), num_layers=1)

    def forward(self, input):
        sentence_embedding = torch.zeros(input.size(0), 1, self.hidden_size, device=self.device)
        for i, word in enumerate(input):
            embedded = self.embedding(word).view(1, 1, -1)
            sentence_embedding[i] = embedded
        output = sentence_embedding
        output = self.transformer_encoder(output)
        output = torch.mean(output, dim=0)
        return output

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(Decoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):

        # Your code here #
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
