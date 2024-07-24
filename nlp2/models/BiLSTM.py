import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderBiLSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.bilstm = nn.LSTM(hidden_size, hidden_size, bidirectional=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        # output, hidden = self.bilstm(embedded, hidden)
        output, (h_n, c_n) = self.bilstm(embedded, hidden)
        output = (output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]) / 2
        hidden = (h_n, c_n)
        # print("hidden size:", hidden.len())
        return output, hidden

    # (h_0, c_0)
    def initHidden(self):
        return (torch.zeros(2, 1, self.hidden_size, device=self.device), torch.zeros(2, 1, self.hidden_size, device=self.device))

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
