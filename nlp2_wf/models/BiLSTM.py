import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderBiLSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_directions = 2
        self.embedding = nn.Embedding(input_size, hidden_size)
        # 使用双向 LSTM，注意 hidden_size 保持不变
        self.bilstm = nn.LSTM(hidden_size, hidden_size, bidirectional=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        print("hidden_size:", hidden.size())
        output, (h_n, c_n) = self.bilstm(embedded, hidden)
        # 将双向输出的两部分取平均，以保持维度一致性
        
        # print("output_size:", output.size())

        output = (output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]) / 2
        print("output_2_size:", output.size())
        # 将双向的输出平均合并
        h_n = torch.mean(h_n, dim=0, keepdim=True)
        h_n_2 = torch.cat(h_n, h_n, dim = 0)

        c_n = torch.mean(c_n, dim=0, keepdim=True)
        c_n_2 = torch.cat(c_n, c_n, dim = 0)

        print("h_n:", h_n_2.size())
        print("c_n:", c_n_2.size())

        hidden = (h_n, c_n)

        print("h_n2:", h_n.size())

        return output, hidden

    def initHidden(self):
        # 为每个方向初始化隐藏状态和单元状态，每个方向的 hidden_size 为原始的一半
        return (torch.zeros(2, 1, self.hidden_size, device=self.device),
                torch.zeros(2, 1, self.hidden_size, device=self.device))


class BiLSTMDecoder(nn.Module):
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
