from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
# https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html#attention_specific_functions
class AttentionEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, device, attention):
        super(AttentionDecoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        if attention == "dot":
            self.attention = DotAttention()
        elif attention == "bilinear":
            self.attention = BilinearAttention(hidden_size)
        elif attention == "multi_layer_perceptron":
            self.attention = MultiLayerPerceptron(hidden_size)
        elif attention == 'QK':
            self.attention = QKAttention()
        else:
            raise ValueError("Invalid attention type")

    def forward(self, input, hidden, encoder_hiddens):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        # encoder_hiddens = encoder_hiddens[:-1]
        hidden = self.attention(encoder_hiddens, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

class DotAttention(nn.Module):
    def __init__(self):
        super(DotAttention, self).__init__()
        self.hidden_size = torch.tensor(512).to('cuda')

    def forward(self, encoder_hiddens, decoder_hidden):

        attn_score = encoder_hiddens.squeeze(1) @ decoder_hidden.squeeze(1).t()
        # print('encoder_hidden_size:', encoder_hiddens.squeeze(1).size())
        # print('decoder_hidden_size:', decoder_hidden.squeeze(1).size())
        attn_score = attn_score / torch.sqrt(self.hidden_size)

        attn_weights = F.softmax(attn_score, dim=0)
        # print("attn_weights:", attn_weights)
        attn_applied = attn_weights.t() @ encoder_hiddens.squeeze(1)
        attn_applied = attn_applied.unsqueeze(1)
        
        return attn_applied


class QKAttention(nn.Module):
    def __init__(self):
        super(QKAttention, self).__init__()
        self.hidden_size = torch.tensor(512).to('cuda')
        self.W_q = nn.Linear(self.hidden_size, self.hidden_size)  # 查询的权重矩阵
        self.W_k = nn.Linear(self.hidden_size, self.hidden_size)  # 键的权重矩阵

    def forward(self, encoder_hiddens, decoder_hidden):

        query = self.W_q(decoder_hidden)
        keys = self.W_k(encoder_hiddens)

        attn_score = keys.squeeze(1) @ query.squeeze(1).t()
        # print('encoder_hidden_size:', encoder_hiddens.squeeze(1).size())
        # print('decoder_hidden_size:', decoder_hidden.squeeze(1).size())
        attn_score = attn_score / torch.sqrt(self.hidden_size)

        attn_weights = F.softmax(attn_score, dim=0)
        # print("attn_weights:", attn_weights)
        attn_applied = attn_weights.t() @ encoder_hiddens.squeeze(1)
        attn_applied = attn_applied.unsqueeze(1)
        
        return attn_applied


class BilinearAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BilinearAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, encoder_hiddens, decoder_hidden):
        attn_score = encoder_hiddens.squeeze(1) @ self.attn(decoder_hidden).squeeze(1).t()
        attn_weights = F.softmax(attn_score, dim=0)
        attn_applied = attn_weights.t() @ encoder_hiddens.squeeze(1)
        attn_applied = attn_applied.unsqueeze(1)
        return attn_applied

class MultiLayerPerceptron(nn.Module):
    def __init__(self, hidden_size):
        super(MultiLayerPerceptron, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, encoder_hiddens, decoder_hidden):
        decoder_hiddens = decoder_hidden.repeat(encoder_hiddens.size(0), 1, 1)
        cat_hiddens = torch.cat((encoder_hiddens, decoder_hiddens), dim=2)
        cat_hiddens = cat_hiddens.squeeze(1)
        attn_score = self.fc2(self.tanh(self.fc1(cat_hiddens)))
        attn_weights = F.softmax(attn_score, dim=0)
        attn_applied = attn_weights.t() @ encoder_hiddens.squeeze(1)
        attn_applied = attn_applied.unsqueeze(1)
        return attn_applied