import torch.nn as nn
import torch


class RnnModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RnnModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, x, h):
        x, _ = self.rnn(x, h)
        x = self.output(x.contiguous().view(-1, self.hidden_size))

        return x

    def init_hidden(self):
        return torch.zeros(1, 8, self.hidden_size)
