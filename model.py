import torch.nn as nn
import torch


class RnnModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RnnModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, x, h):
        x, hidden = self.rnn(x, h)
        x = self.output(x.contiguous().view(-1, self.hidden_size))

        return x, hidden

    def init_hidden(self):
        return (torch.zeros(2, 128, self.hidden_size), torch.zeros(2, 128, self.hidden_size))
