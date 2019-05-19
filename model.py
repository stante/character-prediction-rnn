import torch.nn as nn
import torch


class RnnModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RnnModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, x, h):
        x, hidden = self.rnn(x, h)
        x = self.dropout(x)
        x = self.output(x.contiguous().view(-1, self.hidden_size))

        return x, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))
