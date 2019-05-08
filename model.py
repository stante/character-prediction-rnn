import torch.nn as nn


class RnnModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RnnModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = self.rnn(x)
        x = self.output(x.view(-1, self.input_size))

        return x
