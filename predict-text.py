import click
from model import RnnModel
import torch
import torch.nn.functional as F
import numpy as np


@click.command()
@click.option('--n', default=1000)
@click.argument('read-model')
def main(n, read_model):
    state = torch.load(read_model)
    word2int = state['word2int']
    int2word = state['int2word']
    input_size = len(int2word)
    hidden_size = state['hidden_size']
    num_layers = state['num_layers']
    model = RnnModel(input_size, hidden_size, num_layers)
    model.load_state_dict(state['state_dict'])

    initial = [word2int[c] for c in "anna"]
    h = model.init_hidden(1)
    model.eval()
    for c in initial:
        print(int2word[c], end='')
        char, h = predict(model, c, h, 3)

    for i in range(n):
        char, h = predict(model, char, h, 3)
        print(int2word[char], end='')


def predict(net, char, h=None, top_k=1):
    input_size = net.input_size
    input = torch.zeros(1, 1, input_size)
    input[0, 0, char] = 1
    prediction, h = net.forward(input, h)
    prediction = F.softmax(prediction, dim=1).detach()
    p, top_char = prediction.topk(top_k)
    p, top_char = p.numpy().squeeze(), top_char.numpy().squeeze()
    char = np.random.choice(top_char, p=p/p.sum())
    return char, h


if __name__ == '__main__':
    main()
