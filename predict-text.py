import click
from model import RnnModel
import torch
import torch.nn.functional as F


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

    initial = [word2int[c] for c in "the"]
    h = (torch.zeros(2, 1, hidden_size), torch.zeros(2, 1, hidden_size))

    for c in initial:
        print(int2word[c], end='')
        x = torch.zeros(1, 1, input_size)
        x[0, 0, c] = 1
        output, h = model.forward(x, h)

    for i in range(n):
        k = torch.topk(F.softmax(output, dim=1), k=3)
        i = torch.multinomial(k[0], 1).item()
        c = k[1][0, i].item()
        print(int2word[c], end='')
        x = torch.zeros(1, 1, input_size)
        x[0, 0, c] = 1
        output, h = model.forward(x, h)


if __name__ == '__main__':
    main()
