import click
from model import RnnModel
import torch
import torch.nn.functional as F


@click.command()
@click.option('--n', default=100)
@click.argument('read-model')
def main(n, read_model):
    model = RnnModel(57, 30)
    state = torch.load(read_model)
    model.load_state_dict(state['state_dict'])
    word2int = state['word2int']
    int2word = state['int2word']
    initial = [word2int[c] for c in "jemand musste"]
    h = torch.zeros(1, 1, 30)
    for i in range(n):
        for c in initial:
            x = torch.randn(1, 1, 57)
            output, h = model.forward(x, h)
            print(int2word[torch.argmax(F.softmax(output, dim=1)).item()], end='')


if __name__ == '__main__':
    main()
