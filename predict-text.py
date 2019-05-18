import click
from model import RnnModel
import torch
import torch.nn.functional as F


@click.command()
@click.option('--n', default=100)
@click.argument('read-model')
def main(n, read_model):
    model = RnnModel(57, 512)
    state = torch.load(read_model)
    model.load_state_dict(state['state_dict'])
    word2int = state['word2int']
    int2word = state['int2word']
    initial = [word2int[c] for c in "jemand musste"]
    h = torch.zeros(1, 1, 512)

    for c in initial:
        print(int2word[c], end='')
        x = torch.zeros(1, 1, 57)
        x[0, 0, c] = 1
        output, h = model.forward(x, h)

    for i in range(n):
        k = torch.topk(F.softmax(output, dim=1), k=3)
        i = torch.multinomial(k[0], 1).item()
        c = k[1][0, i].item()
        print(int2word[c], end='')
        x = torch.zeros(1, 1, 57)
        x[0, 0, c] = 1
        output, h = model.forward(x, h)


if __name__ == '__main__':
    main()
