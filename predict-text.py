import click
from model import RnnModel
import torch
import torch.nn.functional as F


@click.command()
@click.option('--n', default=100)
@click.argument('read-model')
def main(n, read_model):
    model = RnnModel(57, 30)
    model.load_state_dict(torch.load(read_model))
    initial = "Lorem ipsum"
    h = torch.zeros(1, 1, 30)
    for i in range(n):
        for c in initial:
            x = torch.randn(1, 1, 57)
            output, h = model.forward(x, h)
            print(torch.argmax(F.softmax(output)).item())


if __name__ == '__main__':
    main()
