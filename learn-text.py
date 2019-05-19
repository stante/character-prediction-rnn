import torch
import torch.nn as nn
import numpy as np
import click
import re
from model import RnnModel
from tqdm import tqdm


@click.command()
@click.option('--epochs', default=100)
@click.option('--batch-size', default=128)
@click.option('--seq-length', default=100)
@click.option('--hidden-size', default=512)
@click.option('--num-layers', default=2)
@click.argument('text-file')
@click.argument('write-model')
def main(epochs, batch_size, seq_length, hidden_size, num_layers, text_file, write_model):
    text = read_text(text_file)
    vocabulary = set(text)

    print("Text file #words: {}".format(len(text)))
    print("Dictionary size: {}".format(len(vocabulary)))

    if torch.cuda.is_available:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    int2word, word2int = create_lookup(vocabulary)
    encoded_text = np.array([word2int[t] for t in text])

    model = RnnModel(len(vocabulary), hidden_size, num_layers=num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    i = 0
    for x, y in generate_batches(encoded_text, batch_size, seq_length):
        i += 1

    print("Iterations: {} Total: {}".format(i, i * batch_size * seq_length))

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        h = tuple([state.to(device) for state in model.init_hidden(batch_size)])
        for x, y in generate_batches(encoded_text, batch_size, seq_length):
            optimizer.zero_grad()
            x = one_hot_encoder(x, len(vocabulary))
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

            h = tuple([state.clone().detach() for state in h])

            output, h = model.forward(x, h)

            loss = criterion(output, y.view(batch_size * seq_length))
            loss.backward()
            optimizer.step()

        pbar.set_description("Loss: {:0.6f}".format(loss.item()))

    state = {
        'state_dict': model.state_dict(),
        'word2int': word2int,
        'int2word': int2word,
        'hidden_size': hidden_size,
        'num_layers': num_layers
    }

    torch.save(state, write_model)


def generate_batches(text, batch_size, seq_length):
    original = text
    prediction = np.roll(text, -1)
    nbatches = len(text) // (batch_size * seq_length)

    for i in range(0, seq_length*nbatches*batch_size, seq_length * batch_size):
        x = original[i:i + seq_length * batch_size]
        y = prediction[i:i + seq_length * batch_size]

        yield x.reshape(batch_size, seq_length), y.reshape(batch_size, seq_length)


def one_hot_encoder(batch, length):
    one_hot = np.zeros((np.multiply(*batch.shape), length), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), batch.flatten()] = 1
    one_hot = one_hot.reshape((*batch.shape, length))

    return one_hot


def read_text(file):
    with open(file) as f:
        s = f.read().lower()
        s = re.sub(r'\[.+\]', '', s)

        return np.array(list(s))


def create_lookup(vocabulary):
    int2word = dict(enumerate(vocabulary))
    word2int = {v: k for k, v in int2word.items()}

    return int2word, word2int


if __name__ == '__main__':
    main()
