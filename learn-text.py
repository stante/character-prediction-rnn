import torch
import torch.nn as nn
import numpy as np
import click
import re
from model import RnnModel


@click.command()
@click.option('--epochs')
@click.argument('text-file')
@click.argument('write-model')
def main(epochs, text_file, write_model):
    text = read_text(text_file)
    vocabulary = set(text)

    print("Text file #words: {}".format(len(text)))
    print("Dictionary size: {}".format(len(vocabulary)))

    int2word, word2int = create_lookup(vocabulary)
    encoded_text = np.array([word2int[t] for t in text])

    model = RnnModel(len(vocabulary), 30)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    for x, y in generate_batches(encoded_text, 8, 20):
        optimizer.zero_grad()
        x = one_hot_encoder(x, len(vocabulary))
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        output = model.forward(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        break


def generate_batches(text, batch_size, seq_length):
    original = text
    prediction = np.roll(text, 1)
    nbatches = len(text) // (batch_size * seq_length)

    for i in range(0, nbatches, seq_length*batch_size):
        x = original[i:seq_length * batch_size]
        y = prediction[i:seq_length * batch_size]

        yield x.reshape(batch_size, seq_length), y.reshape(batch_size, seq_length)


def one_hot_encoder(batch, length):
    one_hot = np.zeros((np.multiply(*batch.shape), length), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), batch.flatten()] = 1
    one_hot = one_hot.reshape((*batch.shape, length))

    return one_hot


def read_text(file):
    with open(file) as f:
        s = f.read().lower()
        s = s.replace('.', ' <stop> ')
        s = s.replace(',', ' <comma> ')
        s = s.replace('?', ' <question-mark> ')
        s = s.replace('!', ' <exclamation-mark>')
        s = s.replace(';', ' <semicolon> ')
        s = s.replace(':', ' <colon> ')
        s = s.replace('„', ' <quote-start> ')
        s = re.sub(r'\[.+\]', '', s)

        return np.array(list(s))


def create_lookup(vocabulary):
    int2word = dict(enumerate(vocabulary))
    word2int = {v: k for k, v in int2word.items()}

    return int2word, word2int


if __name__ == '__main__':
    main()
