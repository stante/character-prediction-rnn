import numpy as np
import click
import re


@click.command()
@click.option('--epochs')
@click.argument('text-file')
@click.argument('write-model')
def main(epochs, text_file, write_model):
    corpus = read_text(text_file)
    vocabulary = set(corpus)

    print("Text file #words: {}".format(len(corpus)))
    print("Dictionary size: {}".format(len(vocabulary)))

    int2word, word2int = create_lookup(vocabulary)


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

        return np.array(s.split())


def create_lookup(vocabulary):
    int2word = dict(enumerate(vocabulary))
    word2int = {v: k for k, v in int2word.items()}

    return int2word, word2int


if __name__ == '__main__':
    main()
