import click
import re


@click.command()
@click.option('--epochs')
@click.argument('text-file')
@click.argument('write-model')
def main(epochs, text_file, write_model):
    corpus = read_text(text_file)
    dictionary = set(corpus)
    print(list(dictionary)[:1000])

    print("Text file #words: {}".format(len(corpus)))
    print("Dictionary size: {}".format(len(dictionary)))


def read_text(file):
    with open(file) as f:
        s = f.read().lower()
        s = s.replace('.', ' <fullstop> ')
        s = s.replace(',', ' <comma> ')
        s = s.replace('?', ' <question> ')
        s = s.replace('!', ' <exclamation-mark>')
        s = s.replace(';', ' <semicolon> ')
        s = s.replace(':', ' <colon> ')
        s = s.replace('„', ' <quote-start> ')
        s = re.sub(r'\[.+\]', '', s)

        return s.split()


if __name__ == '__main__':
    main()
