import click


@click.command()
@click.option('epochs')
@click.argument('text-file')
@click.argument('write-model')
def main(epochs, text_file, write_model):
    pass


if __name__ == '__main__':
    main()
