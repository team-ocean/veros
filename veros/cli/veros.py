import click


@click.group('veros')
@click.version_option()
def cli():
    """Veros command-line tools"""
    pass


if __name__ == '__main__':
    cli()
