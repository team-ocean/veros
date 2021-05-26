import click


@click.group("veros")
@click.version_option()
def cli():
    """Veros command-line tools"""
    pass
