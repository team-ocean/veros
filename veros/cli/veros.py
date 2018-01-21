import click


@click.group("veros")
def cli():
    """Veros command-line tools"""
    pass


if __name__ == "__main__":
    cli()
