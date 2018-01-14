try:
    import click
    have_click = True
except ImportError:
    have_click = False

if not have_click:
    raise ImportError("The Veros command line tools require click (e.g. through `pip install click`)")
