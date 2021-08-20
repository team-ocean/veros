# -*- coding: utf-8 -*-
#
# Veros documentation build configuration file, created by
# sphinx-quickstart on Tue Mar  7 23:56:46 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('_3rdparty'))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
import sphinx_fontawesome

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_fontawesome',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'Veros'
copyright = u'2017-2020, The Veros Team, NBI Copenhagen'
author = u'The Veros Team, NBI Copenhagen'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
from veros import __version__ as veros_version

# The short X.Y version.
version = veros_version
# The full version, including alpha/beta/rc tags.
release = veros_version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'veros_doc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'veros.tex', u'Veros Documentation',
     u'The Veros Team', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'Veros', u'Veros Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'Veros', u'Veros Documentation',
     author, 'Veros', 'The versatile ocean simulator in pure Python',
     'Miscellaneous'),
]

# -- Options for autodoc --------------------------------------------------
autodoc_member_order = "bysource"
autodoc_default_options = {"show-inheritance": None}
autodoc_mock_imports = ["loguru", "numpy", "h5netcdf", "scipy"]

# -- Options for intersphinx ----------------------------------------------
intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

# -- Custom directives ----------------------------------------------------

from os.path import basename

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from docutils import nodes, statemachine
from docutils.parsers.rst import Directive, directives


class ExecDirective(Directive):
    """Execute the specified python code and insert the output into the document"""
    has_content = True

    def run(self):
        old_stdout, sys.stdout = sys.stdout, StringIO()

        tab_width = self.options.get('tab-width', self.state.document.settings.tab_width)
        source = self.state_machine.input_lines.source(self.lineno - self.state_machine.input_offset - 1)

        try:
            exec('\n'.join(self.content))
            text = sys.stdout.getvalue()
            lines = statemachine.string2lines(text, tab_width, convert_whitespace=True)
            self.state_machine.insert_input(lines, source)
            return []
        except Exception:
            warn_text = "Unable to execute python code at %s:%d" % (basename(source), self.lineno)
            warning = self.state_machine.reporter.warning(warn_text)
            return [warning, nodes.error(None, nodes.paragraph(text=warn_text), nodes.paragraph(text=str(sys.exc_info()[1])))]
        finally:
            sys.stdout = old_stdout


class ClickDirective(Directive):
    """Execute the specified click command and insert the output into the document"""
    required_arguments = 1
    optional_arguments = 0
    option_spec = {
        'args': directives.unchanged
    }
    has_content = False

    def run(self):
        import importlib
        import shlex
        from click.testing import CliRunner

        arg = self.arguments[0]
        options = shlex.split(self.options.get('args', ''))

        try:
            modname, funcname = arg.split(':')
        except ValueError:
            raise self.error('run-click argument must be "module:function"')

        try:
            mod = importlib.import_module(modname)
            func = getattr(mod, funcname)

            runner = CliRunner()
            with runner.isolated_filesystem():
                result = runner.invoke(func, options)

            text = result.output

            if result.exit_code != 0:
                raise RuntimeError('Command exited with non-zero exit code; output: "%s"' % text)

            node = nodes.literal_block(text=text)
            node['language'] = 'text'
            return [node]

        except Exception:
            warn_text = "Error while running command %s %s" % (arg, ' '.join(map(shlex.quote, options)))
            warning = self.state_machine.reporter.warning(warn_text)
            return [warning, nodes.error(None, nodes.paragraph(text=warn_text), nodes.paragraph(text=str(sys.exc_info()[1])))]


def setup(app):
    app.add_directive('exec', ExecDirective)
    app.add_directive('run-click', ClickDirective)
