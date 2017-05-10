#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Module sphinx_fontawesome
"""

import os
from sphinx import addnodes
from docutils.nodes import strong, emphasis, reference, Text
from docutils.parsers.rst.roles import set_classes
from docutils.parsers.rst import Directive
import docutils.parsers.rst.directives as directives

import sphinx_fontawesome.constant

__version_info__ = (0, 0, 2)
__version__ = '.'.join([str(val) for val in  __version_info__])

sub_special = [
                {'id' : 'o', 'key' : 'square-o'},
                {'id' : 'x', 'key' : 'check-square-o'},
                {'id' : 'smile', 'key' : 'smile-o'},
                {'id' : 'mail', 'key' : 'envelope'},
                {'id' : 'note', 'key' : 'info-circle'},
              ]
# add role
def fa_global(key=''):
    def fa(role, rawtext, text, lineno, inliner, options={}, content=[]):
        options.update({'classes': []})
        options['classes'].append('fa')
        if key:
            options['classes'].append('fa-%s' % key)
        else:
             for x in text.split(","):
                options['classes'].append('fa-%s' % x)
        set_classes(options)
        node = emphasis(**options)
        return [node], []
    return fa

#add directive
class Fa(Directive):

    has_content = True

    def run(self):
        options= {'classes' : []}
        options['classes'].append('fa')
        for x in self.content[0].split(' '):
            options['classes'].append('fa-%s' % x)
        set_classes(options)
        node = emphasis(**options)
        return [node]
 
prolog = '\n'.join(['.. |%s| fa:: %s' % (icon, icon) for icon in sphinx_fontawesome.constant.icons])
prolog += '\n'
prolog += '\n'.join(['.. |%s| fa:: %s' % (icon['id'], icon['key']) for icon in sub_special])



def setup(app):
    app.add_role('fa', fa_global())
    app.add_directive('fa', Fa)
    app.config.rst_prolog = prolog
    return {'version': __version__}

