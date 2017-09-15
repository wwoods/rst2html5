#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

# Author: Florian Brucker <mail@florianbrucker.de>
# Copyright: This module has been placed in the public domain.

"""
Math handling for ``html5css3``.
"""

from __future__ import unicode_literals

import asyncio
import codecs
import hashlib
import os.path
import re
import shutil
import struct
import subprocess
import tempfile

from docutils.utils.math.unichar2tex import uni2tex_table
from docutils.utils.math import math2html, pick_math_environment
from docutils.utils.math.latex2mathml import parse_latex_math

from .html import *


__all__ = ['HTMLMathHandler', 'LateXMathHandler', 'MathHandler',
           'MathJaxMathHandler', 'MathMLMathHandler', 'SimpleMathHandler']


class MathHandler(object):
    """
    Abstract math handler.
    """
    CLASS = None
    BLOCK_WRAPPER = '%(code)s'
    INLINE_WRAPPER = '%(code)s'

    def __init__(self):
        self._setup_done = False

    def convert(self, translator, node, block):
        if not self._setup_done:
            self._setup(translator)
            self._setup_done = True
        code = node.astext()
        if block:
            env = pick_math_environment(code)
            wrapper = self.BLOCK_WRAPPER
        else:
            env = ''
            wrapper = self.INLINE_WRAPPER
        code = code.translate(uni2tex_table)
        code = wrapper % {'code': code, 'env': env}
        tag = self._create_tag(code, block)
        if self.CLASS:
            tag.attrib['class'] = self.CLASS
        return tag

    def _create_tag(self, code, block):
        raise NotImplementedError('Must be implemented in subclass.')

    def _setup(self, translator):
        pass


class SimpleMathHandler(MathHandler):
    """
    Base class for simple math handlers.
    """
    BLOCK_TAG = None
    INLINE_TAG = None

    def _create_tag(self, code, block):
        if block:
            return self.BLOCK_TAG(code)
        else:
            return self.INLINE_TAG(code)


class LaTeXMathHandler(SimpleMathHandler):
    """
    Math handler for raw LaTeX output.
    """
    BLOCK_TAG = Pre
    INLINE_TAG = Tt
    CLASS = 'math'
    BLOCK_WRAPPER = '%(code)s'
    INLINE_WRAPPER = '%(code)s'


class ImgMathHandler(MathHandler):
    """Mostly taken from sphinx.ext.math; renders LaTeX math expressions to
    an image and embeds it.
    """
    IMAGE_FORMAT = 'png'
    FONT_SIZE = 12

    DOC_HEAD = r'''
\documentclass[preview,12pt]{standalone}
\usepackage[utf8x]{inputenc}
\usepackage{amsmath}
\usepackage{amsthm}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage{anyfontsize}
\usepackage{bm}
\usepackage{geometry}
\usepackage{mathtools}
'''

    PREAMBLE = r''

    DOC_BODY = r'''
\begin{document}
\fontsize{%d}{%d}\selectfont %s%%
\end{document}
'''

    DOC_BODY_PREVIEW = r'''
\usepackage[active]{preview}
\begin{document}
\begin{preview}
\fontsize{%d}{%d}\selectfont %s%%
\end{preview}
\end{document}
'''
    DOC_BODY_PREVIEW = DOC_BODY

    def __init__(self, font_size=FONT_SIZE):
        super().__init__()

        self.FONT_SIZE = font_size

        self._async_tasks = []


    def finalize(self):
        self._async_tasks and asyncio.get_event_loop().run_until_complete(
                asyncio.wait(self._async_tasks))
        self._async_tasks = []


    def _create_tag(self, code, block):

        cls = []
        if block:
            cls.append('imgmath-block')
        else:
            cls.append('imgmath-inline')
        mathtag = tag = Img(class_=' '.join(cls))
        container = None

        if not block:
            # Don't distort text lines
            container = Span(class_='imgmath-inline-container')
            # Images are weird
            container2 = Span()
            container2.append(tag)
            container.append(container2)
            tag = container

        async def render_inner():
            size, imgdata = await self._render_math(code, block)
            mathtag.attrib['src'] = imgdata

            dim_style = (
                    f"width:{size[0] * .3528}mm;"
                    f"height:{size[1] * .3528}mm;"
                    f"max-width:none;max-height:none;")
            mathtag.attrib['style'] = dim_style
            if container is not None:
                container.attrib['style'] = "width:{}mm".format(size[0] * 0.3528)
                container2.attrib['style'] = dim_style
            if size[2] is not None:
                # Add depth information, align with text
                mathtag.attrib['style'] += (";bottom:{}mm".format(
                        -size[2] * 0.3528)) # (size[1] - size[2]) * 0.3528 - 3.8))
        self._async_tasks.append(render_inner())

        return tag

    async def _render_math(self, code, block, use_preview=True):
        """Returns ((width in pt, height in pt, depth in pt), embedded image
                data for src tag).
        """
        import base64
        import binascii

        fmt = self.IMAGE_FORMAT
        if fmt not in ('png', 'svg'):
            raise ValueError('Must be "png" or "svg": {}'.format(fmt))

        font_size = self.FONT_SIZE
        if fmt == 'png':
            # Scale up, as png can be low DPI otherwise
            png_upscale = 4
            font_size *= png_upscale

        body = self.DOC_BODY if not use_preview else self.DOC_BODY_PREVIEW

        if not block:
            body = rf'\geometry{{paperwidth=100in}}{body}'
            math = '${}$'.format(code)
        else:
            # Very loosely from sphinx.ext.mathbase.wrap_displaymath
            math = (r'\begin{{equation*}}\begin{{split}}'
                    r'{}\end{{split}}\end{{equation*}}'.format(code))

        latex = [self.DOC_HEAD, self.PREAMBLE]
        latex.append(body % (font_size, int(round(font_size * 1.2)), math))
        latex = ''.join(latex)
        shasum = "{}.{}".format(hashlib.sha1(latex.encode('utf-8')).hexdigest(),
                fmt)

        async def run_cmd(cmd, working_dir):
            """Returns stdout"""
            p = await asyncio.create_subprocess_exec(*cmd,
                    cwd=working_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await p.communicate()

            stdout = stdout.decode('utf-8')
            stderr = stderr.decode('utf-8')

            if p.returncode != 0:
                raise ValueError(f"Latex exited with error:\n\nSTDOUT:\n{stdout}\n\n"
                        f"STDERR:\n{stderr}")
            return stdout

        # Do building in a temp directory
        tmp = tempfile.mkdtemp(prefix='{}-'.format(shasum))
        try:
            with open(os.path.join(tmp, 'math.tex'), 'w') as f:
                f.write(latex)
            latex_cmd = ['latex', '--interaction=nonstopmode', 'math.tex']
            await run_cmd(latex_cmd, tmp)

            if fmt == 'png':
                cmd = ['dvipng', '--width', '--height', '-T', 'tight', 'z9',
                        '-bg', 'Transparent',
                        '-o', 'math.png', 'math.dvi']
                if use_preview:
                    cmd.append('--depth')
                output = 'math.png'
                content_header = 'data:image/png;base64,'
                def get_dims(stdout):
                    """Should return (width in pt, height in pt, depth in pt)."""
                    dims = stdout.split('\n[')
                    if len(dims) < 2:
                        raise ValueError("Bad output, no width and height? {}".format(stdout))

                    w = re.search(r'width=-?(\d+)', dims[1])
                    if w is None:
                        raise ValueError("No width? {}".format(dims[1]))
                    w = int(w.group(1))

                    h = re.search(r'height=-?(\d+)', dims[1])
                    if h is None:
                        raise ValueError("No height? {}".format(dims[1]))
                    h = int(h.group(1))

                    if use_preview:
                        d = re.search(r'depth=(-?\d+)', dims[1])
                        if d is None:
                            raise ValueError("No depth? {}".format(dims[1]))
                        d = int(d.group(1))

                    # Must add depth to height to get real height.
                    h = h + d

                    # Already in pt, default at 72 dpi
                    # Was double-size rendered
                    return (
                            w * 1. / png_upscale,
                            h * 1. / png_upscale,
                            d * 1. / png_upscale,
                    )

                cmd_out = await run_cmd(cmd, tmp)
                dims = get_dims(cmd_out)
            elif fmt == 'svg':
                raise NotImplementedError("Looks bad, broken use_preview...")
                cmd = ['dvisvgm', '-o', 'math.svg', 'math.dvi']
                output = 'math.svg'
                content_header = 'data:image/svg+xml;base64,'

                cmd_out = await run_cmd(cmd, tmp)
                raise NotImplementedError("Cannot get height.")
            else:
                raise NotImplementedError(fmt)

            with open(os.path.join(tmp, output), 'rb') as f:
                content = f.read()

            return dims, '{}{}'.format(content_header,
                    base64.b64encode(content).decode('utf-8'))
        finally:
            shutil.rmtree(tmp)


class MathJaxMathHandler(SimpleMathHandler):
    """
    Math handler for MathJax output.
    """
    BLOCK_TAG = Div
    INLINE_TAG = Span
    CLASS = 'math'
    BLOCK_WRAPPER = '\\begin{%(env)s}\n%(code)s\n\\end{%(env)s}'
    INLINE_WRAPPER = '\(%(code)s\)'

    DEFAULT_URL = 'http://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js'
    DEFAULT_CONFIG = """
        MathJax.Hub.Config({
            extensions: ["tex2jax.js"],
            jax: ["input/TeX", "output/HTML-CSS"],
            tex2jax: {
                inlineMath: [["\\\\(","\\\\)"]],
                displayMath: [['$$','$$'], ["\\\\[","\\\\]"]],
                processEscapes: true
            },
            "HTML-CSS": { availableFonts: ["TeX"] }
        });"""

    def __init__(self, js_url=None, config_filename=None):
        super(MathJaxMathHandler, self).__init__()
        self.js_url = js_url or self.DEFAULT_URL
        if config_filename:
            with codecs.open(config_filename, 'r', encoding='utf8') as f:
                self.config = f.read()
        else:
            self.config = self.DEFAULT_CONFIG

    def _setup(self, translator):
        translator.head.append(Script(self.config,
                               type="text/x-mathjax-config"))
        translator.head.append(Script(src=self.js_url))


class MathMLMathHandler(MathHandler):
    """
    Math handler for MathML output.
    """
    BLOCK_WRAPPER = '%(code)s'
    INLINE_WRAPPER = '%(code)s'

    def _create_tag(self, code, block):
        tree = parse_latex_math(code, inline=(not block))
        html = ''.join(tree.xml())
        tag = html_to_tags(html)[0]

        def strip_ns(tag):
            del tag.attrib['xmlns']
            for child in tag:
                strip_ns(child)

        for child in tag:
            strip_ns(child)
        return tag


class HTMLMathHandler(MathHandler):
    """
    Math handler for HTML output.
    """
    CLASS = 'formula'
    BLOCK_WRAPPER = '\\begin{%(env)s}\n%(code)s\n\\end{%(env)s}'
    INLINE_WRAPPER = '$%(code)s$'
    DEFAULT_CSS = os.path.join(os.path.dirname(__file__), 'math.css')

    def __init__(self, css_filename=None):
        super(HTMLMathHandler, self).__init__()
        self.css_filename = css_filename or self.DEFAULT_CSS

    def _create_tag(self, code, block):
        math2html.DocumentParameters.displaymode = block
        html = math2html.math2html(code)
        tags = html_to_tags(html)
        if block:
            return Div(*tags)
        else:
            return Span(*tags)

    def _setup(self, translator):
        translator.css(os.path.relpath(self.css_filename))


