from __future__ import absolute_import
import os
import sys

try:
    from scss import Compiler as ScssCompiler
    has_scss = True
except ImportError:
    has_scss = False

import html5css3
import json
from . import html

IS_PY3 = sys.version[0] == '3'

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.parsers.rst import Directive

BASE_PATH = os.path.dirname(__file__)

join_path = os.path.join

if IS_PY3:
    def read_file(path):
        return open(path).read()
else:
    def read_file(path):
        return open(path).read().decode('utf-8')

def as_list(val):
    """return a list with val if val is not already a list, val otherwise"""
    if isinstance(val, list):
        return val
    else:
        return [val]

def abspath(path):
    return join_path(BASE_PATH, path)

def js_fullpath(path, embed=True):
    content = read_file(path) 

    if embed:
        return html.Script(content)
    else:
        return html.Script(src=path)

def js(path, embed=True):
    return js_fullpath(abspath(path), embed)

def css(path, embed=True):
    content = read_file(abspath(path))

    if path.endswith('.scss'):
        if not has_scss:
            raise ValueError("Please install pyscss for scss support")

        content = ScssCompiler().compile_string(content)

    if embed:
        return html.Style(content, type="text/css")
    else:
        return html.Link(href=path, rel="stylesheet", type="text/css")

def pretty_print_code(tree, embed=True, params=None):
    head = tree[0]
    body = tree[1]

    body.append(js(join_path("thirdparty", "prettify.js"), embed))
    body.append(html.Script("$(function () { prettyPrint() })"))

    langs_str = params.get("langs", "")
    langs = [x.strip() for x in langs_str.split(":") if x.strip() != ""]

    for lang in langs:
        lang_path = join_path("thirdparty", "prettify", "lang-" + lang + ".js")
        body.append(js(lang_path, embed))

    head.append(css(join_path("thirdparty", "prettify.css")))

def jquery(tree, embed=True, params=None):
    body = tree[1]
    body.append(js(join_path("thirdparty", "jquery.js"), embed))

def add_class(element, cls_name):
    cls = element.get("class", "")

    if cls:
        cls += " " + cls_name
    else:
        cls += cls_name

    element.set("class", cls)

def deckjs(tree, embed=True, params=None):
    head = tree[0]
    body = tree[1]

    def path(*args):
        return join_path("thirdparty", "deckjs", *args)

    add_class(body, "deck-container")

    for section in tree.findall(".//section"):
        add_class(section, "slide")

    # Core and extension CSS files
    head.append(css(path("core", "deck.core.css"), embed))
    head.append(css(path("extensions", "goto", "deck.goto.css"), embed))
    head.append(css(path("extensions", "menu", "deck.menu.css"), embed))
    head.append(css(path("extensions", "navigation", "deck.navigation.css"), embed))
    head.append(css(path("extensions", "status", "deck.status.css"), embed))

    # Theme CSS files (menu swaps these out)
    head.append(css(path("themes", "style", "web-2.0.css"), embed))
    head.append(css(path("themes", "transition", "horizontal-slide.css"), embed))

    body.append(js(path("modernizr.custom.js"), embed))
    jquery(tree, embed)

    # Deck Core and extensions
    body.append(js(path("core", "deck.core.js"), embed))
    body.append(js(path("extensions", "menu", "deck.menu.js"), embed))
    body.append(js(path("extensions", "goto", "deck.goto.js"), embed))
    body.append(js(path("extensions", "status", "deck.status.js"), embed))
    body.append(js(path("extensions", "navigation", "deck.navigation.js"), embed))

    body.append(html.Script("$(function () { $.deck('.slide'); });"))

def add_js(tree, embed=True, params=None):
    params = params or {}
    paths = as_list(params.get("path", []))

    body = tree[1]
    for path in paths:
        body.append(js_fullpath(path, embed))

def revealjs(tree, embed=True, params=None):
    head = tree[0]
    body = tree[1]
    params = params or {}
    theme_name = params.pop("theme", "league")
    theme_base_dir = params.pop("themepath", None)
    printpdf = params.pop("printpdf", False)

    def path(*args):
        return join_path("thirdparty", "revealjs", *args)

    if theme_base_dir:
        theme_path = join_path(os.path.abspath(os.path.expanduser(theme_base_dir)), theme_name)
    else:
        theme_path = path("css", "theme", theme_name)

    if os.path.lexists(theme_path + '.scss'):
        theme_path += '.scss'
    elif os.path.lexists(theme_path + '.css'):
        theme_path += '.css'
    else:
        raise ValueError("Could not find reveal.js theme at {}".format(
                theme_path))

    add_class(body, "reveal")
    slides = html.Div(class_="slides")

    for item in list(body):
        body.remove(item)
        slides.append(item)

    body.append(slides)

    # Any headings that are second-level shall become vertical slides
    for slide_body in tree.findall("./body/div[@class='slides']"):
        curgroup = None
        for slide in list(slide_body.findall('./section')):
            # Turn nested sections into divs to not confuse reveal.js
            for c in slide.findall('.//section'):
                c.tag = 'div'

            if ('slide-group' not in
                    (slide.get("class") or "").lower().split(' ')):

                # Add to existing group
                if curgroup is not None:
                    slide_body.remove(slide)
                    curgroup.append(slide)

                continue

            # Make everything until the next slide-group part of this class
            curgroup = html.Section()
            slide_body.insert(list(slide_body).index(slide), curgroup)
            slide_body.remove(slide)
            curgroup.append(slide)


    # <link rel="stylesheet" href="css/reveal.css">
    # <link rel="stylesheet" href="css/theme/default.css" id="theme">
    head.append(css(path("css", "reveal.css"), embed))
    head.append(css(theme_path, embed))

    if printpdf:
        head.append(css(path("css", "print", "pdf.css"), embed))
    else:
        # Embed print-pdf URL semantics
        css_print = read_file(abspath(path("css", "print", "pdf.css")))
        # Escape for HTML page
        css_print = json.dumps(css_print)
        script = html.Script(r"""
                (function() {{
                    if (window.location.search.match( /print-pdf/gi )) {{
                        var printStyle = document.createElement( 'style' );
                        printStyle.type = 'text/css';
                        printStyle.innerHTML = {};
                        document.getElementsByTagName( 'head' )[0].appendChild( printStyle );

                        //Fix for slide numbers...
                        function trySet() {{
                            if (typeof Reveal === "undefined") {{
                                return;
                            }}

                            clearInterval(trySet_id);
                            if (Reveal.getConfig().slideNumber !== 'c') return;

                            $('.slide-number.slide-number-pdf').each(function(idx, el) {{
                                el.innerHTML = idx+1;
                            }});
                        }}
                        var trySet_id = setInterval(trySet, 100);
                    }}
                }})();
                """.format(css_print))
        head.append(script)

    # <script src="lib/js/head.min.js"></script>
    # <script src="js/reveal.js"></script>
    body.append(js(path("lib", "js", "head.min.js"), embed))
    body.append(js(path("js", "reveal.js"), embed))

    head.append(css("rst2html5-reveal.css", embed))

    for c in tree.findall(".//div[@class='rst2html5-html5options']"):
        opts = c.get('data-options')
        if not opts.startswith('revealjs:'):
            continue
        opts = opts[9:]
        for keyval in opts.split(','):
            if '=' not in keyval:
                continue
            k, v = keyval.split('=')
            k = k.strip()
            v = v.strip()
            try:
                v = json.loads(v)
            except ValueError:
                # Leave as string
                pass
            params[k] = v

    params['history'] = True
    param_s = json.dumps(params)
    body.append(
        html.Script("$(function () { Reveal.initialize(%s); });" % param_s))

def impressjs(tree, embed=True, params=None):
    head = tree[0]
    body = tree[1]

    def path(*args):
        return join_path("thirdparty", "impressjs", *args)

    # remove the default style
    #head.remove(head.find("./style"))
    add_class(body, "impress-not-supported")
    failback = html.Div('<div class="fallback-message">' +
        '<p>Your browser <b>doesn\'t support the features required</b> by' +
        'impress.js, so you are presented with a simplified version of this' +
        'presentation.</p>' +
        '<p>For the best experience please use the latest <b>Chrome</b>,' +
        '<b>Safari</b> or <b>Firefox</b> browser.</p></div>')

    slides = html.Div(id="impress")

    for item in list(body):
        body.remove(item)
        slides.append(item)

    body.append(slides)

    # <script src="js/impress.js"></script>
    body.append(js(path("js", "impress.js"), embed))

    body.append(html.Script("impress().init();"))

def bootstrap_css(tree, embed=True, params=None):
    head = tree[0]

    head.append(css(join_path("thirdparty", "bootstrap.css"), embed))

def embed_images(tree, embed=True, params=None):
    import base64, re
    def image_to_data(path):
        lowercase_path = path.lower()

        if lowercase_path.endswith(".png"):
            content_type = "image/png"
        elif lowercase_path.endswith(".jpg"):
            content_type = "image/jpg"
        elif lowercase_path.endswith(".gif"):
            content_type = "image/gif"
        elif lowercase_path.endswith(".svg"):
            content_type = "image/svg+xml"
        else:
            return path

        encoded = base64.b64encode(open(path, 'rb').read()).decode('utf-8')
        content = "data:%s;base64,%s" % (content_type, encoded)
        return content

    for image in tree.findall(".//img"):
        path = image.attrib['src']
        image.set('src', image_to_data(path))

    for style in tree.findall(".//style"):
        def embed_stylesheet_image(match):
            path = match.group(2)
            print("TODO: no hardcoded theme/")
            print("REPLACING " + match.group(0))
            return 'url({})'.format(image_to_data('../theme/' + path))

        style.text = re.sub(r'''(url\(['"])([^'"]+)(['"]\))''', 
                embed_stylesheet_image, style.text)

    for object in tree.findall(".//object[@type='image/svg+xml']"):
        path = object.attrib['data']
        if path.startswith('data:'): continue

        object.set('data', image_to_data(path))

def pygmentize(tree, embed=True, params=None):
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name
    from pygments.formatters import HtmlFormatter

    pygments_formatter = HtmlFormatter()
    body = tree[1]

    def highlight_code(lang, code):
        try:
            lexer = get_lexer_by_name(lang)
        except ValueError:
            # no lexer found - use the text one instead of an exception
            lexer = get_lexer_by_name('text')

        parsed = highlight(code, lexer, pygments_formatter)
        return parsed

    for block in body.findall(".//pre"):
        cls = block.attrib.get('class', '')
        classes = cls.split()
        if 'code' in classes:
            lang_classes = [cls for cls in classes if cls.startswith('lang-')]

            if len(lang_classes) > 0:
                lang = lang_classes[0][5:]

                new_content = highlight_code(lang, block.text)
                block.tag = 'div'
                block.text = new_content

def mathjax(tree, embed=True, params=None):
    body = tree[1]
    params = params or {}
    config_path = params.get("config")
    url = params.get("url", "http://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js")


    if config_path is None:
        content = """
      MathJax.Hub.Config({
        extensions: ["tex2jax.js"],
        jax: ["input/TeX", "output/HTML-CSS"],
        tex2jax: {
          inlineMath: [ ['$','$'], ["\\(","\\)"] ],
          displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
          processEscapes: true
        },
        "HTML-CSS": { availableFonts: ["TeX"] }
      });
        """
    else:
        with open(config_path) as f_in:
            content = f_in.read()

    body.append(html.Script(content, type="text/x-mathjax-config"))
    body.append(html.Script(src=url))


PROCESSORS = [
    ("mathjax", {
        "name": "add mathjax support",
        "processor": mathjax
    }),
    ("jquery", {
        "name": "add jquery",
        "processor": jquery
    }),
    ("pretty_print_code", {
        "name": "pretty print code",
        "processor": pretty_print_code
    }),
    ("pygments", {
        "name": "pygments",
        "processor": pygmentize
    }),
    ("deck_js", {
        "name": "deck.js",
        "processor": deckjs
    }),
    ("reveal_js", {
        "name": "reveal.js",
        "processor": revealjs
    }),
    ("impress_js", {
        "name": "impress.js",
        "processor": impressjs
    }),
    ("bootstrap_css", {
        "name": "bootstrap css",
        "processor": bootstrap_css
    }),
    ("embed_images", {
        "name": "embed images",
        "processor": embed_images
    }),
    ("add_js", {
        "name": "add js files",
        "processor": add_js
    })
]


class Html5Options(nodes.Inline, nodes.TextElement): pass
class RevealJsOpts(Directive):
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    has_content = False

    def run(self):
        return [Html5Options('', 'revealjs:' + self.arguments[0])]
directives.register_directive('reveal-js-opts', RevealJsOpts)

class Code(Directive):
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    has_content = True

    def run(self):
        language = self.arguments[0]
        content = self.content

        attrs = {
            'class': "code lang-" + language
        }

        return [nodes.literal_block('', "\n".join(content), **attrs)]

class Slide3D(Directive):

    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {
            'x': int,
            'y': int,
            'z': int,
            'rotate': int,
            'rotate-x': int,
            'rotate-y': int,
            'scale': int,
            'class': directives.unchanged,
            'id': directives.unchanged,
            'title': directives.unchanged
    }
    has_content = True

    def run(self):
        attributes = {}

        for key, value in self.options.items():
            if key in ('class', 'id', 'title'):
                attributes[key] = value
            else:
                attributes['data-' + key] = value

        node = nodes.container(rawsource=self.block_text, **attributes)
        self.state.nested_parse(self.content, self.content_offset, node)

        return [node]

class Video(Directive):
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    has_content = False
    option_spec = {
            'autoplay': bool,
            'preload': bool,
            'poster': str,
            'controls': bool,
            'height': int,
            'width': int,
            'loop': bool,
            'muted': bool,
            'class': directives.unchanged,
            'id': directives.unchanged,
            'title': directives.unchanged
    }

    def run(self):
        src = self.arguments[0]
        opts = self.options

        code = '<video src="%s"' % src

        if opts.get('controls'):
            code += ' controls="true"'

        if opts.get('muted'):
            code += ' muted="true"'

        if opts.get('loop'):
            code += ' loop="true"'

        if opts.get('autoplay'):
            code += ' autoplay="true"'

        preload = opts.get('preload')

        width = opts.get('width')
        if width is not None:
            code += ' width="%s"' % width

        height = opts.get('height')
        if height is not None:
            code += ' height="%s"' % height

        poster = opts.get('poster')
        if poster is not None:
            code += ' poster="%s"' % poster

        if preload:
            if preload in ['none', 'metadata', 'auto']:
                code += ' preload="%s"' % preload

        code += '></video>'

        return [nodes.raw('', code, format='html')]

directives.register_directive('slide-3d', Slide3D)
directives.register_directive('code-block', Code)
directives.register_directive('video', Video)
