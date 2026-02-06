# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from sphinx.addnodes import toctree as TocTreeNode

project = 'Pixels2GenAI'
copyright = '2026, Burak Kağan Yılmazer & Kristian Rother'
author = 'Burak Kağan Yılmazer & Kristian Rother'
release = '3.0'
html_title = f"{project}"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_design',
    'sphinx_copybutton',
    'sphinx.ext.todo',
    'myst_parser',
    ]

exclude_patterns = ['experimental', '_build', 'Thumbs.db', '.DS_Store', 'stylegan_env', '.venv', 'venv', '*_env', 'solutions']

language = 'en'
templates_path = ['_templates']


def _build_chapter_nav(app):
    """Collect hidden top-level toctrees to drive the header dropdowns."""
    env = app.builder.env
    root_doc = app.config.root_doc
    doctree = env.get_doctree(root_doc)
    chapter_nav = []

    for node in doctree.traverse(TocTreeNode):
        caption = node.get('caption')
        if not caption or not node.get('hidden'):
            continue

        items = []
        for title, ref in node['entries']:
            if not ref:
                continue
            resolved_title = title or env.titles[ref].astext()
            items.append({
                'title': resolved_title,
                'ref': ref,
            })

        if items:
            chapter_nav.append({
                'title': caption,
                'items': items,
            })

    return chapter_nav


def _inject_chapter_nav(app, pagename, templatename, context, doctree):
    """Expose chapter navigation data to all templates."""
    if not hasattr(app.env, '_chapter_nav'):
        app.env._chapter_nav = _build_chapter_nav(app)
    context['chapter_nav'] = app.env._chapter_nav

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_logo = None
html_favicon = '_static/favicon-64x64.png'

html_css_files = [
    'github-icon.css',
    'footer.css',
    'content.css',
    'scroll-progress.css',
    'carousel.css',
    'modules.css',
]
html_js_files = [
    'github-icon.js',
    'scroll-progress.js',
    'carousel.js',
]
html_theme_options = {
    "source_repository": "https://github.com/burakkagann/numpy-to-genAI",
    "source_branch": "main",
    "source_directory": "",
    "logo": {
        "text": "Pixels to GenAI",
        "image_light": "favicon-192x192.png",
        "image_dark": "favicon-192x192.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/burakkagann/numpy-to-genAI",
            "icon": "fa-brands fa-github",
        },
    ],
}


def setup(app):
    app.connect('html-page-context', _inject_chapter_nav)
