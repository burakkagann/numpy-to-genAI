# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NumPy to GenAI'
copyright = '2024, Kristian Rother & Burak Kagan Yilmazer'
author = 'Kristian Rother & Burak Kağan Yılmazer'
release = '1.0'
html_title = f"{project}"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_design',
    'sphinx_copybutton',
    'sphinx.ext.todo',
    'myst_parser',
    ]

exclude_patterns = ['experimental', '_build', 'Thumbs.db', '.DS_Store']

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinxawesome_theme'
html_static_path = ['_static']
html_logo = None
html_favicon = None

html_css_files = [
    'github-icon.css',
]
html_js_files = [
    'github-icon.js',
]
html_theme_options = {
    "source_repository": "https://github.com/burakkagann/numpy-to-genAI",
    "source_branch": "main",
    "source_directory": "",
}