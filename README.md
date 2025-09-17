# NumPy to GenAI

NumPy to GenAI is an open collection of hands-on lessons that lead learners from
image-processing fundamentals all the way to creative, AI-powered visuals. The
project marries approachable NumPy exercises with progressively advanced
examples—fractals, visual effects, animations, and machine-learning driven art—
packaged as a Sphinx documentation site ready for classrooms, workshops, and
self-paced study.

## Highlights

- **Progressive curriculum** – five themed tracks (First Steps, Elementary
  Geometry, Machine Learning, Effects, Fractals, Animations) that build on each
  other.
- **Creative coding focus** – every lesson pairs concepts with runnable Python
  scripts and illustrative output images or GIFs.
- **Ready-to-teach documentation** – the site is powered by Sphinx with the
  PyData theme, Sphinx Design components, and copy-friendly code blocks.
- **Open and extensible** – organised content folders, solutions, and
  experimental areas invite adaptation for new workshops or courses.

## Repository Layout

```
├── conf.py                  # Sphinx configuration (PyData theme + custom navbar)
├── index.rst                # Landing page wiring top-level chapters
├── content/                 # Lesson material grouped by chapter
│   ├── 01_first_steps/      # NumPy basics: grayscale, RGB, tiling, flags…
│   ├── 02_elementary_geometry/
│   ├── 03_machine_learning/
│   ├── 04_effects/
│   ├── 05_fractals/
│   └── 06_animations/
├── images/                  # Shared illustration assets
├── solutions/               # Reference solutions for selected challenges
├── _static/                 # Custom CSS/JS, favicon, logo assets
├── _templates/              # Template overrides (navbar dropdowns)
├── build/                   # Generated HTML (ignored in deployments)
├── requirements.txt         # Runtime dependencies for the examples
├── dev_requirements.txt     # Docs build / authoring dependencies
└── Makefile                 # Convenience wrapper around `sphinx-build`
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/burakkagann/numpy-to-genAI.git
cd numpy-to-genAI
```

### 2. Create and activate a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

Install the packages used in the examples:

```bash
pip install -r requirements.txt
```

If you also plan to build the documentation locally, add the authoring tools:

```bash
pip install -r dev_requirements.txt
```

## Building the Documentation

The site is authored with Sphinx. Build the HTML version with either `make` or
`python -m sphinx`:

```bash
make html
# or
sphinx-build -b html . build/html
```

The generated site lives in `build/html/index.html`. Open it directly in a
browser or serve it via any static-site host.

## Running Lesson Code

Every subfolder in `content/` contains a `README.rst` tutorial, Python scripts,
images, and (for animations) GIFs. To experiment with a lesson:

```bash
python content/03_machine_learning/kmeans/decolorize.py
```

Most scripts rely only on the packages from `requirements.txt`. Some notebooks
(e.g. convolution) require Jupyter if you want to explore them interactively.

## Customising the Site

- `conf.py` controls theme options, navbar dropdowns, and custom icon links.
- `_static/content.css` and related files tweak styling (headers, footer,
  branding).
- New chapters can be added by creating a folder in `content/` and referencing
  it from the landing page via another hidden `toctree` in `index.rst`.

## Contributing

Contributions are welcome! Ideas include:

1. Adding new creative coding lessons or variations on existing exercises.
2. Improving visuals, GIFs, or documentation copy.
3. Enhancing build tooling (CI, automated tests, linting) or portability.

Open an issue or pull request describing your proposal. Please keep the
educational tone and ensure new scripts run with the listed dependencies.

## License

This project is distributed under the terms of the [MIT License](LICENSE).

## Acknowledgements

The curriculum is inspired by the teaching work of Kristian Rother and Burak
Kağan Yılmazer, and leverages the [PyData Sphinx Theme] for a clean, accessible
presentation.

[PyData Sphinx Theme]: https://pydata-sphinx-theme.readthedocs.io/
