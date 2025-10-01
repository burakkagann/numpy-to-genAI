# NumPy to GenAI

NumPy to GenAI is an open collection of hands-on lessons that lead learners from
image-processing fundamentals all the way to creative, AI-powered visuals. The
project marries approachable NumPy exercises with progressively advanced
examples—fractals, visual effects, animations, and machine-learning driven art—
packaged as a Sphinx documentation site ready for classrooms, workshops, and
self-paced study.

## Highlights

- **Progressive curriculum** – 15 comprehensive modules that build from basic
  pixel manipulation to advanced generative AI applications.
- **Creative coding focus** – every lesson pairs concepts with runnable Python
  scripts and illustrative output images or GIFs.
- **Ready-to-teach documentation** – the site is powered by Sphinx with the
  PyData theme, Sphinx Design components, and copy-friendly code blocks.
- **Open and extensible** – organised content folders, solutions, and
  experimental areas invite adaptation for new workshops or courses.

## Learning Modules

The curriculum is structured as 15 progressive modules:

- **Module 0**: Python NumPy Fundamentals
- **Module 1**: Pixel Fundamentals
- **Module 2**: Geometry Mathematics
- **Module 3**: Noise Patterns
- **Module 4**: Image Processing Filters
- **Module 5**: Fractals Self-Similarity
- **Module 6**: Advanced Mathematical Art
- **Module 7**: Animation Motion Graphics
- **Module 8**: Real-Time Interactive Systems
- **Module 9**: Data Visualization Techniques
- **Module 10**: Machine Learning Foundations
- **Module 11**: Neural Networks Computer Vision
- **Module 12**: Generative AI Art Creation
- **Module 13**: Advanced AI Applications
- **Module 14**: Capstone Projects

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
python content/Module_01_pixel_fundamentals/1.1_grayscale_color_basics/1.1.1_color_basics/rgb/rgb.py
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
