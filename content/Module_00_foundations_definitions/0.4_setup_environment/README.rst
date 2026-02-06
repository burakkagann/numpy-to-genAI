.. _module-0-4-1-setup-environment:

=====================================
0.4.1 - Setup Environment
=====================================

:Duration: 8-10 minutes
:Level: Beginner


Overview
========

This module will guide you through creating a clean, isolated Python environment and installing all the dependencies needed for your journey from basic image processing to advanced generative AI art.

**Learning Objectives**

By completing this module, you will:

* Create an isolated Python virtual environment for the course
* Install and verify all required libraries and dependencies
* Understand the purpose and capabilities of each major library category
* Test your setup with hands on verification exercises
* Troubleshoot common installation issues


Quick Start: Environment Check
==============================

Before we begin, let's verify your Python installation:

.. code-block:: bash
   :caption: Check your Python version

   python --version
   # or
   python3 --version

.. admonition:: System Requirements ⚙️

   **Minimum Requirements:**

   * **Python**: 3.8 or higher (3.11.9 recommended)
   * **Operating System**: macOS, Windows 10+, or Linux
   * **RAM**: 4GB minimum (8GB+ recommended for AI work)
   * **Storage**: 2GB free space for libraries and dependencies

   **GPU Support (Optional but Recommended):**

   * NVIDIA GPU with CUDA support for accelerated AI training
   * macOS with Metal Performance Shaders for Apple Silicon

Download: Requirements File
=================================

Before setting up your environment, you can download the complete requirements file that contains all necessary dependencies for the course.

.. admonition:: Download Requirements File
   :class: tip


   :download:`requirements.txt <../../../requirements.txt>`

Step 1: Create Your Virtual Environment
=======================================

A virtual environment isolates your project dependencies from your system Python installation, preventing conflicts and ensuring consistency.

Create and activate environment
-------------------------------

.. code-block:: bash
   :caption: Create virtual environment

   # Navigate to your project directory
   cd path/to/pixels-to-genAI

   # Create virtual environment
   python -m venv .venv

   # Alternative command
   python3 -m venv .venv

.. code-block:: bash
   :caption: Activate virtual environment

   # macOS and Linux
   source .venv/bin/activate

   # Windows Command Prompt
   .venv\Scripts\activate

   # Windows PowerShell
   .venv\Scripts\Activate.ps1

.. tip::

   **Success Indicator**: When activated, your terminal prompt should change to show `(.venv)` at the beginning, indicating you're working in the virtual environment.

Verify activation
-----------------

.. code-block:: bash
   :caption: Verify environment activation

   # Check which Python you're using (should point to .venv)
   which python    # macOS/Linux
   where python    # Windows

   # Verify pip is also from the virtual environment
   which pip       # macOS/Linux
   where pip       # Windows

Step 2: Install Core Dependencies
==================================

Now we'll install all the libraries needed for the course using the requirements file. You can either use the downloaded requirements.txt file from the previous section or the one included in the repository.

Install from requirements.txt
------------------------------

.. code-block:: bash
   :caption: Install all dependencies

   # Upgrade pip first (recommended)
   pip install --upgrade pip

   # Option 1: If you downloaded the requirements.txt file
   pip install -r /path/to/downloaded/requirements.txt

   # Option 2: Use the requirements.txt from the repository
   pip install -r requirements.txt

.. note::

   **Installation Time**: This process typically takes 5-15 minutes depending on your internet connection and system. The AI libraries (PyTorch, TensorFlow) are the largest downloads.

Verify installation
-------------------

.. code-block:: bash
   :caption: Check installed packages

   # List all installed packages
   pip list

   # Check specific key packages
   pip show numpy pillow opencv-python torch tensorflow

Step 3: Understanding Your Toolkit
===================================

Let's explore the libraries you've just installed and understand their roles in generative art creation.

Core Numerical Computing
-------------------------

These libraries form the mathematical foundation for all our work:

**- NumPy**
   The fundamental package for scientific computing. All image data, mathematical operations, and array manipulations start here.

**- SciPy**
   Advanced mathematical functions, optimization, and signal processing tools essential for complex generative algorithms.

**- Pandas**
   Data manipulation and analysis, particularly useful when working with datasets for AI training or data-driven art.

.. code-block:: python
   :caption: Core libraries quick test

   import numpy as np
   import scipy as sp
   import pandas as pd

   print(f"NumPy version: {np.__version__}")
   print(f"SciPy version: {sp.__version__}")
   print(f"Pandas version: {pd.__version__}")

Image Processing & Computer Vision
-----------------------------------

These libraries handle image creation, manipulation, and analysis:

**- Pillow**
   Python Imaging Library for basic image operations—loading, saving, resizing, and format conversion.

**- OpenCV**
   Computer vision powerhouse for advanced image processing, real-time video manipulation, and feature detection.

**- ImageIO**
   Versatile image and video I/O library, especially useful for creating animations and GIFs.

.. code-block:: python
   :caption: Image processing libraries test

   from PIL import Image
   import cv2
   import imageio

   print(f"Pillow (PIL) version: {Image.__version__}")
   print(f"OpenCV version: {cv2.__version__}")
   print(f"ImageIO version: {imageio.__version__}")

Machine Learning & AI Frameworks
---------------------------------

The cutting edge tools for intelligent art generation:

**- Scikit-learn**
   Traditional machine learning algorithms for pattern recognition, clustering, and data analysis.

**- PyTorch**
   Dynamic neural network framework, preferred for research and experimentation in generative AI.

**- TensorFlow**
   Google's machine learning platform, excellent for production AI applications and complex models.

.. code-block:: python
   :caption: AI frameworks test

   import sklearn
   import torch
   import tensorflow as tf

   print(f"Scikit-learn version: {sklearn.__version__}")
   print(f"PyTorch version: {torch.__version__}")
   print(f"TensorFlow version: {tf.__version__}")

   # Check GPU availability
   print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
   print(f"TensorFlow GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

Visualization & Graphics
-------------------------

Tools for creating beautiful visual outputs:

**- Matplotlib**
   The foundational plotting library for creating charts, graphs, and scientific visualizations.

**- Seaborn**
   Statistical data visualization built on matplotlib, ideal for exploring patterns in data.

**- Pygame**
   Real-time graphics and interactive applications, great for games and interactive art installations.

.. code-block:: python
   :caption: Visualization libraries test

   import matplotlib
   import seaborn as sns
   import pygame

   print(f"Matplotlib version: {matplotlib.__version__}")
   print(f"Seaborn version: {sns.__version__}")
   print(f"Pygame version: {pygame.version.ver}")

Specialized Creative Tools
--------------------------

Unique libraries for specific generative art techniques:

**- Noise & Perlin-noise**
   Generate natural-looking random patterns which is essential for organic textures, terrains, and flowing animations.

**- Librosa & SoundFile**
   Audio analysis and manipulation for music visualization and sound reactive art.

**- Trimesh & Pyglet**
   3D geometry processing and OpenGL graphics for three dimensional generative art.

.. code-block:: python
   :caption: Specialized tools test

   import noise
   import librosa
   import trimesh
   import pyglet

   print(f"Noise library version: {noise.__version__}")
   print(f"Librosa version: {librosa.__version__}")
   print(f"Trimesh version: {trimesh.__version__}")
   print(f"Pyglet version: {pyglet.version}")

Development & Documentation Tools
----------------------------------

Supporting tools for learning and development:

**- Jupyter & IPywidgets**
   Interactive notebooks for experimentation and learning, with widgets for parameter control.


.. code-block:: python
   :caption: Development tools test

   import jupyter
   import ipywidgets

   print(f"Jupyter version: {jupyter.__version__}")
   print(f"IPywidgets version: {ipywidgets.__version__}")


Summary
=======

Congratulations! You've successfully set up a complete development environment for generative art and AI creation.

**Your toolkit now includes:**

* **Numerical computing**: NumPy, SciPy, Pandas for mathematical foundations
* **Image processing**: Pillow, OpenCV, ImageIO for visual manipulation
* **AI frameworks**: PyTorch, TensorFlow, Scikit-learn for intelligent art generation
* **Visualization**: Matplotlib, Seaborn for data-driven aesthetics
* **Creative tools**: Noise generators, audio processing, 3D graphics support
* **Development environment**: Jupyter notebooks for interactive experimentation

.. tip::

   **Remember to activate your environment**: Always run `source .venv/bin/activate` (or equivalent) before working on course exercises to ensure you're using the correct libraries.

References
==========

.. [Python] Python Software Foundation. "Python Documentation." https://docs.python.org/

.. [NumPy] Harris, Charles R., et al. "Array programming with NumPy." *Nature* 585.7825 (2020): 357-362.

.. [PyTorch] Paszke, Adam, et al. "PyTorch: An imperative style, high-performance deep learning library." *Advances in neural information processing systems* 32 (2019).

.. [TensorFlow] Abadi, Martín, et al. "TensorFlow: Large-scale machine learning on heterogeneous systems." (2015).

.. [OpenCV] Bradski, Gary. "The opencv library." *Dr. Dobb's journal of software tools* 120 (2000): 122-125.

.. [SciPy] Virtanen, Pauli, et al. "SciPy 1.0: fundamental algorithms for scientific computing in Python." *Nature methods* 17.3 (2020): 261-272.

.. [Matplotlib] Hunter, John D. "Matplotlib: A 2D graphics environment." *Computing in science & engineering* 9.03 (2007): 90-95.