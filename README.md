# NumPy to GenAI: A Progressive Educational Framework for Generative Art

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://burakkagann.github.io/numpy-to-genAI/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> **From Pixels to AI**: A comprehensive educational pathway that bridges fundamental NumPy operations with modern AI-driven generative art, inspired by artists like Refik Anadol.

![Title Image](images/title.png)

## 🎯 Project Vision

This repository extends the excellent educational foundation created by **Dr. Kristian Rother** in his original [Graphics with NumPy](https://github.com/krother/generative_art) tutorial. Building upon his proven pedagogical approach, we create a progressive 6-month curriculum that prepares students for careers in creative technology and AI-driven art.

**Core Philosophy:**
- **Learn by Creating**: Every concept demonstrated through visual, interactive examples
- **Progressive Complexity**: From basic array operations to advanced neural network visualizations  
- **AI-Prepared**: Each exercise builds foundational knowledge for modern AI/ML workflows
- **Industry-Ready**: Real-world applications using professional tools and techniques

## 🚀 Quick Start

### Installation

If you're using [Anaconda](https://www.anaconda.com/), you should have most libraries already. Otherwise:

```bash
# Clone the repository
git clone https://github.com/burakkagann/numpy-to-genAI.git
cd numpy-to-genAI

# Install dependencies
pip install -r requirements.txt

# For development (documentation, testing)
pip install -r dev_requirements.txt
```

### First Steps

Start with the foundational concepts and progress through the curriculum:

```python
# Example: Create your first AI-ready image
import numpy as np
from PIL import Image

# Create a tensor-ready image (height, width, channels)
image = np.zeros((200, 200, 3), dtype=np.uint8)
image[:, :, 0] = 255  # Red channel
image[:, :, 2] = 128  # Blue channel

# Save as PNG
Image.fromarray(image).save('my_first_tensor.png')
```

## 📚 Curriculum Structure

### **Part I: Enhanced NumPy Graphics** *(Months 1-2)*

Building on Dr. Rother's excellent foundation with AI-preparatory extensions:

#### **First Steps** - *Enhanced with AI Concepts*
| Original (by Dr. Rother) | New AI Extensions |
|--------------------------|-------------------|
| [Creating Images](grayscale/README.rst) | **Pixel as Data** - Understanding images as tensors |
| [Color](rgb/README.rst) | **Color Spaces** - HSV, LAB for style transfer |
| [Random Tiles](random_tiles/README.rst) | **Noise Functions** - Perlin/Simplex for generative patterns |
| [Flags](flags/README.rst) | **Data Visualization** - Real-world datasets as art |
| [Repeat](repeat/README.rst) | **Pattern Recognition** - Foundation for CNNs |

#### **Elementary Geometry** - *Extended to 3D*
| Original | New Extensions |
|----------|----------------|
| [Stars](stars/README.rst) | **3D Transformations** - Extending to 3D with `meshcat-python` |
| [Lines](lines/README.rst) | **Parametric Surfaces** - Mathematical foundations |
| [Gradient](gradient/README.rst) | **Interactive Geometry** - Real-time manipulation |
| [Triangles](triangles/README.rst) | **Mesh Operations** - 3D model processing |
| [Circles](circles/README.rst) | **Spherical Coordinates** - Advanced transformations |
| [Spiral](spiral/README.rst) | **Fractal Geometry** - Recursive patterns |
| [Mask](mask/README.rst) | **Alpha Channels** - Transparency for compositing |

#### **Machine Learning** - *Significantly Extended*
| Original | New Comprehensive Additions |
|----------|---------------------------|
| [K-Means](kmeans/README.rst) | **Statistical Foundations for AI** |
| [Decision Tree](dtree/README.rst) | - PCA Visualization for latent spaces |
| [Convolution](convolution/README.rst) | - Gaussian Mixtures (VAE foundations) |
| [Monte Carlo](montecarlo/README.rst) | **Introduction to Neural Networks** |
| | - Perceptron Drawing (visual neurons) |
| | - Backpropagation Visualization |
| | - Simple MNIST Classifier |
| | **Generative Models Basics** |
| | - Autoencoder Art |
| | - Simple GAN (28x28 patterns) |
| | - Style Matrices extraction |

#### **Effects & Animations** - *Enhanced with Advanced Techniques*
| Original | New Extensions |
|----------|----------------|
| [Rotation](rotate/README.rst) | **Fourier Transforms** - Frequency domain |
| [Shadow](shadow/README.rst) | **Reaction-Diffusion** - Turing patterns |
| [Warhol](warhol/README.rst) | **Particle Systems** - Fluid simulation foundations |
| [Puzzle](puzzle/README.rst) | **Real-time Rendering** - Performance optimization |
| [Contour Lines](contour/README.rst) | **Advanced Shaders** - GLSL introduction |
| [Edge Detection](sobel/README.rst) | **Computer Vision** - Feature detection |

#### **Fractals** - *Extended with AI Applications*
| Original | New Extensions |
|----------|----------------|
| [Dragon Curve](dragon_curve/README.rst) | **Fractal Neural Networks** - Self-similar architectures |
| [Mandelbrot](mandelbrot/README.rst) | **Complex Dynamics** - Chaos theory in AI |
| [Fractal Square](fractal_square/README.rst) | **Recursive Generation** - Foundation for GANs |

### **Part II: Bridge to Real-Time Systems** *(Month 3)*

#### **New Chapter: "From NumPy to TouchDesigner"**
- **Node-Based Programming**: Recreate NumPy exercises in TouchDesigner
- **Python Integration**: Script TOP with NumPy arrays
- **Real-time Processing**: Convert static examples to interactive
- **Sensor Integration**: Webcam, microphone, OSC, MIDI inputs
- **Performance Optimization**: GPU vs CPU, instancing, compute shaders

### **Part III: AI Integration** *(Months 4-5)*

#### **New Chapter: "Practical AI for Artists"**

**Pre-trained Models:**
- **Style Transfer Pipeline**: VGG19 → TouchDesigner at 30fps
- **Pose-Driven Particles**: MediaPipe → Refik Anadol-inspired forms
- **Text-to-Visual**: Stable Diffusion API integration

**Custom Model Training:**
- **Personal Style GAN**: StyleGAN2-ADA with student datasets
- **Movement Predictor**: LSTM for gesture sequences
- **Multi-modal Installation**: Audio + Video + Sensors ensemble

#### **New Chapter: "Data as Material"**
- **Weather Data** → Abstract landscapes (NOAA API)
- **Social Media** → Color fields (Twitter sentiment)
- **Urban Sensors** → City portraits (OpenData)
- **Space Imagery** → Cosmic narratives (NASA API)

### **Part IV: Capstone Projects** *(Month 6)*

**Track A: "Anadol Study" - Data Sculpture**
- Large dataset visualization (1GB+)
- Custom trained GAN/VAE
- 4K output at 60fps

**Track B: "Interactive Installation"**
- Multi-user computer vision
- Real-time generative responses
- Spatial audio integration

**Track C: "Educational Tool"**
- Interactive AI tutorial system
- Web deployment
- Open-source contribution

## 🛠 Technical Stack

### Core Dependencies
```python
# Numerical Computing
numpy>=1.21.0
scipy>=1.7.0

# Image Processing  
pillow>=8.3.0
opencv-python>=4.5.0
imageio>=2.9.0

# Machine Learning
scikit-learn>=1.0.0
torch>=1.9.0
tensorflow>=2.8.0

# 3D Visualization
meshcat-python
matplotlib>=3.4.0

# Real-time Systems
touchdesigner  # Non-commercial license
python-osc
mido

# Generative Art
noise  # Perlin/Simplex
librosa  # Audio processing
```

### Hardware Recommendations
- **Minimum**: GTX 1660 (6GB VRAM), 16GB RAM
- **Recommended**: RTX 3060 (12GB VRAM), 32GB RAM  
- **Ideal**: RTX 4080 (16GB VRAM), 64GB RAM
- **Alternative**: Cloud GPUs via Colab/Paperspace

## 🎓 Learning Paths

### For Beginners
1. Start with **First Steps** chapters
2. Progress through **Elementary Geometry**
3. Explore **Machine Learning** concepts
4. Experiment with **Effects** and **Animations**

### For Intermediate Learners
1. Focus on **AI Extensions** in each chapter
2. Complete **TouchDesigner** integration
3. Build **Custom Models** projects
4. Explore **Data as Material** concepts

### For Advanced Users
1. Jump to **AI Integration** chapters
2. Complete **Capstone Projects**
3. Contribute to **Open Source** development
4. Develop **Industry Applications**

## 📖 Documentation

- **[Full Documentation](https://burakkagann.github.io/numpy-to-genAI/)** - Complete curriculum with examples
- **[API Reference](docs/api.md)** - Function documentation
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions
- **[Contributing](CONTRIBUTING.md)** - How to contribute to the project

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/burakkagann/numpy-to-genAI.git
cd numpy-to-genAI
pip install -r dev_requirements.txt

# Run tests
pytest

# Build documentation
make html
```

## 📄 License & Credits

### Original Work
This project builds upon the excellent educational foundation created by **Dr. Kristian Rother** in his original [Graphics with NumPy](https://github.com/krother/generative_art) tutorial. All original content is preserved and properly attributed.

**Original License**: MIT License  
**Original Author**: Dr. Kristian Rother (`kristian.rother@posteo.de`)

### Extensions & Modifications
- **Extended by**: Burak Kagan (Master's Thesis Project)
- **License**: MIT License (inherited from original)
- **Academic Context**: Master's Thesis - "From Pixels to AI: A Progressive Educational Framework for Generative Art"

### References
- [The Brandenburg Gate image](https://commons.wikimedia.org/wiki/File:Brandenburger_Tor_abends.jpg) by Thomas Wolf, www.foto-tw.de / Wikimedia Commons / CC BY-SA 3.0
- Refik Anadol's generative art techniques and methodologies
- TouchDesigner community resources and tutorials

## 🌟 Acknowledgments

- **Dr. Kristian Rother** for creating the foundational educational materials
- **Refik Anadol** for inspiring the AI-driven generative art direction
- **TouchDesigner Community** for real-time graphics expertise
- **Open Source Contributors** who make this educational journey possible

---

## 📞 Contact & Support

- **Repository**: [github.com/burakkagann/numpy-to-genAI](https://github.com/burakkagann/numpy-to-genAI)
- **Documentation**: [burakkagann.github.io/numpy-to-genAI](https://burakkagann.github.io/numpy-to-genAI/)
- **Issues**: [GitHub Issues](https://github.com/burakkagann/numpy-to-genAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/burakkagann/numpy-to-genAI/discussions)

---

*"The best way to learn is by creating. The best way to create is by understanding the mathematics behind the magic."* - Building upon Dr. Rother's educational philosophy
