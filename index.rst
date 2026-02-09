:hide-toc:

Pixels2GenAI
===============

An open source educational platform that creates a comprehensive pathway into AI-driven generative art, bridging mathematical and visual foundations to modern creative AI techniques. This 15-module curriculum takes learners from fundamental pixel manipulation and NumPy operations through advanced generative models, neural networks, and real-time interactive systems.

.. raw:: html

   <div class="cube-container">
     <div class="cube-gallery"></div>
   </div>
   <p class="cube-caption">Showcasing exercises from across all modules</p>



.. |icon-pathway| replace:: :octicon:`book;1.4em;sd-text-secondary`
.. |icon-learners| replace:: :octicon:`mortar-board;1.4em;sd-text-secondary`
.. |icon-theory| replace:: :octicon:`beaker;1.4em;sd-text-secondary`
.. |icon-curriculum| replace:: :octicon:`stack;1.4em;sd-text-secondary`
.. |icon-community| replace:: :octicon:`people;1.4em;sd-text-secondary`
.. |icon-audiences| replace:: :octicon:`project;1.4em;sd-text-secondary`


Project at a Glance
-------------------

.. grid:: 1 1 2 3
   :gutter: 2

   .. grid-item-card:: |icon-pathway| Educational Pathway
      :class-card: pst-card

      Creating an approachable journey into AI-driven generative art that connects visual intuition with modern machine learning practice through 15 progressive modules.

   .. grid-item-card:: |icon-learners| Designed for Learners
      :class-card: pst-card

      Materials welcome semi-beginners through semi-experienced programmers, with optional guidance for newcomers willing to self-study foundational topics.

   .. grid-item-card:: |icon-theory| Theory Meets Practice
      :class-card: pst-card

      Each module balances mathematical ideas, NumPy techniques, and creative coding projects so learners see how concepts translate into visuals and AI applications.

   .. grid-item-card:: |icon-curriculum| Progressive Curriculum
      :class-card: pst-card

      Lessons build sequentially from image fundamentals through fractals, simulations, and generative AI, allowing confidence to grow alongside complexity.

   .. grid-item-card:: |icon-community| Creative Community Impact
      :class-card: pst-card

      Supports programming teachers, self-learners, artists, and data scientists who want memorable exercises for classes, portfolios, or passion projects.

   .. grid-item-card:: |icon-audiences| Use Cases & Audiences
      :class-card: pst-card

      Ideal for course builders, independent learners, curious engineers, and creatives exploring AI-enhanced artistry across classrooms and studios.


Repository
----------

The source code is available on GitHub:

`https://github.com/burakkagann/Pixels2GenAI <https://github.com/burakkagann/Pixels2GenAI>`__

Clone the repository:

.. code-block:: bash

   git clone https://github.com/burakkagann/Pixels2GenAI.git
   cd Pixels2GenAI


.. dropdown:: Installation
   :class-title: sd-fs-5

   **System Requirements:**

   - `Python 3.11.9 <https://www.python.org/downloads/release/python-3119/>`__ (recommended)
   - For neural network modules (7+): NVIDIA GPU recommended but not required
   - For diffusion models (Module 12): 8GB RAM minimum, GPU strongly recommended

   **Option 1: Using pyproject.toml (Recommended)**

   .. code-block:: bash

      # Core dependencies (Modules 0-6)
      pip install .

      # With machine learning packages (Modules 7-13)
      pip install .[ml]

      # All optional dependencies
      pip install .[all]

   **Option 2: Using requirements.txt**

   .. code-block:: bash

      pip install -r requirements.txt

Learning Modules
----------------

.. raw:: html

   <div class="module-section-header">Creative Coding Foundations</div>
   <div class="module-section-subtitle">Modules 0-6 · ~80 exercises · Start here if new to creative coding</div>

.. dropdown:: Module 0: Foundations & Definitions


   Setting the conceptual and technical groundwork for generative art and AI.

   .. toctree::
      :maxdepth: 1

      0.1 - What Is Generative Art <content/Module_00_foundations_definitions/0.1_what_is_generative_art/README.rst>
      0.2 - Defining AI ML Algorithms <content/Module_00_foundations_definitions/0.2_defining_ai_ml_algorithms/README.rst>
      0.4 - Setup Environment <content/Module_00_foundations_definitions/0.4_setup_environment/README.rst>

.. dropdown:: Module 1: Pixel Fundamentals


   Understanding images at the atomic level through color theory and manipulation patterns.

   **1.1 - Grayscale & Color Basics**

   .. toctree::
      :maxdepth: 1

      1.1.1 - Color Basics <content/Module_01_pixel_fundamentals/1.1_grayscale_color_basics/1.1.1_color_basics/README.rst>

   .. raw:: html

      <span class="incomplete-exercise">1.1.2 - Color Theory Spaces</span>

   **1.2 - Pixel Manipulation Patterns**

   .. toctree::
      :maxdepth: 1

      1.2.1 - Random Patterns <content/Module_01_pixel_fundamentals/1.2_pixel_manipulation_patterns/1.2.1_random_patterns/README.rst>
      1.2.2 - Cellular Automata <content/Module_01_pixel_fundamentals/1.2_pixel_manipulation_patterns/1.2.2_cellular_automata/README.rst>

   .. raw:: html

      <span class="incomplete-exercise">1.2.3 - Reaction Diffusion</span>

   **1.3 - Structured Compositions**

   .. toctree::
      :maxdepth: 1

      1.3.1 - Flags <content/Module_01_pixel_fundamentals/1.3_structured_compositions/1.3.1_flags/README.rst>
      1.3.2 - Repeat <content/Module_01_pixel_fundamentals/1.3_structured_compositions/1.3.2_repeat/README.rst>

   .. raw:: html

      <span class="incomplete-exercise">1.3.3 - Truchet Tiles</span>
      <span class="incomplete-exercise">1.3.4 - Wang Tiles</span>

.. dropdown:: Module 2: Geometry & Mathematics


   Mathematical foundations for generative art through shapes, coordinates, and mathematical patterns.

   **2.1 - Basic Shapes & Primitives**

   .. toctree::
      :maxdepth: 1

      2.1.1 - Lines <content/Module_02_geometry_mathematics/2.1_basic_shapes_primitives/2.1.1_lines/README.rst>
      2.1.2 - Triangles <content/Module_02_geometry_mathematics/2.1_basic_shapes_primitives/2.1.2_triangles/README.rst>
      2.1.3 - Circles <content/Module_02_geometry_mathematics/2.1_basic_shapes_primitives/2.1.3_circles/README.rst>
      2.1.4 - Stars <content/Module_02_geometry_mathematics/2.1_basic_shapes_primitives/2.1.4_stars/README.rst>

   .. raw:: html

      <span class="incomplete-exercise">2.1.5 - Polygons & Polyhedra</span>

   **2.2 - Coordinate Systems & Fields**

   .. toctree::
      :maxdepth: 1

      2.2.1 - Gradient <content/Module_02_geometry_mathematics/2.2_coordinate_systems_fields/2.2.1_gradient/README.rst>
      2.2.2 - Spiral <content/Module_02_geometry_mathematics/2.2_coordinate_systems_fields/2.2.2_spiral/README.rst>
      2.2.3 - Vector Fields <content/Module_02_geometry_mathematics/2.2_coordinate_systems_fields/2.2.3_vector_fields/README.rst>
      2.2.4 - Distance Fields <content/Module_02_geometry_mathematics/2.2_coordinate_systems_fields/2.2.4_distance_fields/README.rst>

   **2.3 - Mathematical Art**

   .. toctree::
      :maxdepth: 1

      2.3.2 - Rose Curves <content/Module_02_geometry_mathematics/2.3_mathematical_art/2.3.2_rose_curves/README.rst>
      2.3.3 - Harmonograph Simulation <content/Module_02_geometry_mathematics/2.3_mathematical_art/2.3.3_harmonograph_simulation/README.rst>

   .. raw:: html

      <span class="incomplete-exercise">2.3.1 - Lissajous Curves</span>
      <span class="incomplete-exercise">2.3.4 - Strange Attractors</span>

.. dropdown:: Module 3: Transformations & Effects


   Manipulating visual data through geometric transformations, masking, and artistic filters.

   **3.1 - Geometric Transformations**

   .. toctree::
      :maxdepth: 1

      3.1.1 - Rotation <content/Module_03_transformations_effects/3.1_geometric_transformations/3.1.1_rotation/README.rst>
      3.1.2 - Affine Transformations <content/Module_03_transformations_effects/3.1_geometric_transformations/3.1.2_affine_transformations/README.rst>
      3.1.3 - Nonlinear Distortions <content/Module_03_transformations_effects/3.1_geometric_transformations/3.1.3_nonlinear_distortions/README.rst>
      3.1.4 - Kaleidoscope Effects <content/Module_03_transformations_effects/3.1_geometric_transformations/3.1.4_kaleidoscope_effects/README.rst>

   **3.2 - Masking & Compositing**

   .. raw:: html

      <span class="incomplete-exercise">3.2.1 - Mask</span>
      <span class="incomplete-exercise">3.2.2 - Meme Generator</span>
      <span class="incomplete-exercise">3.2.3 - Shadow</span>
      <span class="incomplete-exercise">3.2.4 - Blend Modes</span>

   **3.3 - Artistic Filters**

   .. toctree::
      :maxdepth: 1

      3.3.1 - Warhol <content/Module_03_transformations_effects/3.3_artistic_filters/3.3.1_warhol/README.rst>
      3.3.3 - Hexpanda <content/Module_03_transformations_effects/3.3_artistic_filters/3.3.3_hexpanda/README.rst>
      3.3.5 - Delaunay Triangulation <content/Module_03_transformations_effects/3.3_artistic_filters/3.3.5_delaunay_triangulation/README.rst>

   .. raw:: html

      <span class="incomplete-exercise">3.3.2 - Puzzle (Array Concatenation)</span>
      <span class="incomplete-exercise">3.3.4 - Voronoi Diagrams</span>

   **3.4 - Signal Processing**

   .. toctree::
      :maxdepth: 1

      3.4.1 - Convolution <content/Module_03_transformations_effects/3.4_signal_processing/3.4.1_convolution/README.rst>

   .. raw:: html

      <span class="incomplete-exercise">3.4.2 - Edge Detection (Sobel Operator)</span>
      <span class="incomplete-exercise">3.4.3 - Contour Lines</span>
      <span class="incomplete-exercise">3.4.4 - Fourier Art</span>

.. dropdown:: Module 4: Fractals & Recursion


   Self-similarity and infinite complexity through classical fractals, natural patterns, and L-systems.

   **4.1 - Classical Fractals**

   .. toctree::
      :maxdepth: 1

      4.1.1 - Fractal Square <content/Module_04_fractals_recursion/4.1_classical_fractals/4.1.1_fractal_square/README.rst>
      4.1.2 - Dragon Curve <content/Module_04_fractals_recursion/4.1_classical_fractals/4.1.2_dragon_curve/README.rst>
      4.1.3 - Mandelbrot <content/Module_04_fractals_recursion/4.1_classical_fractals/4.1.3_mandelbrot/README.rst>

   .. raw:: html

      <span class="incomplete-exercise">4.1.4 - Julia Sets</span>
      <span class="incomplete-exercise">4.1.5 - Sierpinski</span>

   **4.2 - Natural Fractals**

   .. toctree::
      :maxdepth: 1

      4.2.1 - Fractal Trees <content/Module_04_fractals_recursion/4.2_natural_fractals/4.2.1_fractal_trees/README.rst>

   .. raw:: html

      <span class="incomplete-exercise">4.2.2 - Lightning Bolts</span>
      <span class="incomplete-exercise">4.2.3 - Fractal Landscapes</span>
      <span class="incomplete-exercise">4.2.4 - Diffusion Limited Aggregation</span>

   **4.3 - L-Systems**

   .. raw:: html

      <span class="incomplete-exercise">4.3.1 - Plant Generation</span>
      <span class="incomplete-exercise">4.3.2 - Koch Snowflake</span>
      <span class="incomplete-exercise">4.3.3 - Penrose Tiling</span>

.. dropdown:: Module 5: Simulation & Emergent Behavior


   Complex systems from simple rules: particle systems, flocking behavior, and physics simulations.

   **5.1 - Particle Systems**

   .. toctree::
      :maxdepth: 1

      5.1.1 - Sand <content/Module_05_simulation_emergent_behavior/5.1_particle_systems/5.1.1_sand/README.rst>

   .. raw:: html

      <span class="incomplete-exercise">5.1.2 - Vortex</span>
      <span class="incomplete-exercise">5.1.3 - Fireworks Simulation</span>
      <span class="incomplete-exercise">5.1.4 - Fluid Simulation</span>

   **5.2 - Flocking & Swarms**

   .. toctree::
      :maxdepth: 1

      5.2.1 - Boids <content/Module_05_simulation_emergent_behavior/5.2_flocking_swarms/5.2.1_boids/README.rst>

   .. raw:: html

      <span class="incomplete-exercise">5.2.2 - Fish Schooling</span>
      <span class="incomplete-exercise">5.2.3 - Ant Colony Optimization</span>

   **5.3 - Physics Simulations**

   .. toctree::
      :maxdepth: 1

      5.3.3 - Double Pendulum Chaos <content/Module_05_simulation_emergent_behavior/5.3_physics_simulations/5.3.3_double_pendulum_chaos/README.rst>

   .. raw:: html

      <span class="incomplete-exercise">5.3.1 - Bouncing Ball Animation</span>
      <span class="incomplete-exercise">5.3.2 - N-Body Planet Simulation</span>
      <span class="incomplete-exercise">5.3.4 - Cloth Rope Simulation</span>
      <span class="incomplete-exercise">5.3.5 - Magnetic Field Visualization</span>

   **5.4 - Growth & Morphogenesis**

   .. raw:: html

      <span class="incomplete-exercise">5.4.1 - Eden Growth Model</span>
      <span class="incomplete-exercise">5.4.2 - Differential Growth</span>
      <span class="incomplete-exercise">5.4.3 - Space Colonization Algorithm</span>
      <span class="incomplete-exercise">5.4.4 - Turing Patterns</span>

.. dropdown:: Module 6: Noise & Procedural Generation


   Controlled randomness for natural effects: noise functions, terrain, textures, and wave patterns.

   **6.1 - Noise Functions**

   .. toctree::
      :maxdepth: 1

      6.1.1 - Perlin Noise <content/Module_06_noise_procedural_generation/6.1_noise_functions/6.1.1_perlin_noise/README.rst>

   .. raw:: html

      <span class="incomplete-exercise">6.1.2 - Simplex Noise</span>
      <span class="incomplete-exercise">6.1.3 - Worley Noise</span>
      <span class="incomplete-exercise">6.1.4 - Colored Noise</span>

   **6.2 - Terrain Generation**

   .. raw:: html

      <span class="incomplete-exercise">6.2.1 - Height Maps</span>
      <span class="incomplete-exercise">6.2.2 - Erosion Simulation</span>
      <span class="incomplete-exercise">6.2.3 - Cave Generation</span>
      <span class="incomplete-exercise">6.2.4 - Island Generation</span>

   **6.3 - Texture Synthesis**

   .. raw:: html

      <span class="incomplete-exercise">6.3.1 - Marble Wood Textures</span>
      <span class="incomplete-exercise">6.3.2 - Cloud Generation</span>
      <span class="incomplete-exercise">6.3.3 - Abstract Patterns</span>
      <span class="incomplete-exercise">6.3.4 - Procedural Materials</span>

   **6.4 - Wave & Interference Patterns**

   .. raw:: html

      <span class="incomplete-exercise">6.4.1 - Moire Patterns</span>
      <span class="incomplete-exercise">6.4.2 - Wave Interference</span>
      <span class="incomplete-exercise">6.4.3 - Cymatics Visualization</span>

.. raw:: html

   <div class="module-section-header">ML & Animation</div>
   <div class="module-section-subtitle">Modules 7-9 · ~36 exercises · Machine learning and motion</div>

.. dropdown:: Module 7: Classical Machine Learning


   Traditional ML for creative applications: clustering, classification, and statistical methods.

   **7.1 - Clustering & Segmentation**

   .. raw:: html

      <span class="incomplete-exercise">7.1.1 - KMeans Clustering</span>
      <span class="incomplete-exercise">7.1.2 - Meanshift Segmentation</span>
      <span class="incomplete-exercise">7.1.3 - DBSCAN Pattern Detection</span>

   **7.2 - Classification & Recognition**

   .. raw:: html

      <span class="incomplete-exercise">7.2.1 - Decision Tree Classifier</span>
      <span class="incomplete-exercise">7.2.2 - Random Forests</span>
      <span class="incomplete-exercise">7.2.3 - SVM Style Detection</span>

   **7.3 - Dimensionality Reduction**

   .. raw:: html

      <span class="incomplete-exercise">7.3.1 - PCA Color Palette</span>
      <span class="incomplete-exercise">7.3.2 - t-SNE Visualization</span>
      <span class="incomplete-exercise">7.3.3 - UMAP Visualizations</span>

   **7.4 - Statistical Methods**

   .. raw:: html

      <span class="incomplete-exercise">7.4.1 - Monte Carlo Sampling</span>
      <span class="incomplete-exercise">7.4.2 - Markov Chains</span>
      <span class="incomplete-exercise">7.4.3 - Hidden Markov Models</span>

.. dropdown:: Module 8: Animation & Time


   Adding the fourth dimension: animation fundamentals, organic motion, and cinematic effects.

   **8.1 - Animation Fundamentals**

   .. raw:: html

      <span class="incomplete-exercise">8.1.1 - Image Transformations</span>
      <span class="incomplete-exercise">8.1.2 - Easing Functions</span>
      <span class="incomplete-exercise">8.1.3 - Interpolation Techniques</span>
      <span class="incomplete-exercise">8.1.4 - Sprite Sheets</span>

   **8.2 - Organic Motion**

   .. toctree::
      :maxdepth: 1

      8.2.2 - Infinite Blossom <content/Module_08_animation_time/8.2_organic_motion/8.2.2_infinite_blossom/README.rst>

   .. raw:: html

      <span class="incomplete-exercise">8.2.1 - Flower Assembly</span>
      <span class="incomplete-exercise">8.2.3 - Walk Cycles</span>
      <span class="incomplete-exercise">8.2.4 - Breathing Pulsing</span>

   **8.3 - Cinematic Effects**

   .. toctree::
      :maxdepth: 1

      8.3.1 - Star Wars Titles <content/Module_08_animation_time/8.3_cinematic_effects/8.3.1_starwars_titles/README.rst>
      8.3.2 - Thank You <content/Module_08_animation_time/8.3_cinematic_effects/8.3.2_thank_you/README.rst>

   .. raw:: html

      <span class="incomplete-exercise">8.3.3 - Particle Text Reveals</span>
      <span class="incomplete-exercise">8.3.4 - Morphing Transitions</span>

   **8.4 - Generative Animation**

   .. raw:: html

      <span class="incomplete-exercise">8.4.1 - Music Visualization</span>
      <span class="incomplete-exercise">8.4.3 - Animated Fractals</span>
      <span class="incomplete-exercise">8.4.2 - Data Driven Animation</span>

.. dropdown:: Module 9: Introduction to Neural Networks


   Bridge to modern AI: neural network fundamentals, architectures, and training dynamics.

   **9.1 - Neural Network Fundamentals**

   .. toctree::
      :maxdepth: 1

      9.1.1 - Perceptron Scratch <content/Module_09_intro_neural_networks/9.1_neural_network_fundamentals/9.1.1_perceptron_scratch/README.rst>
      9.1.2 - Backpropagation Visualization <content/Module_09_intro_neural_networks/9.1_neural_network_fundamentals/9.1.2_backpropagation_visualization/README.rst>
      9.1.3 - Activation Functions Art <content/Module_09_intro_neural_networks/9.1_neural_network_fundamentals/9.1.3_activation_functions_art/README.rst>

   **9.2 - Network Architectures**

   .. raw:: html

      <span class="incomplete-exercise">9.2.1 - Feedforward Networks</span>
      <span class="incomplete-exercise">9.2.2 - Convolutional Networks Visualization</span>
      <span class="incomplete-exercise">9.2.3 - Recurrent Networks for Sequences</span>

   **9.3 - Training Dynamics**

   .. raw:: html

      <span class="incomplete-exercise">9.3.1 - Loss Landscape Visualization</span>
      <span class="incomplete-exercise">9.3.2 - Gradient Descent Animation</span>
      <span class="incomplete-exercise">9.3.3 - Overfitting Underfitting Demos</span>

   **9.4 - Feature Visualization**

   .. raw:: html

      <span class="incomplete-exercise">9.4.1 - DeepDream Implementation</span>
      <span class="incomplete-exercise">9.4.2 - Feature Map Art</span>
      <span class="incomplete-exercise">9.4.3 - Network Attention Visualization</span>

.. raw:: html

   <div class="module-section-header">Real-Time & AI Integration</div>
   <div class="module-section-subtitle">Modules 10-13 · ~48 exercises · TouchDesigner and generative AI</div>

.. dropdown:: Module 10: TouchDesigner Fundamentals


   Real-time visual programming: TD environment, NumPy integration, and interactive controls.

   **10.1 - TD Environment & Workflow**

   .. toctree::
      :maxdepth: 1

      10.1.1 - Node Networks <content/Module_10_touchdesigner_fundamentals/10.1_td_environment_workflow/10.1.1_node_networks/README.rst>

   .. raw:: html

      <span class="incomplete-exercise">10.1.2 - Python Integration Basics</span>
      <span class="incomplete-exercise">10.1.3 - Performance Monitoring</span>

   **10.2 - Recreating Static Exercises**

   .. raw:: html

      <span class="incomplete-exercise">10.2.1 - Core Exercises Realtime</span>
      <span class="incomplete-exercise">10.2.2 - Boids Flocking in TouchDesigner</span>
      <span class="incomplete-exercise">10.2.3 - Planet Simulation TD</span>
      <span class="incomplete-exercise">10.2.4 - Fractals Realtime</span>

   **10.3 - NumPy to TD Pipeline**

   .. raw:: html

      <span class="incomplete-exercise">10.3.1 - Script Operators</span>
      <span class="incomplete-exercise">10.3.2 - Array Processing</span>
      <span class="incomplete-exercise">10.3.3 - Custom Components</span>

   **10.4 - Interactive Controls**

   .. raw:: html

      <span class="incomplete-exercise">10.4.1 - UI Building</span>
      <span class="incomplete-exercise">10.4.2 - Parameter Mapping</span>
      <span class="incomplete-exercise">10.4.3 - Preset Systems</span>

.. dropdown:: Module 11: Interactive Systems


   Sensors and real-time response: input devices, computer vision, and physical computing.

   **11.1 - Input Devices**

   .. raw:: html

      <span class="incomplete-exercise">11.1.1 - Webcam Processing</span>
      <span class="incomplete-exercise">11.1.2 - Audio Reactivity</span>
      <span class="incomplete-exercise">11.1.3 - MIDI OSC Control</span>
      <span class="incomplete-exercise">11.1.4 - Kinect Leap Motion</span>

   **11.2 - Computer Vision in TD**

   .. toctree::
      :maxdepth: 1

      11.2.3 - Face Detection <content/Module_11_interactive_systems/11.2_computer_vision_td/11.2.3_face_detection/README.rst>

   .. raw:: html

      <span class="incomplete-exercise">11.2.1 - Motion Detection</span>
      <span class="incomplete-exercise">11.2.2 - Blob Tracking</span>
      <span class="incomplete-exercise">11.2.4 - Optical Flow</span>

   **11.3 - Physical Computing**

   .. raw:: html

      <span class="incomplete-exercise">11.3.1 - Arduino Integration</span>
      <span class="incomplete-exercise">11.3.2 - DMX Lighting Control</span>
      <span class="incomplete-exercise">11.3.3 - Projection Mapping Basics</span>

   **11.4 - Network Communication**

   .. raw:: html

      <span class="incomplete-exercise">11.4.1 - Multi Machine Setups</span>
      <span class="incomplete-exercise">11.4.2 - WebSocket WebRTC</span>
      <span class="incomplete-exercise">11.4.3 - Remote Control Interfaces</span>

.. dropdown:: Module 12: Generative AI Models


   Modern generative techniques: GANs, VAEs, diffusion models, and language models for art.

   **12.1 - Generative Adversarial Networks**

   .. toctree::
      :maxdepth: 1

      12.1.1 - GAN Architecture <content/Module_12_generative_ai_models/12.1_generative_adversarial_networks/12.1.1_gan_architecture/README.rst>
      12.1.2 - DCGAN Art <content/Module_12_generative_ai_models/12.1_generative_adversarial_networks/12.1.2_dcgan_art/README.rst>
      12.1.3 - StyleGAN Exploration <content/Module_12_generative_ai_models/12.1_generative_adversarial_networks/12.1.3_stylegan_exploration/README.rst>
      12.1.4 - Pix2Pix Applications <content/Module_12_generative_ai_models/12.1_generative_adversarial_networks/12.1.4_pix2pix_applications/README.rst>

   **12.2 - Variational Autoencoders**

   .. toctree::
      :maxdepth: 1

      12.2.1 - Latent Space Exploration <content/Module_12_generative_ai_models/12.2_variational_autoencoders/12.2.1_latent_space_exploration/README.rst>
      12.2.2 - Interpolation Animations <content/Module_12_generative_ai_models/12.2_variational_autoencoders/12.2.2_interpolation_animations/README.rst>
      12.2.3 - Conditional VAEs <content/Module_12_generative_ai_models/12.2_variational_autoencoders/12.2.3_conditional_vaes/README.rst>

   **12.3 - Diffusion Models**

   .. toctree::
      :maxdepth: 1

      12.3.1 - DDPM Basics <content/Module_12_generative_ai_models/12.3_diffusion_models/12.3.1_ddpm_basics/README.rst>
      12.3.2 - ControlNet Guided Generation <content/Module_12_generative_ai_models/12.3_diffusion_models/12.3.2_controlnet_guided_generation/README.rst>

   **12.4 - Bridging Paradigms**

   .. raw:: html

      <span class="incomplete-exercise">12.4.1 - Neural Style Transfer</span>
      <span class="incomplete-exercise">12.4.2 - VQ-VAE and VQ-GAN</span>

   **12.5 - Personalization & Efficiency**

   .. toctree::
      :maxdepth: 1

      12.5.1 - DreamBooth Personalization <content/Module_12_generative_ai_models/12.5_personalization_efficiency/12.5.1_dreambooth_personalization/README.rst>

   **12.6 - Transformer Generation**

   .. raw:: html

      <span class="incomplete-exercise">12.6.1 - Taming Transformers</span>
      <span class="incomplete-exercise">12.6.2 - Diffusion Transformer (DiT)</span>

   **12.7 - Modern Frontiers**

   .. toctree::
      :maxdepth: 1

      12.7.1 - Flow Matching <content/Module_12_generative_ai_models/12.7_modern_frontiers/12.7.1_flow_matching/README.rst>

.. dropdown:: Module 13: AI + TouchDesigner Integration


   Combining AI with real-time systems: ML models in TD, real-time effects, and hybrid pipelines.

   **13.1 - ML Models in TD**

   .. raw:: html

      <span class="incomplete-exercise">13.1.1 - MediaPipe Integration</span>
      <span class="incomplete-exercise">13.1.2 - RunwayML Bridge</span>
      <span class="incomplete-exercise">13.1.3 - ONNX Runtime</span>

   **13.2 - Real-time AI Effects**

   .. raw:: html

      <span class="incomplete-exercise">13.2.1 - Style Transfer Live</span>
      <span class="incomplete-exercise">13.2.2 - Realtime Segmentation</span>
      <span class="incomplete-exercise">13.2.3 - Pose Driven Effects</span>

   **13.3 - Generative Models Live**

   .. raw:: html

      <span class="incomplete-exercise">13.3.1 - GAN Inference Optimization</span>
      <span class="incomplete-exercise">13.3.2 - Latent Space Navigation UI</span>
      <span class="incomplete-exercise">13.3.3 - Model Switching Systems</span>

   **13.4 - Hybrid Pipelines**

   .. raw:: html

      <span class="incomplete-exercise">13.4.1 - Preprocessing TD</span>
      <span class="incomplete-exercise">13.4.2 - Python ML Processing</span>
      <span class="incomplete-exercise">13.4.3 - Post Processing Chains</span>

.. raw:: html

   <div class="module-section-header">Data & Capstone</div>
   <div class="module-section-subtitle">Modules 14-15 · Final projects</div>

.. dropdown:: Module 14: Data as Material


   Information visualization and sonification: data sources, visualization techniques, and physical sculptures.

   **14.1 - Data Sources**

   .. raw:: html

      <span class="incomplete-exercise">14.1.1 - APIs and Data Scraping</span>
      <span class="incomplete-exercise">14.1.2 - Sensor Networks</span>
      <span class="incomplete-exercise">14.1.3 - Social Media Streams</span>
      <span class="incomplete-exercise">14.1.4 - Environmental Data</span>

   **14.2 - Visualization Techniques**

   .. raw:: html

      <span class="incomplete-exercise">14.2.1 - Network Graphs</span>
      <span class="incomplete-exercise">14.2.2 - Flow Visualization</span>
      <span class="incomplete-exercise">14.2.3 - Multidimensional Scaling</span>
      <span class="incomplete-exercise">14.2.4 - Time Series Art</span>

   **14.3 - Sonification**

   .. raw:: html

      <span class="incomplete-exercise">14.3.1 - Data Sound Mapping</span>
      <span class="incomplete-exercise">14.3.2 - Granular Synthesis</span>
      <span class="incomplete-exercise">14.3.3 - Rhythmic Patterns</span>

   **14.4 - Physical Data Sculptures**

   .. raw:: html

      <span class="incomplete-exercise">14.4.1 - 3D Printing Preparation</span>
      <span class="incomplete-exercise">14.4.2 - Laser Cutting Patterns</span>
      <span class="incomplete-exercise">14.4.3 - CNC Toolpaths</span>

.. dropdown:: Module 15: Capstone Project - Eternal Flow


   Synthesis of all learned concepts: StyleGAN-based evolving Ebru marbling artwork for projection display.

   .. toctree::
      :maxdepth: 1

      15 - Capstone Project <content/Module_15_capstone_project/README.rst>

.. raw:: html

   <div class="doc-footer">

.. rubric:: License

This work is licensed under the MIT License.

- Burak Kağan Yılmazer (2025) — burak.kagan@protonmail.com
- Dr. Kristian Rother (2024) — kristian.rother@posteo.de

See :download:`LICENSE` for full license terms.

.. rubric:: Built With

This curriculum is built with:

**Core Computing**: `NumPy <https://numpy.org/>`__, `SciPy <https://scipy.org/>`__, `pandas <https://pandas.pydata.org/>`__

**Image Processing**: `Pillow <https://pillow.readthedocs.io/>`__, `OpenCV <https://opencv.org/>`__, `ImageIO <https://imageio.readthedocs.io/>`__

**Machine Learning**: `scikit-learn <https://scikit-learn.org/>`__, `PyTorch <https://pytorch.org/>`__, `TensorFlow <https://tensorflow.org/>`__

**Visualization**: `matplotlib <https://matplotlib.org/>`__, `Jupyter <https://jupyter.org/>`__

**Real-time Systems**: `TouchDesigner <https://derivative.ca/>`__

For academic citations and detailed references, see individual module documentation.

.. rubric:: Inspiration

- `Generative Art: A Practical Guide <https://www.manning.com/books/generative-art>`__ by Matt Pearson
- `Processing: A Programming Handbook <https://processing.org/handbook/>`__ by Casey Reas & Ben Fry

.. raw:: html

   </div>