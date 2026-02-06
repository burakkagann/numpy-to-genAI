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

`https://github.com/burakkagann/numpy-to-genAI <https://github.com/burakkagann/numpy-to-genAI>`__

Clone the repository:

.. code-block:: bash

   git clone https://github.com/burakkagann/numpy-to-genAI.git
   cd numpy-to-genAI


Installation
------------

To execute the examples, you need to install the libraries in :download:`requirements.txt`.
Install them with:

:::

   pip install -r requirements.txt

If you are using the `Anaconda distribution <https://www.anaconda.com/>`__,
you should have all necessary libraries already.

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

   **1.2 - Pixel Manipulation Patterns**

   .. toctree::
      :maxdepth: 1

      1.2.1 - Random Patterns <content/Module_01_pixel_fundamentals/1.2_pixel_manipulation_patterns/1.2.1_random_patterns/README.rst>
      1.2.2 - Cellular Automata <content/Module_01_pixel_fundamentals/1.2_pixel_manipulation_patterns/1.2.2_cellular_automata/README.rst>

   **1.3 - Structured Compositions**

   .. toctree::
      :maxdepth: 1

      1.3.1 - Flags <content/Module_01_pixel_fundamentals/1.3_structured_compositions/1.3.1_flags/README.rst>
      1.3.2 - Repeat <content/Module_01_pixel_fundamentals/1.3_structured_compositions/1.3.2_repeat/README.rst>

.. dropdown:: Module 2: Geometry & Mathematics


   Mathematical foundations for generative art through shapes, coordinates, and mathematical patterns.

   **2.1 - Basic Shapes & Primitives**

   .. toctree::
      :maxdepth: 1

      2.1.1 - Lines <content/Module_02_geometry_mathematics/2.1_basic_shapes_primitives/2.1.1_lines/README.rst>
      2.1.2 - Triangles <content/Module_02_geometry_mathematics/2.1_basic_shapes_primitives/2.1.2_triangles/README.rst>
      2.1.3 - Circles <content/Module_02_geometry_mathematics/2.1_basic_shapes_primitives/2.1.3_circles/README.rst>
      2.1.4 - Stars <content/Module_02_geometry_mathematics/2.1_basic_shapes_primitives/2.1.4_stars/README.rst>

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

   .. toctree::
      :maxdepth: 1

      3.2.1 - Mask <content/Module_03_transformations_effects/3.2_masking_compositing/3.2.1_mask/README.rst>
      3.2.2 - Memegen <content/Module_03_transformations_effects/3.2_masking_compositing/3.2.2_memegen/README.rst>
      3.2.3 - Shadow <content/Module_03_transformations_effects/3.2_masking_compositing/3.2.3_shadow/README.rst>
      3.2.4 - Blend Modes <content/Module_03_transformations_effects/3.2_masking_compositing/3.2.4_blend_modes/README.rst>

   **3.3 - Artistic Filters**

   .. toctree::
      :maxdepth: 1

      3.3.1 - Warhol <content/Module_03_transformations_effects/3.3_artistic_filters/3.3.1_warhol/README.rst>
      3.3.2 - Puzzle <content/Module_03_transformations_effects/3.3_artistic_filters/3.3.2_puzzle/README.rst>
      3.3.3 - Hexpanda <content/Module_03_transformations_effects/3.3_artistic_filters/3.3.3_hexpanda/README.rst>
      3.3.4 - Voronoi Diagrams <content/Module_03_transformations_effects/3.3_artistic_filters/3.3.4_voronoi_diagrams/README.rst>
      3.3.5 - Delaunay Triangulation <content/Module_03_transformations_effects/3.3_artistic_filters/3.3.5_delaunay_triangulation/README.rst>

   **3.4 - Signal Processing**

   .. toctree::
      :maxdepth: 1

      3.4.1 - Convolution <content/Module_03_transformations_effects/3.4_signal_processing/3.4.1_convolution/README.rst>
      3.4.2 - Edge Detection <content/Module_03_transformations_effects/3.4_signal_processing/3.4.2_edge_detection/README.rst>
      3.4.3 - Contour Lines <content/Module_03_transformations_effects/3.4_signal_processing/3.4.3_contour_lines/README.rst>
      3.4.4 - Fourier Art <content/Module_03_transformations_effects/3.4_signal_processing/3.4.4_fourier_art/README.rst>

.. dropdown:: Module 4: Fractals & Recursion


   Self-similarity and infinite complexity through classical fractals, natural patterns, and L-systems.

   **4.1 - Classical Fractals**

   .. toctree::
      :maxdepth: 1

      4.1.1 - Fractal Square <content/Module_04_fractals_recursion/4.1_classical_fractals/4.1.1_fractal_square/README.rst>
      4.1.2 - Dragon Curve <content/Module_04_fractals_recursion/4.1_classical_fractals/4.1.2_dragon_curve/README.rst>
      4.1.3 - Mandelbrot <content/Module_04_fractals_recursion/4.1_classical_fractals/4.1.3_mandelbrot/README.rst>
      4.1.4 - Julia Sets <content/Module_04_fractals_recursion/4.1_classical_fractals/4.1.4_julia_sets/README.rst>
      4.1.5 - Sierpinski <content/Module_04_fractals_recursion/4.1_classical_fractals/4.1.5_sierpinski/README.rst>

   **4.2 - Natural Fractals**

   .. toctree::
      :maxdepth: 1

      4.2.1 - Fractal Trees <content/Module_04_fractals_recursion/4.2_natural_fractals/4.2.1_fractal_trees/README.rst>
      4.2.2 - Lightning Bolts <content/Module_04_fractals_recursion/4.2_natural_fractals/4.2.2_lightning_bolts/README.rst>
      4.2.3 - Fractal Landscapes <content/Module_04_fractals_recursion/4.2_natural_fractals/4.2.3_fractal_landscapes/README.rst>
      4.2.4 - Diffusion Limited Aggregation <content/Module_04_fractals_recursion/4.2_natural_fractals/4.2.4_diffusion_limited_aggregation/README.rst>

   **4.3 - L-Systems**

   .. toctree::
      :maxdepth: 1

      4.3.1 - Plant Generation <content/Module_04_fractals_recursion/4.3_l_systems/4.3.1_plant_generation/README.rst>
      4.3.2 - Koch Snowflake <content/Module_04_fractals_recursion/4.3_l_systems/4.3.2_koch_snowflake/README.rst>
      4.3.3 - Penrose Tiling <content/Module_04_fractals_recursion/4.3_l_systems/4.3.3_penrose_tiling/README.rst>

.. dropdown:: Module 5: Simulation & Emergent Behavior


   Complex systems from simple rules: particle systems, flocking behavior, and physics simulations.

   **5.1 - Particle Systems**

   .. toctree::
      :maxdepth: 1

      5.1.1 - Sand <content/Module_05_simulation_emergent_behavior/5.1_particle_systems/5.1.1_sand/README.rst>
      5.1.2 - Vortex <content/Module_05_simulation_emergent_behavior/5.1_particle_systems/5.1.2_vortex/README.rst>
      5.1.3 - Fireworks Simulation <content/Module_05_simulation_emergent_behavior/5.1_particle_systems/5.1.3_fireworks_simulation/README.rst>
      5.1.4 - Fluid Simulation <content/Module_05_simulation_emergent_behavior/5.1_particle_systems/5.1.4_fluid_simulation/README.rst>

   **5.2 - Flocking & Swarms**

   .. toctree::
      :maxdepth: 1

      5.2.1 - Boids <content/Module_05_simulation_emergent_behavior/5.2_flocking_swarms/5.2.1_boids/README.rst>
      5.2.2 - Fish Schooling <content/Module_05_simulation_emergent_behavior/5.2_flocking_swarms/5.2.2_fish_schooling/README.rst>
      5.2.3 - Ant Colony Optimization <content/Module_05_simulation_emergent_behavior/5.2_flocking_swarms/5.2.3_ant_colony_optimization/README.rst>

   **5.3 - Physics Simulations**

   .. toctree::
      :maxdepth: 1

      5.3.1 - Bouncing Ball <content/Module_05_simulation_emergent_behavior/5.3_physics_simulations/5.3.1_bouncing_ball/README.rst>
      5.3.2 - N-Body Planet Simulation <content/Module_05_simulation_emergent_behavior/5.3_physics_simulations/5.3.2_nbody_planet_simulation/README.rst>
      5.3.3 - Double Pendulum Chaos <content/Module_05_simulation_emergent_behavior/5.3_physics_simulations/5.3.3_double_pendulum_chaos/README.rst>
      5.3.4 - Cloth Rope Simulation <content/Module_05_simulation_emergent_behavior/5.3_physics_simulations/5.3.4_cloth_rope_simulation/README.rst>
      5.3.5 - Magnetic Field Visualization <content/Module_05_simulation_emergent_behavior/5.3_physics_simulations/5.3.5_magnetic_field_visualization/README.rst>

   **5.4 - Growth & Morphogenesis**

   .. toctree::
      :maxdepth: 1

      5.4.1 - Eden Growth Model <content/Module_05_simulation_emergent_behavior/5.4_growth_morphogenesis/5.4.1_eden_growth_model/README.rst>
      5.4.2 - Differential Growth <content/Module_05_simulation_emergent_behavior/5.4_growth_morphogenesis/5.4.2_differential_growth/README.rst>
      5.4.3 - Space Colonization Algorithm <content/Module_05_simulation_emergent_behavior/5.4_growth_morphogenesis/5.4.3_space_colonization_algorithm/README.rst>
      5.4.4 - Turing Patterns <content/Module_05_simulation_emergent_behavior/5.4_growth_morphogenesis/5.4.4_turing_patterns/README.rst>

.. dropdown:: Module 6: Noise & Procedural Generation


   Controlled randomness for natural effects: noise functions, terrain, textures, and wave patterns.

   **6.1 - Noise Functions**

   .. toctree::
      :maxdepth: 1

      6.1.1 - Perlin Noise <content/Module_06_noise_procedural_generation/6.1_noise_functions/6.1.1_perlin_noise/README.rst>
      6.1.2 - Simplex Noise <content/Module_06_noise_procedural_generation/6.1_noise_functions/6.1.2_simplex_noise/README.rst>
      6.1.3 - Worley Noise <content/Module_06_noise_procedural_generation/6.1_noise_functions/6.1.3_worley_noise/README.rst>
      6.1.4 - Colored Noise <content/Module_06_noise_procedural_generation/6.1_noise_functions/6.1.4_colored_noise/README.rst>

   **6.2 - Terrain Generation**

   .. toctree::
      :maxdepth: 1

      6.2.1 - Height Maps <content/Module_06_noise_procedural_generation/6.2_terrain_generation/6.2.1_height_maps/README.rst>
      6.2.2 - Erosion Simulation <content/Module_06_noise_procedural_generation/6.2_terrain_generation/6.2.2_erosion_simulation/README.rst>
      6.2.3 - Cave Generation <content/Module_06_noise_procedural_generation/6.2_terrain_generation/6.2.3_cave_generation/README.rst>
      6.2.4 - Island Generation <content/Module_06_noise_procedural_generation/6.2_terrain_generation/6.2.4_island_generation/README.rst>

   **6.3 - Texture Synthesis**

   .. toctree::
      :maxdepth: 1

      6.3.1 - Marble Wood Textures <content/Module_06_noise_procedural_generation/6.3_texture_synthesis/6.3.1_marble_wood_textures/README.rst>
      6.3.2 - Cloud Generation <content/Module_06_noise_procedural_generation/6.3_texture_synthesis/6.3.2_cloud_generation/README.rst>
      6.3.3 - Abstract Patterns <content/Module_06_noise_procedural_generation/6.3_texture_synthesis/6.3.3_abstract_patterns/README.rst>
      6.3.4 - Procedural Materials <content/Module_06_noise_procedural_generation/6.3_texture_synthesis/6.3.4_procedural_materials/README.rst>

   **6.4 - Wave & Interference Patterns**

   .. toctree::
      :maxdepth: 1

      6.4.1 - Moire Patterns <content/Module_06_noise_procedural_generation/6.4_wave_interference_patterns/6.4.1_moire_patterns/README.rst>
      6.4.2 - Wave Interference <content/Module_06_noise_procedural_generation/6.4_wave_interference_patterns/6.4.2_wave_interference/README.rst>
      6.4.3 - Cymatics Visualization <content/Module_06_noise_procedural_generation/6.4_wave_interference_patterns/6.4.3_cymatics_visualization/README.rst>

.. raw:: html

   <div class="module-section-header">ML & Animation</div>
   <div class="module-section-subtitle">Modules 7-9 · ~36 exercises · Machine learning and motion</div>

.. dropdown:: Module 7: Classical Machine Learning


   Traditional ML for creative applications: clustering, classification, and statistical methods.

   **7.1 - Clustering & Segmentation**

   .. toctree::
      :maxdepth: 1

      7.1.1 - KMeans Clustering <content/Module_07_classical_machine_learning/7.1_clustering_segmentation/7.1.1_kmeans_clustering/README.rst>
      7.1.2 - Meanshift Segmentation <content/Module_07_classical_machine_learning/7.1_clustering_segmentation/7.1.2_meanshift_segmentation/README.rst>
      7.1.3 - DBSCAN Pattern Detection <content/Module_07_classical_machine_learning/7.1_clustering_segmentation/7.1.3_dbscan_pattern_detection/README.rst>

   **7.2 - Classification & Recognition**

   .. toctree::
      :maxdepth: 1

      7.2.1 - Decision Tree Classifier <content/Module_07_classical_machine_learning/7.2_classification_recognition/7.2.1_decision_tree_classifier/README.rst>
      7.2.2 - Random Forests <content/Module_07_classical_machine_learning/7.2_classification_recognition/7.2.2_random_forests/README.rst>
      7.2.3 - SVM Style Detection <content/Module_07_classical_machine_learning/7.2_classification_recognition/7.2.3_svm_style_detection/README.rst>

   **7.3 - Dimensionality Reduction**

   .. toctree::
      :maxdepth: 1

      7.3.1 - PCA Color Palette <content/Module_07_classical_machine_learning/7.3_dimensionality_reduction/7.3.1_pca_color_palette/README.rst>
      7.3.2 - t-SNE Visualization <content/Module_07_classical_machine_learning/7.3_dimensionality_reduction/7.3.2_tsne_visualization/README.rst>
      7.3.3 - UMAP Visualizations <content/Module_07_classical_machine_learning/7.3_dimensionality_reduction/7.3.3_umap_visualizations/README.rst>

   **7.4 - Statistical Methods**

   .. toctree::
      :maxdepth: 1

      7.4.1 - Monte Carlo Sampling <content/Module_07_classical_machine_learning/7.4_statistical_methods/7.4.1_monte_carlo_sampling/README.rst>
      7.4.2 - Markov Chains <content/Module_07_classical_machine_learning/7.4_statistical_methods/7.4.2_markov_chains/README.rst>
      7.4.3 - Hidden Markov Models <content/Module_07_classical_machine_learning/7.4_statistical_methods/7.4.3_hidden_markov_models/README.rst>

.. dropdown:: Module 8: Animation & Time


   Adding the fourth dimension: animation fundamentals, organic motion, and cinematic effects.

   **8.1 - Animation Fundamentals**

   .. toctree::
      :maxdepth: 1

      8.1.1 - Image Transformations <content/Module_08_animation_time/8.1_animation_fundamentals/8.1.1_image_transformations/README.rst>
      8.1.2 - Easing Functions <content/Module_08_animation_time/8.1_animation_fundamentals/8.1.2_easing_functions/README.rst>
      8.1.3 - Interpolation Techniques <content/Module_08_animation_time/8.1_animation_fundamentals/8.1.3_interpolation_techniques/README.rst>
      8.1.4 - Sprite Sheets <content/Module_08_animation_time/8.1_animation_fundamentals/8.1.4_sprite_sheets/README.rst>

   **8.2 - Organic Motion**

   .. toctree::
      :maxdepth: 1

      8.2.1 - Flower Assembly <content/Module_08_animation_time/8.2_organic_motion/8.2.1_flower_assembly/README.rst>
      8.2.2 - Infinite Blossom <content/Module_08_animation_time/8.2_organic_motion/8.2.2_infinite_blossom/README.rst>
      8.2.3 - Walk Cycles <content/Module_08_animation_time/8.2_organic_motion/8.2.3_walk_cycles/README.rst>
      8.2.4 - Breathing Pulsing <content/Module_08_animation_time/8.2_organic_motion/8.2.4_breathing_pulsing/README.rst>

   **8.3 - Cinematic Effects**

   .. toctree::
      :maxdepth: 1

      8.3.1 - Star Wars Titles <content/Module_08_animation_time/8.3_cinematic_effects/8.3.1_starwars_titles/README.rst>
      8.3.2 - Thank You <content/Module_08_animation_time/8.3_cinematic_effects/8.3.2_thank_you/README.rst>
      8.3.3 - Particle Text Reveals <content/Module_08_animation_time/8.3_cinematic_effects/8.3.3_particle_text_reveals/README.rst>
      8.3.4 - Morphing Transitions <content/Module_08_animation_time/8.3_cinematic_effects/8.3.4_morphing_transitions/README.rst>

   **8.4 - Generative Animation**

   .. toctree::
      :maxdepth: 1

      8.4.1 - Music Visualization <content/Module_08_animation_time/8.4_generative_animation/8.4.1_music_visualization/README.rst>
      8.4.2 - Data Driven Animation <content/Module_08_animation_time/8.4_generative_animation/8.4.2_data_driven_animation/README.rst>
      8.4.3 - Animated Fractals <content/Module_08_animation_time/8.4_generative_animation/8.4.3_animated_fractals/README.rst>

.. dropdown:: Module 9: Introduction to Neural Networks


   Bridge to modern AI: neural network fundamentals, architectures, and training dynamics.

   **9.1 - Neural Network Fundamentals**

   .. toctree::
      :maxdepth: 1

      9.1.1 - Perceptron Scratch <content/Module_09_intro_neural_networks/9.1_neural_network_fundamentals/9.1.1_perceptron_scratch/README.rst>
      9.1.2 - Backpropagation Visualization <content/Module_09_intro_neural_networks/9.1_neural_network_fundamentals/9.1.2_backpropagation_visualization/README.rst>
      9.1.3 - Activation Functions Art <content/Module_09_intro_neural_networks/9.1_neural_network_fundamentals/9.1.3_activation_functions_art/README.rst>

   **9.2 - Network Architectures**

   .. toctree::
      :maxdepth: 1

      9.2.1 - Feedforward Networks <content/Module_09_intro_neural_networks/9.2_network_architectures/9.2.1_feedforward_networks/README.rst>
      9.2.2 - Convolutional Networks Visualization <content/Module_09_intro_neural_networks/9.2_network_architectures/9.2.2_convolutional_networks_visualization/README.rst>
      9.2.3 - Recurrent Networks Sequences <content/Module_09_intro_neural_networks/9.2_network_architectures/9.2.3_recurrent_networks_sequences/README.rst>

   **9.3 - Training Dynamics**

   .. toctree::
      :maxdepth: 1

      9.3.1 - Loss Landscape Visualization <content/Module_09_intro_neural_networks/9.3_training_dynamics/9.3.1_loss_landscape_visualization/README.rst>
      9.3.2 - Gradient Descent Animation <content/Module_09_intro_neural_networks/9.3_training_dynamics/9.3.2_gradient_descent_animation/README.rst>
      9.3.3 - Overfitting Underfitting Demos <content/Module_09_intro_neural_networks/9.3_training_dynamics/9.3.3_overfitting_underfitting_demos/README.rst>

   **9.4 - Feature Visualization**

   .. toctree::
      :maxdepth: 1

      9.4.1 - DeepDream Implementation <content/Module_09_intro_neural_networks/9.4_feature_visualization/9.4.1_deepdream_implementation/README.rst>
      9.4.2 - Feature Map Art <content/Module_09_intro_neural_networks/9.4_feature_visualization/9.4.2_feature_map_art/README.rst>
      9.4.3 - Network Attention Visualization <content/Module_09_intro_neural_networks/9.4_feature_visualization/9.4.3_network_attention_visualization/README.rst>

.. raw:: html

   <div class="module-section-header">Real-Time & AI Integration</div>
   <div class="module-section-subtitle">Modules 10-13 · ~48 exercises · TouchDesigner and generative AI</div>

.. dropdown:: Module 10: TouchDesigner Fundamentals


   Real-time visual programming: TD environment, NumPy integration, and interactive controls.

   **10.1 - TD Environment & Workflow**

   .. toctree::
      :maxdepth: 1

      10.1.1 - Node Networks <content/Module_10_touchdesigner_fundamentals/10.1_td_environment_workflow/10.1.1_node_networks/README.rst>
      10.1.2 - Python Integration Basics <content/Module_10_touchdesigner_fundamentals/10.1_td_environment_workflow/10.1.2_python_integration_basics/README.rst>
      10.1.3 - Performance Monitoring <content/Module_10_touchdesigner_fundamentals/10.1_td_environment_workflow/10.1.3_performance_monitoring/README.rst>

   **10.2 - Recreating Static Exercises**

   .. toctree::
      :maxdepth: 1

      10.2.1 - Core Exercises Realtime <content/Module_10_touchdesigner_fundamentals/10.2_recreating_static_exercises/10.2.1_core_exercises_realtime/README.rst>
      10.2.2 - Boids Flocking TD <content/Module_10_touchdesigner_fundamentals/10.2_recreating_static_exercises/10.2.2_boids_flocking_td/README.rst>
      10.2.3 - Planet Simulation TD <content/Module_10_touchdesigner_fundamentals/10.2_recreating_static_exercises/10.2.3_planet_simulation_td/README.rst>
      10.2.4 - Fractals Realtime <content/Module_10_touchdesigner_fundamentals/10.2_recreating_static_exercises/10.2.4_fractals_realtime/README.rst>

   **10.3 - NumPy to TD Pipeline**

   .. toctree::
      :maxdepth: 1

      10.3.1 - Script Operators <content/Module_10_touchdesigner_fundamentals/10.3_numpy_td_pipeline/10.3.1_script_operators/README.rst>
      10.3.2 - Array Processing <content/Module_10_touchdesigner_fundamentals/10.3_numpy_td_pipeline/10.3.2_array_processing/README.rst>
      10.3.3 - Custom Components <content/Module_10_touchdesigner_fundamentals/10.3_numpy_td_pipeline/10.3.3_custom_components/README.rst>

   **10.4 - Interactive Controls**

   .. toctree::
      :maxdepth: 1

      10.4.1 - UI Building <content/Module_10_touchdesigner_fundamentals/10.4_interactive_controls/10.4.1_ui_building/README.rst>
      10.4.2 - Parameter Mapping <content/Module_10_touchdesigner_fundamentals/10.4_interactive_controls/10.4.2_parameter_mapping/README.rst>
      10.4.3 - Preset Systems <content/Module_10_touchdesigner_fundamentals/10.4_interactive_controls/10.4.3_preset_systems/README.rst>

.. dropdown:: Module 11: Interactive Systems


   Sensors and real-time response: input devices, computer vision, and physical computing.

   **11.1 - Input Devices**

   .. toctree::
      :maxdepth: 1

      11.1.1 - Webcam Processing <content/Module_11_interactive_systems/11.1_input_devices/11.1.1_webcam_processing/README.rst>
      11.1.2 - Audio Reactivity <content/Module_11_interactive_systems/11.1_input_devices/11.1.2_audio_reactivity/README.rst>
      11.1.3 - MIDI OSC Control <content/Module_11_interactive_systems/11.1_input_devices/11.1.3_midi_osc_control/README.rst>
      11.1.4 - Kinect Leap Motion <content/Module_11_interactive_systems/11.1_input_devices/11.1.4_kinect_leap_motion/README.rst>

   **11.2 - Computer Vision in TD**

   .. toctree::
      :maxdepth: 1

      11.2.1 - Motion Detection <content/Module_11_interactive_systems/11.2_computer_vision_td/11.2.1_motion_detection/README.rst>
      11.2.2 - Blob Tracking <content/Module_11_interactive_systems/11.2_computer_vision_td/11.2.2_blob_tracking/README.rst>
      11.2.3 - Face Detection <content/Module_11_interactive_systems/11.2_computer_vision_td/11.2.3_face_detection/README.rst>
      11.2.4 - Optical Flow <content/Module_11_interactive_systems/11.2_computer_vision_td/11.2.4_optical_flow/README.rst>

   **11.3 - Physical Computing**

   .. toctree::
      :maxdepth: 1

      11.3.1 - Arduino Integration <content/Module_11_interactive_systems/11.3_physical_computing/11.3.1_arduino_integration/README.rst>
      11.3.2 - DMX Lighting Control <content/Module_11_interactive_systems/11.3_physical_computing/11.3.2_dmx_lighting_control/README.rst>
      11.3.3 - Projection Mapping Basics <content/Module_11_interactive_systems/11.3_physical_computing/11.3.3_projection_mapping_basics/README.rst>

   **11.4 - Network Communication**

   .. toctree::
      :maxdepth: 1

      11.4.1 - Multi Machine Setups <content/Module_11_interactive_systems/11.4_network_communication/11.4.1_multi_machine_setups/README.rst>
      11.4.2 - WebSocket WebRTC <content/Module_11_interactive_systems/11.4_network_communication/11.4.2_websocket_webrtc/README.rst>
      11.4.3 - Remote Control Interfaces <content/Module_11_interactive_systems/11.4_network_communication/11.4.3_remote_control_interfaces/README.rst>

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

   **12.4 - Language Models for Art**

   .. toctree::
      :maxdepth: 1

      12.4.1 - CLIP Guidance <content/Module_12_generative_ai_models/12.4_language_models_art/12.4.1_clip_guidance/README.rst>
      12.4.2 - Prompt Engineering <content/Module_12_generative_ai_models/12.4_language_models_art/12.4.2_prompt_engineering/README.rst>
      12.4.3 - Text to Image Pipelines <content/Module_12_generative_ai_models/12.4_language_models_art/12.4.3_text_to_image_pipelines/README.rst>

   **12.5 - Personalization & Efficiency**

   .. toctree::
      :maxdepth: 1

      12.5.1 - DreamBooth Personalization <content/Module_12_generative_ai_models/12.5_personalization_efficiency/12.5.1_dreambooth_personalization/README.rst>

.. dropdown:: Module 13: AI + TouchDesigner Integration


   Combining AI with real-time systems: ML models in TD, real-time effects, and hybrid pipelines.

   **13.1 - ML Models in TD**

   .. toctree::
      :maxdepth: 1

      13.1.1 - MediaPipe Integration <content/Module_13_ai_touchdesigner_integration/13.1_ml_models_td/13.1.1_mediapipe_integration/README.rst>
      13.1.2 - RunwayML Bridge <content/Module_13_ai_touchdesigner_integration/13.1_ml_models_td/13.1.2_runwayml_bridge/README.rst>
      13.1.3 - ONNX Runtime <content/Module_13_ai_touchdesigner_integration/13.1_ml_models_td/13.1.3_onnx_runtime/README.rst>

   **13.2 - Real-time AI Effects**

   .. toctree::
      :maxdepth: 1

      13.2.1 - Style Transfer Live <content/Module_13_ai_touchdesigner_integration/13.2_realtime_ai_effects/13.2.1_style_transfer_live/README.rst>
      13.2.2 - Realtime Segmentation <content/Module_13_ai_touchdesigner_integration/13.2_realtime_ai_effects/13.2.2_realtime_segmentation/README.rst>
      13.2.3 - Pose Driven Effects <content/Module_13_ai_touchdesigner_integration/13.2_realtime_ai_effects/13.2.3_pose_driven_effects/README.rst>

   **13.3 - Generative Models Live**

   .. toctree::
      :maxdepth: 1

      13.3.1 - GAN Inference Optimization <content/Module_13_ai_touchdesigner_integration/13.3_generative_models_live/13.3.1_gan_inference_optimization/README.rst>
      13.3.2 - Latent Space Navigation UI <content/Module_13_ai_touchdesigner_integration/13.3_generative_models_live/13.3.2_latent_space_navigation_ui/README.rst>
      13.3.3 - Model Switching Systems <content/Module_13_ai_touchdesigner_integration/13.3_generative_models_live/13.3.3_model_switching_systems/README.rst>

   **13.4 - Hybrid Pipelines**

   .. toctree::
      :maxdepth: 1

      13.4.1 - Preprocessing TD <content/Module_13_ai_touchdesigner_integration/13.4_hybrid_pipelines/13.4.1_preprocessing_td/README.rst>
      13.4.2 - Python ML Processing <content/Module_13_ai_touchdesigner_integration/13.4_hybrid_pipelines/13.4.2_python_ml_processing/README.rst>
      13.4.3 - Post Processing Chains <content/Module_13_ai_touchdesigner_integration/13.4_hybrid_pipelines/13.4.3_post_processing_chains/README.rst>

.. raw:: html

   <div class="module-section-header">Data & Capstone</div>
   <div class="module-section-subtitle">Modules 14-15 · Final projects</div>

.. dropdown:: Module 14: Data as Material


   Information visualization and sonification: data sources, visualization techniques, and physical sculptures.

   **14.1 - Data Sources**

   .. toctree::
      :maxdepth: 1

      14.1.1 - APIs Scraping <content/Module_14_data_as_material/14.1_data_sources/14.1.1_apis_scraping/README.rst>
      14.1.2 - Sensor Networks <content/Module_14_data_as_material/14.1_data_sources/14.1.2_sensor_networks/README.rst>
      14.1.3 - Social Media Streams <content/Module_14_data_as_material/14.1_data_sources/14.1.3_social_media_streams/README.rst>
      14.1.4 - Environmental Data <content/Module_14_data_as_material/14.1_data_sources/14.1.4_environmental_data/README.rst>

   **14.2 - Visualization Techniques**

   .. toctree::
      :maxdepth: 1

      14.2.1 - Network Graphs <content/Module_14_data_as_material/14.2_visualization_techniques/14.2.1_network_graphs/README.rst>
      14.2.2 - Flow Visualization <content/Module_14_data_as_material/14.2_visualization_techniques/14.2.2_flow_visualization/README.rst>
      14.2.3 - Multidimensional Scaling <content/Module_14_data_as_material/14.2_visualization_techniques/14.2.3_multidimensional_scaling/README.rst>
      14.2.4 - Time Series Art <content/Module_14_data_as_material/14.2_visualization_techniques/14.2.4_time_series_art/README.rst>

   **14.3 - Sonification**

   .. toctree::
      :maxdepth: 1

      14.3.1 - Data Sound Mapping <content/Module_14_data_as_material/14.3_sonification/14.3.1_data_sound_mapping/README.rst>
      14.3.2 - Granular Synthesis <content/Module_14_data_as_material/14.3_sonification/14.3.2_granular_synthesis/README.rst>
      14.3.3 - Rhythmic Patterns <content/Module_14_data_as_material/14.3_sonification/14.3.3_rhythmic_patterns/README.rst>

   **14.4 - Physical Data Sculptures**

   .. toctree::
      :maxdepth: 1

      14.4.1 - 3D Printing Preparation <content/Module_14_data_as_material/14.4_physical_data_sculptures/14.4.1_3d_printing_preparation/README.rst>
      14.4.2 - Laser Cutting Patterns <content/Module_14_data_as_material/14.4_physical_data_sculptures/14.4.2_laser_cutting_patterns/README.rst>
      14.4.3 - CNC Toolpaths <content/Module_14_data_as_material/14.4_physical_data_sculptures/14.4.3_cnc_toolpaths/README.rst>

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