:hide-toc:

Pixels to GenAI
===============

An open source educational platform that creates a comprehensive pathway into AI-driven generative art, bridging mathematical and visual foundations to modern creative AI techniques. This 15-module curriculum takes learners from fundamental pixel manipulation and NumPy operations through advanced generative models, neural networks, and real-time interactive systems.

**paint things – create art – have fun!**

.. figure:: /content/Module_03_transformations_effects/3.1_geometric_transformations/3.1.4_kaleidoscope_effects/kaleidoscope/mandala_pattern.png
   :width: 400px
   :align: center
   :alt: Kaleidoscope mandala pattern generated using NumPy transformations

   *Mandala pattern from Module 3.1.4: Kaleidoscope Effects*


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

.. dropdown:: Module 0: Foundations & Definitions

   Setting the conceptual and technical groundwork for generative art and AI.

   .. toctree::
      :maxdepth: 1

      content/Module_00_foundations_definitions/0.1_what_is_generative_art/README.rst
      content/Module_00_foundations_definitions/0.2_defining_ai_ml_algorithms/README.rst
      content/Module_00_foundations_definitions/0.4_setup_environment/README.rst

.. dropdown:: Module 1: Pixel Fundamentals

   Understanding images at the atomic level through color theory and manipulation patterns.

   .. toctree::
      :maxdepth: 1

      content/Module_01_pixel_fundamentals/1.1_grayscale_color_basics/1.1.1_color_basics/rgb/README.rst
      content/Module_01_pixel_fundamentals/1.2_pixel_manipulation_patterns/1.2.1_random_patterns/random_tiles/README.rst
      content/Module_01_pixel_fundamentals/1.2_pixel_manipulation_patterns/1.2.2_cellular_automata/README.rst
      content/Module_01_pixel_fundamentals/1.3_structured_compositions/1.3.1_flags/flags/README.rst
      content/Module_01_pixel_fundamentals/1.3_structured_compositions/1.3.2_repeat/repeat/README.rst

.. dropdown:: Module 2: Geometry & Mathematics

   Mathematical foundations for generative art through shapes, coordinates, and mathematical patterns.

   .. toctree::
      :maxdepth: 1

      content/Module_02_geometry_mathematics/2.1_basic_shapes_primitives/2.1.1_lines/lines/README.rst
      content/Module_02_geometry_mathematics/2.1_basic_shapes_primitives/2.1.2_triangles/triangles/README.rst
      content/Module_02_geometry_mathematics/2.1_basic_shapes_primitives/2.1.3_circles/circles/README.rst
      content/Module_02_geometry_mathematics/2.1_basic_shapes_primitives/2.1.4_stars/stars/README.rst
      content/Module_02_geometry_mathematics/2.2_coordinate_systems_fields/2.2.1_gradient/gradient/README.rst
      content/Module_02_geometry_mathematics/2.2_coordinate_systems_fields/2.2.2_spiral/spiral/README.rst
      content/Module_02_geometry_mathematics/2.2_coordinate_systems_fields/2.2.3_vector_fields/vector_fields/README.rst
      content/Module_02_geometry_mathematics/2.2_coordinate_systems_fields/2.2.4_distance_fields/README.rst
      content/Module_02_geometry_mathematics/2.3_mathematical_art/2.3.2_rose_curves/README.rst
      content/Module_02_geometry_mathematics/2.3_mathematical_art/2.3.3_harmonograph_simulation/README.rst

.. dropdown:: Module 3: Transformations & Effects

   Manipulating visual data through geometric transformations, masking, and artistic filters.

   .. toctree::
      :maxdepth: 1

      content/Module_03_transformations_effects/3.1_geometric_transformations/3.1.1_rotation/rotate/README.rst
      content/Module_03_transformations_effects/3.1_geometric_transformations/3.1.2_affine_transformations/README.rst
      content/Module_03_transformations_effects/3.1_geometric_transformations/3.1.3_nonlinear_distortions/README.rst
      content/Module_03_transformations_effects/3.1_geometric_transformations/3.1.4_kaleidoscope_effects/kaleidoscope/README.rst
      content/Module_03_transformations_effects/3.2_masking_compositing/3.2.1_mask/mask/README.rst
      content/Module_03_transformations_effects/3.2_masking_compositing/3.2.2_memegen/memegen/README.rst
      content/Module_03_transformations_effects/3.2_masking_compositing/3.2.3_shadow/shadow/README.rst
      content/Module_03_transformations_effects/3.2_masking_compositing/3.2.4_blend_modes/README.rst
      content/Module_03_transformations_effects/3.3_artistic_filters/3.3.1_warhol/warhol/README.rst
      content/Module_03_transformations_effects/3.3_artistic_filters/3.3.2_puzzle/puzzle/README.rst
      content/Module_03_transformations_effects/3.3_artistic_filters/3.3.3_hexpanda/hexpanda/README.rst
      content/Module_03_transformations_effects/3.3_artistic_filters/3.3.4_voronoi_diagrams/README.rst
      content/Module_03_transformations_effects/3.3_artistic_filters/3.3.5_delaunay_triangulation/README.rst
      content/Module_03_transformations_effects/3.4_signal_processing/3.4.1_convolution/convolution/README.rst
      content/Module_03_transformations_effects/3.4_signal_processing/3.4.2_edge_detection/sobel/README.rst
      content/Module_03_transformations_effects/3.4_signal_processing/3.4.3_contour_lines/contour/README.rst
      content/Module_03_transformations_effects/3.4_signal_processing/3.4.4_fourier_art/README.rst

.. dropdown:: Module 4: Fractals & Recursion

   Self-similarity and infinite complexity through classical fractals, natural patterns, and L-systems.

   .. toctree::
      :maxdepth: 1

      content/Module_04_fractals_recursion/4.1_classical_fractals/4.1.1_fractal_square/fractal_square/README.rst
      content/Module_04_fractals_recursion/4.1_classical_fractals/4.1.2_dragon_curve/dragon_curve/README.rst
      content/Module_04_fractals_recursion/4.1_classical_fractals/4.1.3_mandelbrot/mandelbrot/README.rst
      content/Module_04_fractals_recursion/4.1_classical_fractals/4.1.4_julia_sets/README.rst
      content/Module_04_fractals_recursion/4.1_classical_fractals/4.1.5_sierpinski/README.rst
      content/Module_04_fractals_recursion/4.2_natural_fractals/4.2.1_fractal_trees/README.rst
      content/Module_04_fractals_recursion/4.2_natural_fractals/4.2.2_lightning_bolts/README.rst
      content/Module_04_fractals_recursion/4.2_natural_fractals/4.2.3_fractal_landscapes/README.rst
      content/Module_04_fractals_recursion/4.2_natural_fractals/4.2.4_diffusion_limited_aggregation/README.rst
      content/Module_04_fractals_recursion/4.3_l_systems/4.3.1_plant_generation/README.rst
      content/Module_04_fractals_recursion/4.3_l_systems/4.3.2_koch_snowflake/README.rst
      content/Module_04_fractals_recursion/4.3_l_systems/4.3.3_penrose_tiling/README.rst

.. dropdown:: Module 5: Simulation & Emergent Behavior

   Complex systems from simple rules: particle systems, flocking behavior, and physics simulations.

   .. toctree::
      :maxdepth: 1

      content/Module_05_simulation_emergent_behavior/5.1_particle_systems/5.1.1_sand/sand/README.rst
      content/Module_05_simulation_emergent_behavior/5.1_particle_systems/5.1.2_vortex/vortex/README.rst
      content/Module_05_simulation_emergent_behavior/5.1_particle_systems/5.1.3_fireworks_simulation/README.rst
      content/Module_05_simulation_emergent_behavior/5.1_particle_systems/5.1.4_fluid_simulation/README.rst
      content/Module_05_simulation_emergent_behavior/5.2_flocking_swarms/5.2.1_boids/README.rst
      content/Module_05_simulation_emergent_behavior/5.2_flocking_swarms/5.2.2_fish_schooling/README.rst
      content/Module_05_simulation_emergent_behavior/5.2_flocking_swarms/5.2.3_ant_colony_optimization/README.rst
      content/Module_05_simulation_emergent_behavior/5.3_physics_simulations/5.3.1_bouncing_ball/bouncing_ball/README.rst
      content/Module_05_simulation_emergent_behavior/5.3_physics_simulations/5.3.2_nbody_planet_simulation/README.rst
      content/Module_05_simulation_emergent_behavior/5.3_physics_simulations/5.3.3_double_pendulum_chaos/README.rst
      content/Module_05_simulation_emergent_behavior/5.3_physics_simulations/5.3.4_cloth_rope_simulation/README.rst
      content/Module_05_simulation_emergent_behavior/5.3_physics_simulations/5.3.5_magnetic_field_visualization/README.rst
      content/Module_05_simulation_emergent_behavior/5.4_growth_morphogenesis/5.4.1_eden_growth_model/README.rst
      content/Module_05_simulation_emergent_behavior/5.4_growth_morphogenesis/5.4.2_differential_growth/README.rst
      content/Module_05_simulation_emergent_behavior/5.4_growth_morphogenesis/5.4.3_space_colonization_algorithm/README.rst
      content/Module_05_simulation_emergent_behavior/5.4_growth_morphogenesis/5.4.4_turing_patterns/README.rst

.. dropdown:: Module 6: Noise & Procedural Generation

   Controlled randomness for natural effects: noise functions, terrain, textures, and wave patterns.

   .. toctree::
      :maxdepth: 1

      content/Module_06_noise_procedural_generation/6.1_noise_functions/6.1.1_perlin_noise/README.rst
      content/Module_06_noise_procedural_generation/6.1_noise_functions/6.1.2_simplex_noise/README.rst
      content/Module_06_noise_procedural_generation/6.1_noise_functions/6.1.3_worley_noise/README.rst
      content/Module_06_noise_procedural_generation/6.1_noise_functions/6.1.4_colored_noise/README.rst
      content/Module_06_noise_procedural_generation/6.2_terrain_generation/6.2.1_height_maps/README.rst
      content/Module_06_noise_procedural_generation/6.2_terrain_generation/6.2.2_erosion_simulation/README.rst
      content/Module_06_noise_procedural_generation/6.2_terrain_generation/6.2.3_cave_generation/README.rst
      content/Module_06_noise_procedural_generation/6.2_terrain_generation/6.2.4_island_generation/README.rst
      content/Module_06_noise_procedural_generation/6.3_texture_synthesis/6.3.1_marble_wood_textures/README.rst
      content/Module_06_noise_procedural_generation/6.3_texture_synthesis/6.3.2_cloud_generation/README.rst
      content/Module_06_noise_procedural_generation/6.3_texture_synthesis/6.3.3_abstract_patterns/README.rst
      content/Module_06_noise_procedural_generation/6.3_texture_synthesis/6.3.4_procedural_materials/README.rst
      content/Module_06_noise_procedural_generation/6.4_wave_interference_patterns/6.4.1_moire_patterns/README.rst
      content/Module_06_noise_procedural_generation/6.4_wave_interference_patterns/6.4.2_wave_interference/README.rst
      content/Module_06_noise_procedural_generation/6.4_wave_interference_patterns/6.4.3_cymatics_visualization/README.rst

.. dropdown:: Module 7: Classical Machine Learning

   Traditional ML for creative applications: clustering, classification, and statistical methods.

   .. toctree::
      :maxdepth: 1

      content/Module_07_classical_machine_learning/7.1_clustering_segmentation/7.1.1_kmeans_clustering/kmeans/README.rst
      content/Module_07_classical_machine_learning/7.1_clustering_segmentation/7.1.2_meanshift_segmentation/README.rst
      content/Module_07_classical_machine_learning/7.1_clustering_segmentation/7.1.3_dbscan_pattern_detection/README.rst
      content/Module_07_classical_machine_learning/7.2_classification_recognition/7.2.1_decision_tree_classifier/dtree/README.rst
      content/Module_07_classical_machine_learning/7.2_classification_recognition/7.2.2_random_forests/README.rst
      content/Module_07_classical_machine_learning/7.2_classification_recognition/7.2.3_svm_style_detection/README.rst
      content/Module_07_classical_machine_learning/7.3_dimensionality_reduction/7.3.1_pca_color_palette/README.rst
      content/Module_07_classical_machine_learning/7.3_dimensionality_reduction/7.3.2_tsne_visualization/README.rst
      content/Module_07_classical_machine_learning/7.3_dimensionality_reduction/7.3.3_umap_visualizations/README.rst
      content/Module_07_classical_machine_learning/7.4_statistical_methods/7.4.1_monte_carlo_sampling/montecarlo/README.rst
      content/Module_07_classical_machine_learning/7.4_statistical_methods/7.4.2_markov_chains/README.rst
      content/Module_07_classical_machine_learning/7.4_statistical_methods/7.4.3_hidden_markov_models/README.rst

.. dropdown:: Module 8: Animation & Time

   Adding the fourth dimension: animation fundamentals, organic motion, and cinematic effects.

   .. toctree::
      :maxdepth: 1

      content/Module_08_animation_time/8.1_animation_fundamentals/8.1.1_image_transformations/image_transformations/README.rst
      content/Module_08_animation_time/8.1_animation_fundamentals/8.1.2_easing_functions/README.rst
      content/Module_08_animation_time/8.1_animation_fundamentals/8.1.3_interpolation_techniques/README.rst
      content/Module_08_animation_time/8.1_animation_fundamentals/8.1.4_sprite_sheets/README.rst
      content/Module_08_animation_time/8.2_organic_motion/8.2.1_flower_assembly/flower_movie/README.rst
      content/Module_08_animation_time/8.2_organic_motion/8.2.2_infinite_blossom/blossom/README.rst
      content/Module_08_animation_time/8.2_organic_motion/8.2.3_walk_cycles/README.rst
      content/Module_08_animation_time/8.2_organic_motion/8.2.4_breathing_pulsing/README.rst
      content/Module_08_animation_time/8.3_cinematic_effects/8.3.1_starwars_titles/starwars/README.rst
      content/Module_08_animation_time/8.3_cinematic_effects/8.3.2_thank_you/thank_you/README.rst
      content/Module_08_animation_time/8.3_cinematic_effects/8.3.3_particle_text_reveals/README.rst
      content/Module_08_animation_time/8.3_cinematic_effects/8.3.4_morphing_transitions/README.rst
      content/Module_08_animation_time/8.4_generative_animation/8.4.1_music_visualization/README.rst
      content/Module_08_animation_time/8.4_generative_animation/8.4.2_data_driven_animation/README.rst
      content/Module_08_animation_time/8.4_generative_animation/8.4.3_animated_fractals/README.rst

.. dropdown:: Module 9: Introduction to Neural Networks

   Bridge to modern AI: neural network fundamentals, architectures, and training dynamics.

   .. toctree::
      :maxdepth: 1

      content/Module_09_intro_neural_networks/9.1_neural_network_fundamentals/9.1.1_perceptron_scratch/README.rst
      content/Module_09_intro_neural_networks/9.1_neural_network_fundamentals/9.1.2_backpropagation_visualization/README.rst
      content/Module_09_intro_neural_networks/9.1_neural_network_fundamentals/9.1.3_activation_functions_art/README.rst
      content/Module_09_intro_neural_networks/9.2_network_architectures/9.2.1_feedforward_networks/README.rst
      content/Module_09_intro_neural_networks/9.2_network_architectures/9.2.2_convolutional_networks_visualization/README.rst
      content/Module_09_intro_neural_networks/9.2_network_architectures/9.2.3_recurrent_networks_sequences/README.rst
      content/Module_09_intro_neural_networks/9.3_training_dynamics/9.3.1_loss_landscape_visualization/README.rst
      content/Module_09_intro_neural_networks/9.3_training_dynamics/9.3.2_gradient_descent_animation/README.rst
      content/Module_09_intro_neural_networks/9.3_training_dynamics/9.3.3_overfitting_underfitting_demos/README.rst
      content/Module_09_intro_neural_networks/9.4_feature_visualization/9.4.1_deepdream_implementation/README.rst
      content/Module_09_intro_neural_networks/9.4_feature_visualization/9.4.2_feature_map_art/README.rst
      content/Module_09_intro_neural_networks/9.4_feature_visualization/9.4.3_network_attention_visualization/README.rst

.. dropdown:: Module 10: TouchDesigner Fundamentals

   Real-time visual programming: TD environment, NumPy integration, and interactive controls.

   .. toctree::
      :maxdepth: 1

      content/Module_10_touchdesigner_fundamentals/10.1_td_environment_workflow/10.1.1_node_networks/README.rst
      content/Module_10_touchdesigner_fundamentals/10.1_td_environment_workflow/10.1.2_python_integration_basics/README.rst
      content/Module_10_touchdesigner_fundamentals/10.1_td_environment_workflow/10.1.3_performance_monitoring/README.rst
      content/Module_10_touchdesigner_fundamentals/10.2_recreating_static_exercises/10.2.1_core_exercises_realtime/README.rst
      content/Module_10_touchdesigner_fundamentals/10.2_recreating_static_exercises/10.2.2_boids_flocking_td/README.rst
      content/Module_10_touchdesigner_fundamentals/10.2_recreating_static_exercises/10.2.3_planet_simulation_td/README.rst
      content/Module_10_touchdesigner_fundamentals/10.2_recreating_static_exercises/10.2.4_fractals_realtime/README.rst
      content/Module_10_touchdesigner_fundamentals/10.3_numpy_td_pipeline/10.3.1_script_operators/README.rst
      content/Module_10_touchdesigner_fundamentals/10.3_numpy_td_pipeline/10.3.2_array_processing/README.rst
      content/Module_10_touchdesigner_fundamentals/10.3_numpy_td_pipeline/10.3.3_custom_components/README.rst
      content/Module_10_touchdesigner_fundamentals/10.4_interactive_controls/10.4.1_ui_building/README.rst
      content/Module_10_touchdesigner_fundamentals/10.4_interactive_controls/10.4.2_parameter_mapping/README.rst
      content/Module_10_touchdesigner_fundamentals/10.4_interactive_controls/10.4.3_preset_systems/README.rst

.. dropdown:: Module 11: Interactive Systems

   Sensors and real-time response: input devices, computer vision, and physical computing.

   .. toctree::
      :maxdepth: 1

      content/Module_11_interactive_systems/11.1_input_devices/11.1.1_webcam_processing/README.rst
      content/Module_11_interactive_systems/11.1_input_devices/11.1.2_audio_reactivity/README.rst
      content/Module_11_interactive_systems/11.1_input_devices/11.1.3_midi_osc_control/README.rst
      content/Module_11_interactive_systems/11.1_input_devices/11.1.4_kinect_leap_motion/README.rst
      content/Module_11_interactive_systems/11.2_computer_vision_td/11.2.1_motion_detection/README.rst
      content/Module_11_interactive_systems/11.2_computer_vision_td/11.2.2_blob_tracking/README.rst
      content/Module_11_interactive_systems/11.2_computer_vision_td/11.2.3_face_detection/README.rst
      content/Module_11_interactive_systems/11.2_computer_vision_td/11.2.4_optical_flow/README.rst
      content/Module_11_interactive_systems/11.3_physical_computing/11.3.1_arduino_integration/README.rst
      content/Module_11_interactive_systems/11.3_physical_computing/11.3.2_dmx_lighting_control/README.rst
      content/Module_11_interactive_systems/11.3_physical_computing/11.3.3_projection_mapping_basics/README.rst
      content/Module_11_interactive_systems/11.4_network_communication/11.4.1_multi_machine_setups/README.rst
      content/Module_11_interactive_systems/11.4_network_communication/11.4.2_websocket_webrtc/README.rst
      content/Module_11_interactive_systems/11.4_network_communication/11.4.3_remote_control_interfaces/README.rst

.. dropdown:: Module 12: Generative AI Models

   Modern generative techniques: GANs, VAEs, diffusion models, and language models for art.

   .. toctree::
      :maxdepth: 1

      content/Module_12_generative_ai_models/12.1_generative_adversarial_networks/12.1.1_gan_architecture/README.rst
      content/Module_12_generative_ai_models/12.1_generative_adversarial_networks/12.1.2_dcgan_art/README.rst
      content/Module_12_generative_ai_models/12.1_generative_adversarial_networks/12.1.3_stylegan_exploration/README.rst
      content/Module_12_generative_ai_models/12.1_generative_adversarial_networks/12.1.4_pix2pix_applications/README.rst
      content/Module_12_generative_ai_models/12.2_variational_autoencoders/12.2.1_latent_space_exploration/README.rst
      content/Module_12_generative_ai_models/12.2_variational_autoencoders/12.2.2_interpolation_animations/README.rst
      content/Module_12_generative_ai_models/12.2_variational_autoencoders/12.2.3_conditional_vaes/README.rst
      content/Module_12_generative_ai_models/12.3_diffusion_models/12.3.1_ddpm_basics/README.rst
      content/Module_12_generative_ai_models/12.3_diffusion_models/12.3.2_stable_diffusion_integration/README.rst
      content/Module_12_generative_ai_models/12.3_diffusion_models/12.3.3_controlnet_guided_generation/README.rst
      content/Module_12_generative_ai_models/12.4_language_models_art/12.4.1_clip_guidance/README.rst
      content/Module_12_generative_ai_models/12.4_language_models_art/12.4.2_prompt_engineering/README.rst
      content/Module_12_generative_ai_models/12.4_language_models_art/12.4.3_text_to_image_pipelines/README.rst

.. dropdown:: Module 13: AI + TouchDesigner Integration

   Combining AI with real-time systems: ML models in TD, real-time effects, and hybrid pipelines.

   .. toctree::
      :maxdepth: 1

      content/Module_13_ai_touchdesigner_integration/13.1_ml_models_td/13.1.1_mediapipe_integration/README.rst
      content/Module_13_ai_touchdesigner_integration/13.1_ml_models_td/13.1.2_runwayml_bridge/README.rst
      content/Module_13_ai_touchdesigner_integration/13.1_ml_models_td/13.1.3_onnx_runtime/README.rst
      content/Module_13_ai_touchdesigner_integration/13.2_realtime_ai_effects/13.2.1_style_transfer_live/README.rst
      content/Module_13_ai_touchdesigner_integration/13.2_realtime_ai_effects/13.2.2_realtime_segmentation/README.rst
      content/Module_13_ai_touchdesigner_integration/13.2_realtime_ai_effects/13.2.3_pose_driven_effects/README.rst
      content/Module_13_ai_touchdesigner_integration/13.3_generative_models_live/13.3.1_gan_inference_optimization/README.rst
      content/Module_13_ai_touchdesigner_integration/13.3_generative_models_live/13.3.2_latent_space_navigation_ui/README.rst
      content/Module_13_ai_touchdesigner_integration/13.3_generative_models_live/13.3.3_model_switching_systems/README.rst
      content/Module_13_ai_touchdesigner_integration/13.4_hybrid_pipelines/13.4.1_preprocessing_td/README.rst
      content/Module_13_ai_touchdesigner_integration/13.4_hybrid_pipelines/13.4.2_python_ml_processing/README.rst
      content/Module_13_ai_touchdesigner_integration/13.4_hybrid_pipelines/13.4.3_post_processing_chains/README.rst

.. dropdown:: Module 14: Data as Material

   Information visualization and sonification: data sources, visualization techniques, and physical sculptures.

   .. toctree::
      :maxdepth: 1

      content/Module_14_data_as_material/14.1_data_sources/14.1.1_apis_scraping/README.rst
      content/Module_14_data_as_material/14.1_data_sources/14.1.2_sensor_networks/README.rst
      content/Module_14_data_as_material/14.1_data_sources/14.1.3_social_media_streams/README.rst
      content/Module_14_data_as_material/14.1_data_sources/14.1.4_environmental_data/README.rst
      content/Module_14_data_as_material/14.2_visualization_techniques/14.2.1_network_graphs/README.rst
      content/Module_14_data_as_material/14.2_visualization_techniques/14.2.2_flow_visualization/README.rst
      content/Module_14_data_as_material/14.2_visualization_techniques/14.2.3_multidimensional_scaling/README.rst
      content/Module_14_data_as_material/14.2_visualization_techniques/14.2.4_time_series_art/README.rst
      content/Module_14_data_as_material/14.3_sonification/14.3.1_data_sound_mapping/README.rst
      content/Module_14_data_as_material/14.3_sonification/14.3.2_granular_synthesis/README.rst
      content/Module_14_data_as_material/14.3_sonification/14.3.3_rhythmic_patterns/README.rst
      content/Module_14_data_as_material/14.4_physical_data_sculptures/14.4.1_3d_printing_preparation/README.rst
      content/Module_14_data_as_material/14.4_physical_data_sculptures/14.4.2_laser_cutting_patterns/README.rst
      content/Module_14_data_as_material/14.4_physical_data_sculptures/14.4.3_cnc_toolpaths/README.rst


.. raw:: html

   <div class="doc-footer">

.. rubric:: License

This work is licensed under the MIT License.

- Burak Kağan Yılmazer (2025) — burak.kagan@protonmail.com
- Dr. Kristian Rother (2024) — kristian.rother@posteo.de

See :download:`LICENSE` for full license terms.

.. rubric:: References

.. rubric:: Images

- `Brandenburg Gate image <https://commons.wikimedia.org/wiki/File:Brandenburger_Tor_abends.jpg>`__ by Thomas Wolf, www.foto-tw.de / Wikimedia Commons / CC BY-SA 3.0

.. rubric:: Libraries

**Core Numerical Computing & Data Science**

- NumPy — Harris, C.R., et al. (2020). Array programming with NumPy. Nature 585, 357–362
- SciPy — Virtanen, P., et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods 17, 261–272
- pandas — McKinney, W. (2010). Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference, 56-61

**Image Processing & Computer Vision**

- Pillow — Clark, A. (2015). Pillow (PIL Fork) Documentation
- OpenCV — Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools
- ImageIO — Klein, A., et al. (2019). ImageIO: a Python library for reading and writing image data. Zenodo. 

**Machine Learning & AI**

- scikit-learn — Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research 12, 2825-2830
- PyTorch — Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems 32, 8024-8035
- TensorFlow — Abadi, M., et al. (2016). TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems.

**Visualization & Interactive Computing**

- matplotlib — Hunter, J.D. (2007). Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering 9(3), 90-95
- Jupyter — Kluyver, T., et al. (2016). Jupyter Notebooks—a publishing format for reproducible computational workflows. Positioning and Power in Academic Publishing, 87-90

.. rubric:: Inspiration

- Generative Art — Pearson, M. (2011). Generative Art: A Practical Guide Using Processing
- Creative Coding — Reas, C. & Fry, B. (2014). Processing: A Programming Handbook for Visual Designers

.. raw:: html

   </div>