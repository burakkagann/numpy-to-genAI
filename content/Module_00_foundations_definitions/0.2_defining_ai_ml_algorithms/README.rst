.. _module-0-2-1-defining-ai-ml-algorithms:

=====================================
0.2.1 - Defining AI, ML, and Algorithms
=====================================

:Duration: 15-18 minutes
:Level: Beginner to Intermediate
:Prerequisites: Module 0.1.1 - What Is Generative Art

.. contents:: Contents
   :local:
   :depth: 2

Overview
========

Artificial Intelligence, Machine Learning and Algorithms is the core of modern generative art. Understanding these concepts and the relationships between them is essential for successfully navigating from traditional algorithmic art to AI driven art generation. In this module, you wiill learn the clear definitions for each term, discover how they relate to hierarchically and understand their unique roles in creative space. 

**Learning Objectives**

By completing this module, you will:

* Define Algorithm, Machine Learning (ML), and Artificial Intelligence (AI) in both technical and creative contexts.
* Understand the hierarchical relationship between these three concepts.
* Distinguish between traditional algorithmic generative art and AI generated art.
* Recognize how these technologies apply specifically to generative art practice.
* Choose the appropriate technology for different creative scenarios.


Quick Start: Identifying the Technology
=========================================

Let's begin by understanding which technology powers different creative systems. Consider these three art generation scenarios:

**Scenario A:** You write Python code using NumPy to create gradient patterns. The code explicitly specifies: "For each pixel at position x, set color value to x/width." The same code always produces the same output.

**Scenario B:** You train a neural network on 10,000 paintings. Then you feed it a new image, and it transforms that image to match the artistic style it learned. Different input images produce different styled outputs.

**Scenario C:** You type the prompt "a surreal landscape in the style of Salvador Dali" into Nano Banana. The AI model generates a completely new image that never existed before, combining learned concepts of "surreal," "landscape," and Dali's aesthetic.

.. admonition:: Quick Question
   
   Which scenario uses: (A) traditional algorithms, (B) machine learning, or (C) advanced AI? Can you identify the key distinguishing features?

**Answers:**

* **Scenario A** uses **traditional algorithms** which is explicit, deterministic instructions that produce predictable results.
* **Scenario B** uses **machine learning** where the system learned patterns from data and applies them to new inputs.
* **Scenario C** uses **generative AI** where the system creates novel content from high-level prompts.

.. tip::
   
   The key progression: **Algorithms** are explicit instructions. **Machine Learning** learns from data to make predictions. **Artificial Intelligence** contain systems that exhibit intelligent behavior including learning, reasoning, and creativity.

Understanding the Three Concepts
==================================

What is an Algorithm?
---------------------

**An algorithm is a finite sequence of well defined instructions for solving a problem or performing a computation.** Think of it as a recipe: a step by step procedure that, given specific inputs, produces specific outputs.

.. figure:: /content/Module_00_foundations_definitions/0.2_defining_ai_ml_algorithms/vera-molnar.png
   :width: 400px
   :align: center
   :alt: Vera Molnár - "(Des)Ordres" (1974)

   Vera Molnár - "(Des)Ordres" (1974)

**Core characteristics:**

1. **Clear and unambiguous**: Each step has only one interpretation.
2. **Finite**: Must complete in a finite number of steps.
3. **Well defined inputs/outputs**: What goes in and what comes out.
4. **Deterministic or pseudo random**: Same input → same output (or controlled randomness).

.. code-block:: text
   :caption: Algorithm Structure
   
   INPUT: Starting data or parameters
   PROCESS: Step by step instructions with clear logic
   OUTPUT: Resulting data or outcome

**In generative art**, algorithms define the rules and procedures that create visual, auditory, or interactive outputs. The artist programs explicit instructions, and the computer executes them precisely.

.. dropdown:: Algorithms in Context

   **Daily life examples:**
   
   * **Making coffee**: Boil water → Add grounds → Pour water → Wait → Enjoy
   * **GPS navigation**: Calculate shortest route based on traffic and destination
   
   **Algorithmic art example:**

   .. code-block:: python
      :caption: Run below script to view the simple algorithmic gradient 
      
      import numpy as np
      from PIL import Image
      
      width, height = 800, 600
      image = np.zeros((height, width, 3))
      
      for x in range(width):
          color_value = x / width  # Position determines color
          image[:, x, :] = color_value

       # Convert to PIL Image and save
       image_pil = Image.fromarray((image * 255).astype(np.uint8))
       image_pil.save("gradient_output.png")
      
      # Result: Smooth black-to-white gradient

      

   This is **deterministic** run it twice, get identical results. The artist explicitly programs every rule.

.. important::
   
   Algorithms pre-date computers by more than a millennia! The term comes from 9th century Persian mathematician Muhammad ibn Musa al-Khwarizmi.

What is Machine Learning (ML)?
-------------------------------

**Machine Learning is the field of study that gives computers the ability to learn from data without being explicitly programmed for every scenario.** Rather than following fixed instructions, ML systems discover patterns in data and use those patterns to make predictions or decisions about new, unseen data.

**Tom Mitchell's formal definition (1997):**

    "A program learns from experience **E** with respect to task **T** and performance measure **P**, if its performance at **T** improves with experience **E**."

In simpler terms: **ML systems get better at tasks by learning from examples** rather than following predetermined rules.

.. code-block:: text
   :caption: ML vs. Traditional Programming
   
   Traditional Programming:
   INPUT: Data + Explicit Rules → OUTPUT: Answers
   
   Machine Learning:
   INPUT: Data + Answers → OUTPUT: Learned Rules

**In generative art**, ML allows systems to learn aesthetic patterns from existing artworks and then generate new works that exhibit those learned patterns without the artist manually programming every action.

.. dropdown:: Deep Dive: Three Types of Machine Learning

   **1. Supervised Learning** (Learning with a teacher)
   
   * System learns from labeled data with known factual answers.
   * **Art example**: Style transfer train on labeled artistic styles and then apply to new images.
   
   **2. Unsupervised Learning** (Learning without teacher)
   
   * System finds patterns in data without being told what to look for.
   * **Art example**: Discovering visual motifs across large image datasets.
   
   **3. Reinforcement Learning** (Learning by trial and error)
   
   * System learns through rewards/penalties based on actions.
   * **Art example**: Training systems to generate aesthetically pleasing compositions through iterative feedback.

ML in generative art: A practical example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Traditional Algorithm Approach:**

.. code-block:: python
   :caption: Explicit flower instructions
   
   def draw_flower():
       # Artist must program every detail
       draw_circle(x, y, size, color)  # Step 1
       draw_circle(x + offset, y, size, color)  # Step 2
       # ... manually define all steps
       draw_circle(x, y, center_size, yellow)  # Last Step

   # Limited: Only creates exactly what you programmed

**Machine Learning Approach:**

.. code-block:: python
   :caption: ML-learned flower generation
   
   # Train on thousands of flower images
   model = train_on_flower_dataset(flower_images)
   
   # Generate new flowers
   new_flower = model.generate()
   
   # System learned what makes something "flower-like"
   # Can generate infinite variations

The ML system discovers what makes flowers recognizable (petals radiating from center, certain color patterns, organic shapes) without those rules being explicitly programmed.

What is Artificial Intelligence (AI)?
--------------------------------------

**Artificial Intelligence is technology that enables computers and machines to simulate human intelligence through learning, reasoning, problem-solving, perception, and decision making.** AI is the broadest concept which contains machine learning, algorithms, and other approaches to creating intelligent like behavior.

**Key capabilities that define AI:**

* **Learning**: Improving from experience (via machine learning)
* **Reasoning**: Drawing logical conclusions from information
* **Problem-solving**: Finding solutions to complex challenges
* **Creativity**: Generating novel outputs (especially relevant for art!)

.. code-block:: text
   :caption: Evolution from Algorithms to AI
   
   ALGORITHMS:        Follow explicit rules
                      ↓
   MACHINE LEARNING:  Learn rules from data
                      ↓
   ARTIFICIAL         Exhibit intelligent behavior:
   INTELLIGENCE:      Learn + Reason + Create + Adapt

**In generative art**, AI systems can autonomously create artistic outputs by learning patterns, making creative decisions, and generating novel content.

.. dropdown:: 📚 Historical Context: AI Evolution

   **Key milestones:**
   
   * **1950**: Alan Turing proposes the Turing Test
   * **1956**: John McCarthy coins "Artificial Intelligence" at Dartmouth Conference
   * **1997**: IBM's Deep Blue defeats chess champion Garry Kasparov
   * **2016**: DeepMind's AlphaGo defeats Go champion Lee Sedol
   * **2022**: ChatGPT demonstrates conversational AI capabilities
   * **2022-2023**: Midjourney, DALL-E 2, Stable Diffusion bring AI art to millions
   
   **In creative applications:**
   
   * **2015**: DeepDream—neural network visualization
   * **2017**: Neural Style Transfer applying artistic styles via deep learning
   * **2019**: GANs creating photorealistic faces (ThisPersonDoesNotExist)
   * **2022+**: Text-to-image AI generating any imaginable visual from prompts

The Hierarchical Relationship
==============================

Understanding how these three concepts relate is crucial:

.. figure:: /content/Module_00_foundations_definitions/0.2_defining_ai_ml_algorithms/ai-ml-algorithms.png
   :width: 600px
   :align: center
   :alt: Hierarchical diagram showing the relationship between AI, ML, and Algorithms

   The hierarchical relationship: Algorithms form the foundation, ML is a subset of AI, and AI is the broadest concept

The hierarchy explained
-----------------------

**Algorithms** = **The Foundation**

Every computational process uses algorithms. They're the fundamental building blocks. Whether you're sorting a list, training a neural network, or generating images with AI models make it possible.

**Machine Learning** = **Learning Algorithms**

ML is a specialized *type* of algorithm that learns from data rather than following predetermined rules. All ML uses algorithms, but not all algorithms involve machine learning.

**Artificial Intelligence** = **The Complete System**

AI is the broadest concept. It encompasses ML, traditional algorithms, and other approaches. An AI system might use ML for some tasks, rule based algorithms for others, and combine multiple techniques.

.. tip::
   
   **Think of it like cooking:**
   
   * **Algorithms** = Individual recipes (step-by-step instructions)
   * **Machine Learning** = Learning to cook by tasting many dishes and understanding patterns
   * **Artificial Intelligence** = A chef who creates new recipes, adapts to dietary restrictions, and combines different techniques creatively

Real-world example: Spam filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Traditional Algorithm:** Explicit rules programmed by humans (if "FREE MONEY" in subject → spam)

**Machine Learning:** System learns patterns from thousands of labeled examples (spam/not spam)

**Artificial Intelligence:** Comprehensive system combining ML patterns + rule-based checks + natural language understanding + adaptive learning from user feedback

This comprehensive approach exhibits "intelligence" by handling novel situations, adapting to changes, and improving over time.

Application to Generative Art
==============================

How these technologies evolved in creative practice
----------------------------------------------------

**Era 1: Algorithmic Art (1960s-2000s)**

Artists wrote explicit code defining every creative decision.

* **Technology:** Traditional algorithms
* **Characteristics:** Rules explicitly programmed, deterministic or pseudo-random, complete artistic control
* **Example:** Processing sketch using Perlin noise for flowing lines

.. code-block:: python
   :caption: Traditional algorithmic art
   
   # Explicit rules for creating a composition
   for i in range(100):
       x = random.randint(0, width)
       y = random.randint(0, height)
       size = random.randint(10, 50)
       draw_circle(x, y, size)

The artist defines every rule. Randomness is controlled. The aesthetic is embedded directly in the code.

**Era 2: ML-Enhanced Art (2010s)**

Systems learned patterns from existing artworks and applied them.

* **Technology:** Machine learning (primarily neural networks)
* **Characteristics:** Systems learn from data, semi-autonomous decisions, emergent behaviors
* **Key developments:** DeepDream (2015), Neural Style Transfer (2015), pix2pix (2017), StyleGAN (2018)

.. code-block:: python
   :caption: ML-enhanced art (style transfer)
   
   # System learns artistic styles from examples
   style_model = load_pretrained_model('neural_style_transfer')
   
   content_image = load_image('photo.jpg')
   style_reference = load_image('van_gogh.jpg')
   
   stylized_output = style_model.transfer(content_image, style_reference)

The artist doesn't program "how to paint like Van Gogh", instead the system learns from examples.

**Era 3: AI-Generated Art (2020s-Present)**

High level prompts generate entirely novel artworks.

* **Technology:** Advanced AI (GANs, Diffusion Models, Transformers)
* **Characteristics:** Natural language prompts, true generative capability, emergent creative behaviors
* **Platforms:** DALL-E (2021), Midjourney (2022), Stable Diffusion (2022), Adobe Firefly (2023), Nano Banana (2025)

.. code-block:: python
   :caption: AI art generation
   
   # High-level creative intent, not explicit rules
   prompt = "A surreal landscape with floating islands, golden hour lighting, digital art"
   
   generated_image = diffusion_model.generate(
       prompt=prompt,
       steps=50,
       guidance_scale=7.5
   )

The artist provides creative direction, but the AI interprets, combines concepts, and generates unique content.

Distinguishing the approaches
------------------------------

.. dropdown:: 📊 Detailed Comparison Table

   .. list-table:: Algorithmic Art vs. ML/AI Art
      :widths: 20 40 40
      :header-rows: 1
   
      * - Aspect
        - Algorithmic Generative Art
        - ML/AI Generative Art
      * - **Artist Control**
        - Direct -> every rule specified
        - Indirect -> guide through data/prompts
      * - **System Behavior**
        - Predictable within defined parameters
        - Can surprise artist with novel outputs
      * - **Required Expertise**
        - Programming, mathematics
        - Data curation, prompt engineering
      * - **Creative Role**
        - System designer & operator
        - Collaborator with AI system
      * - **Repeatability**
        - Same code = same result
        - Stochastic -> varies each generation
      * - **Learning Curve**
        - Steep initially, then precise control
        - Gentler entry, less direct control
      * - **Output Range**
        - Bounded by programmed rules
        - Potentially unbounded creativity

.. admonition:: Critical Distinction 🎯
   
   **Traditional generative art**: Artist creates the algorithm → System executes rules
   
   **AI-generated art**: Artist uses pre-trained AI model → Provides prompts/guidance
   
   Both are valid creative practices with different relationships between artist, tool, and outcome.

Summary
=======

In this module, we've established the conceptual foundation for understanding modern generative art technologies.

**Key takeaways:**

* **Algorithms** are step-by-step instructions. The foundation of all computation. In art, they provide direct control and explicit creative rules.
* **Machine Learning** systems learn from data rather than following predetermined rules. In art, they capture aesthetic patterns and apply them in new contexts.
* **Artificial Intelligence** is the broadest concept, encompassing ML, algorithms, and other approaches to intelligent behavior. In art, AI systems create novel content and exhibit creative autonomy.
* **These concepts are hierarchical**: Algorithms → ML (specialized algorithms that learn) → AI (comprehensive intelligent systems).
* **Technology evolution in art**: Traditional algorithms (1960s-2000s) → ML enhancement (2010s) → AI generation (2020s-present).
* **The tradeoff**: Algorithms offer more control, AI offers more creative autonomy and surprise.
* **Your course journey**: You'll work with all three approaches, learning when each is most appropriate.

.. tip::
   
   **The Right Tool for the Job:** Just as a painter chooses between oils, watercolors, or digital tools based on creative intent, you'll choose between algorithms, ML, and AI based on your needs for control, learning capability, and creative exploration.

Next Steps
==========

Now that you understand these foundational concepts, you're ready to begin creating:

* **Module 1** introduces algorithmic creation with pixel arrays and color manipulation
* **Modules 6-10** will teach you to work with machine learning for style transfer and feature learning
* **Modules 11-14** will explore advanced AI systems including GANs and Diffusion Models
* **Module 15** brings it all together in your capstone project where you'll choose the right technology for your creative vision

Continue to **Module 1: Pixel Fundamentals** to begin your hands-on journey from NumPy arrays to AI-powered generative art.

References
==========

.. [Mitchell1997] Mitchell, Tom M. "Machine Learning." McGraw-Hill, 1997.

.. [Academis2024] Academis.eu. "What is Machine Learning?" Machine Learning Fundamentals. https://www.academis.eu/machine_learning/fundamentals/what_is_ml/

.. [IBM2025] IBM Corporation. "What Is Artificial Intelligence (AI)?" IBM Topics. https://www.ibm.com/topics/artificial-intelligence

.. [Russell2020] Russell, Stuart, and Peter Norvig. "Artificial Intelligence: A Modern Approach." 4th edition. Pearson, 2020.

.. [Wikipedia2025a] Wikipedia contributors. "Artificial Intelligence." https://en.wikipedia.org/wiki/Artificial_intelligence

.. [Wikipedia2025b] Wikipedia contributors. "Algorithm." https://en.wikipedia.org/wiki/Algorithm

.. [Wikipedia2025c] Wikipedia contributors. "Machine Learning." https://en.wikipedia.org/wiki/Machine_learning

.. [Goodfellow2014] Goodfellow, Ian, et al. "Generative Adversarial Networks." *Advances in Neural Information Processing Systems* 27 (2014).

.. [Gatys2015] Gatys, Leon A., et al. "A Neural Algorithm of Artistic Style." *arXiv preprint* arXiv:1508.06576 (2015).

.. [Karras2019] Karras, Tero, et al. "A Style-Based Generator Architecture for GANs." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (2019).

.. [Ramesh2021] Ramesh, Aditya, et al. "Zero-Shot Text-to-Image Generation." *International Conference on Machine Learning* (2021).

.. [Rombach2022] Rombach, Robin, et al. "High-Resolution Image Synthesis with Latent Diffusion Models." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (2022).

.. [Boden2010] Boden, Margaret A. "Creativity and Art: Three Roads to Surprise." Oxford University Press, 2010.

.. [Hertzmann2018] Hertzmann, Aaron. "Can Computers Create Art?" *Arts* 7.2 (2018): 18.

.. [Shiffman2012] Shiffman, Daniel. "The Nature of Code: Simulating Natural Systems with Processing." 2012. https://natureofcode.com

.. [McCarthy2004] McCarthy, John. "What Is Artificial Intelligence?" Stanford University, 2004. http://jmc.stanford.edu/articles/whatisai/

.. [Turing1950] Turing, Alan M. "Computing Machinery and Intelligence." *Mind* 59.236 (1950): 433-460.