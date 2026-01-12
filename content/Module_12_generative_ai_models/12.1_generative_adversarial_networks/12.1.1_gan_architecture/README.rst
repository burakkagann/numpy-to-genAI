.. _module-12-1-1-gan-architecture:

================================
12.1.1 GAN Architecture
================================

:Duration: 35-40 minutes
:Level: Intermediate

Overview
========

Generative Adversarial Networks (GANs) represent one of the most influential breakthroughs in generative artificial intelligence. Introduced by Ian Goodfellow and colleagues in 2014 [Goodfellow2014]_, GANs use a novel adversarial training approach where two neural networks compete against each other: a **Generator** that creates synthetic data and a **Discriminator** that tries to distinguish real data from fake.

This exercise introduces the fundamental architecture of GANs through a hands-on implementation. We start with a minimal GAN that learns to generate numbers from a target distribution, then bridge to simple visual patterns. This approach lets you focus on understanding the core adversarial training loop before moving to image generation in DCGAN (Module 12.1.2).

Learning Objectives
-------------------

By the end of this exercise, you will be able to:

* Explain the generator-discriminator architecture of GANs
* Understand the adversarial training process as a two-player game
* Implement a basic GAN using PyTorch
* Observe how a GAN learns to match a target distribution
* Visualize and interpret GAN training progression


Quick Start: See It In Action
=============================

Run this code to see a minimal Generator (the first half of a GAN):

.. code-block:: python
   :caption: Minimal Generator example

   import torch
   import torch.nn as nn

   # Target: Generate numbers from N(4.0, 1.25)
   DATA_MEAN = 4.0
   DATA_STDDEV = 1.25

   # Simple 3-layer Generator
   class Generator(nn.Module):
       def __init__(self):
           super().__init__()
           self.layer1 = nn.Linear(1, 5)
           self.layer2 = nn.Linear(5, 5)
           self.layer3 = nn.Linear(5, 1)

       def forward(self, x):
           x = torch.tanh(self.layer1(x))
           x = torch.tanh(self.layer2(x))
           return self.layer3(x)

   # Create generator and sample
   generator = Generator()
   noise = torch.rand(100, 1)  # Random input
   output = generator(noise)
   print(f"Generated {len(output)} numbers")
   print(f"Mean: {output.mean():.2f}, Std: {output.std():.2f}")

**Expected output:**

.. code-block:: text

   Generated 100 numbers
   Mean: 0.01, Std: 0.02

The output shows random values near zero because the Generator is **untrained**. The weights are initialized randomly, so it produces arbitrary numbers.

The goal of GAN training is to make the Generator produce numbers matching N(4.0, 1.25). To achieve this, we need a Discriminator to provide feedback. Run the full training script in Exercise 1 to see this in action.


Core Concepts
=============

Concept 1: What Are GANs?
-------------------------

A **Generative Adversarial Network** consists of two neural networks that are trained simultaneously in a competitive process [Goodfellow2014]_. The architecture includes:

**The Generator (G)**

The Generator takes random noise as input and transforms it into synthetic data. Think of it as a counterfeiter trying to create convincing fake currency:

* **Input**: Random noise vector (latent space)
* **Output**: Synthetic data sample (numbers, images, etc.)
* **Goal**: Create outputs that fool the Discriminator

**The Discriminator (D)**

The Discriminator examines data samples and predicts whether they are real (from the training set) or fake (from the Generator). It acts as a detective:

* **Input**: Data sample (real or fake)
* **Output**: Probability that the input is real (0 to 1)
* **Goal**: Correctly classify real vs. fake samples

.. figure:: gan_architecture_diagram.png
   :width: 600px
   :align: center
   :alt: Diagram showing the GAN architecture with Generator transforming noise into fake data and Discriminator classifying real vs fake

   GAN architecture: Generator creates fake data from noise; Discriminator classifies real vs fake.

The key insight of GANs is that these two networks improve each other through competition. As the Discriminator gets better at detecting fakes, the Generator must produce more realistic outputs to succeed. This adversarial dynamic drives both networks toward increasingly sophisticated behavior.

.. admonition:: Did You Know?

   GANs have generated remarkably realistic human faces that do not belong to any real person. The website "This Person Does Not Exist" (thispersondoesnotexist.com) uses StyleGAN to create photorealistic portraits of fictional people [Karras2019]_. Large-scale GANs like BigGAN can generate high-resolution images across thousands of object categories [Brock2019]_.


Concept 2: The Adversarial Game
-------------------------------

GAN training can be understood through game theory as a **minimax game** between two players [Goodfellow2016]_. Each network has an opposing objective:

**The Discriminator's Objective**

The Discriminator wants to maximize its classification accuracy:

* Output high probability (close to 1) for real data
* Output low probability (close to 0) for fake data

**The Generator's Objective**

The Generator wants to minimize the Discriminator's success:

* Create outputs that receive high probability scores from the Discriminator
* In effect, maximize the Discriminator's errors

This creates a **zero-sum game**: what helps one player hurts the other. The mathematical formulation is:

.. math::

   \min_G \max_D \; \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]

In practice, we train each network in alternating steps:

.. code-block:: python
   :caption: Simplified GAN training loop
   :linenos:

   for epoch in range(num_epochs):
       # Sample real data and generate fake data
       real_data = get_real_batch()
       fake_data = generator(random_noise())

       # Step 1: Train Discriminator
       d_loss_real = criterion(discriminator(real_data), ones)
       d_loss_fake = criterion(discriminator(fake_data), zeros)
       d_loss = d_loss_real + d_loss_fake
       d_loss.backward()

       # Step 2: Train Generator
       fake_data = generator(random_noise())
       g_loss = criterion(discriminator(fake_data), ones)
       g_loss.backward()

* **Lines 6-10**: The Discriminator is trained to output 1 for real data and 0 for fake data
* **Lines 13-15**: The Generator is trained to make the Discriminator output 1 for fake data

When training succeeds, the networks reach a **Nash equilibrium** where the Generator produces data indistinguishable from real data, and the Discriminator can only guess randomly (50% accuracy) [Goodfellow2014]_.


Concept 3: Training Dynamics
----------------------------

GAN training is notoriously challenging due to the delicate balance required between the two networks [Salimans2016]_. Understanding common issues helps you diagnose and improve training.

**Monitoring Loss Curves**

During training, we track both the Generator loss and Discriminator loss:

Key observations:

* **Early training**: Discriminator loss drops quickly (easy to detect bad fakes)
* **Mid training**: Generator loss decreases as it learns to fool the Discriminator
* **Convergence**: Both losses stabilize, often with slight oscillation

**Mode Collapse**

A common failure mode where the Generator produces only a limited variety of outputs [Arjovsky2017]_. Signs include:

* Generated samples all look similar
* Generator loss stays low but outputs lack diversity
* Discriminator loss increases as it only sees one type of fake

**Training Tips**

Several techniques improve GAN training stability [Radford2016]_:

1. **Use Batch Normalization** in the Generator (not in the Discriminator's final layer)
2. **Use Leaky ReLU** activations instead of standard ReLU
3. **Balance learning rates** between Generator and Discriminator
4. **Use Adam optimizer** with beta1=0.5 for both networks [PyTorchDocs]_

.. important::

   If the Discriminator becomes too strong too quickly, the Generator receives unhelpful gradients and cannot improve. Techniques like alternating training steps (e.g., train G twice per D training) can help maintain balance.


Hands-On Exercises
==================

Exercise 1: Observe and Understand
----------------------------------

Run the GAN training script to watch a Generator learn to match a target distribution. The script trains a GAN to generate numbers from a Gaussian distribution N(4.0, 1.25).

.. code-block:: bash
   :caption: Run the training script

   python gan_architecture.py

:download:`Download gan_architecture.py <gan_architecture.py>`

.. figure:: gan_training_results.png
   :width: 600px
   :align: center
   :alt: Left plot shows histogram of generated numbers matching target Gaussian distribution. Right plot shows training loss curves.

   Training results: generated distribution (left) and loss curves (right).

**What to observe:**

1. **Training progress**: Watch the Generator's mean and standard deviation approach the target values
2. **Loss dynamics**: Notice how Discriminator and Generator losses interact
3. **Final output**: The histogram should match the target Gaussian curve

**Reflection questions:**

1. How many epochs does it take for the Generator to produce reasonable numbers?
2. What is the final mean and standard deviation of the generated samples?
3. How do the loss curves behave? Do they converge or oscillate?

.. dropdown:: Answers and Explanation
   :class-title: sd-font-weight-bold

   1. **Convergence**: The Generator typically starts producing reasonable numbers (mean close to 4.0) within the first 1000-2000 epochs. Full convergence takes around 4000-5000 epochs.

   2. **Final statistics**: After training, the generated samples should have mean around 3.9-4.1 and standard deviation around 1.2-1.3, close to the target N(4.0, 1.25).

   3. **Loss behavior**: Initially, the Discriminator loss drops as it easily distinguishes random Generator outputs from real data. As training progresses, both losses tend to oscillate and eventually stabilize. Some oscillation is normal and indicates the adversarial process is working.


Exercise 2: Modify Parameters
-----------------------------

Experiment with the GAN by changing key parameters in ``gan_architecture.py``. Observe how each change affects training.

**Goal 1**: Change the target distribution

.. code-block:: python
   :caption: Try different target distributions

   # Original
   DATA_MEAN = 4.0
   DATA_STDDEV = 1.25

   # Try: Different center
   DATA_MEAN = 0.0
   DATA_STDDEV = 1.25

   # Try: Wider distribution
   DATA_MEAN = 4.0
   DATA_STDDEV = 3.0

**Goal 2**: Adjust learning rates

.. code-block:: python
   :caption: Experiment with learning rate

   LEARNING_RATE = 1e-3    # Default
   LEARNING_RATE = 1e-2    # Faster (may be unstable)
   LEARNING_RATE = 1e-4    # Slower (more stable but takes longer)

**Goal 3**: Change training steps ratio

.. code-block:: python
   :caption: Adjust Discriminator/Generator training balance

   # Default: Equal training
   D_STEPS = 20
   G_STEPS = 20

   # More Discriminator training
   D_STEPS = 30
   G_STEPS = 10

   # More Generator training
   D_STEPS = 10
   G_STEPS = 30

.. dropdown:: Solution: Parameter Effects
   :class-title: sd-font-weight-bold

   **Target distribution**: The GAN should be able to learn any Gaussian distribution. Wider distributions (larger stddev) may take longer to learn because the Generator must cover more range.

   **Learning rate**: Higher rates train faster but may overshoot and become unstable. Lower rates are more stable but require more epochs.

   **Training steps ratio**: If the Discriminator trains too much relative to the Generator, it becomes too good and the Generator receives vanishing gradients. If the Generator trains too much, it may find shortcuts that exploit a weak Discriminator. Balance is key.


Exercise 3: Bridge to Visual Patterns
-------------------------------------

Now that you understand how GANs learn distributions, run the visual pattern script to see the same concepts applied to simple images:

.. code-block:: bash
   :caption: Run the pattern generation script

   python gan_patterns.py

:download:`Download gan_patterns.py <gan_patterns.py>`

This script trains a GAN to generate simple 8x8 pixel patterns (horizontal and vertical lines). It uses the same adversarial training loop, but now the Generator produces image data instead of numbers.

**What to observe:**

1. **Real vs Generated**: Compare the real patterns (top row) with generated patterns (bottom rows)
2. **Pattern quality**: How well does the Generator learn to create line patterns?
3. **Training time**: Small images train quickly, preparing you for DCGAN with larger images

.. dropdown:: Understanding the Transition
   :class-title: sd-font-weight-bold

   The key difference between number generation and image generation:

   * **Numbers**: Generator outputs 1 value, Discriminator uses statistical moments to evaluate
   * **Images**: Generator outputs N values (pixels), Discriminator evaluates the entire flattened image

   The core adversarial training loop remains the same. This is why understanding GANs with simple examples helps before moving to complex image models.

.. figure:: gan_patterns_output.png
   :width: 600px
   :align: center
   :alt: Comparison of real 8x8 patterns (top row) versus GAN-generated patterns (bottom rows)

   Real patterns (top) vs generated patterns (bottom).

.. note:: Source Attribution

   The exercises in this module are based on `"GANs in 50 lines of code" by Dev Nag <https://github.com/devnag/pytorch-generative-adversarial-networks>`_ (Apache-2.0 License). The implementation has been adapted with additional comments and simplified variable names for educational purposes.


Summary
=======

Key Takeaways
-------------

* **GANs** consist of two networks: a Generator that creates synthetic data and a Discriminator that distinguishes real from fake
* Training is an **adversarial game** where both networks improve through competition
* The Generator transforms random noise (latent vector) into data samples
* The Discriminator outputs a probability that its input is real
* **Loss curves** help monitor training progress and detect issues like mode collapse
* **Hyperparameter balance** (learning rates, training steps) is critical for stable training

Common Pitfalls
---------------

* **Mode collapse**: Generator produces limited variety. Try reducing Generator learning rate or adding noise
* **Vanishing gradients**: Discriminator becomes too strong. Balance learning rates or reduce D's capacity
* **Training instability**: Losses oscillate wildly. Use batch normalization and appropriate optimizers
* **Poor initialization**: Random seeds matter. Set seeds for reproducibility during development
* **Forgetting .detach()**: When training Discriminator on fake data, detach Generator gradients to prevent unwanted updates


References
==========

.. [Goodfellow2014] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative adversarial nets. In *Advances in Neural Information Processing Systems* (Vol. 27, pp. 2672-2680). https://papers.nips.cc/paper/5423-generative-adversarial-nets

.. [Goodfellow2016] Goodfellow, I. (2016). NIPS 2016 Tutorial: Generative Adversarial Networks. *arXiv preprint arXiv:1701.00160*. https://arxiv.org/abs/1701.00160

.. [Radford2016] Radford, A., Metz, L., & Chintala, S. (2016). Unsupervised representation learning with deep convolutional generative adversarial networks. In *Proceedings of the 4th International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1511.06434

.. [Arjovsky2017] Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein generative adversarial networks. In *Proceedings of the 34th International Conference on Machine Learning* (pp. 214-223). https://arxiv.org/abs/1701.07875

.. [Salimans2016] Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved techniques for training GANs. In *Advances in Neural Information Processing Systems* (Vol. 29, pp. 2234-2242). https://arxiv.org/abs/1606.03498

.. [Karras2019] Karras, T., Laine, S., & Aila, T. (2019). A style-based generator architecture for generative adversarial networks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 4401-4410). https://arxiv.org/abs/1812.04948

.. [Brock2019] Brock, A., Donahue, J., & Simonyan, K. (2019). Large scale GAN training for high fidelity natural image synthesis. In *Proceedings of the 7th International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1809.11096

.. [PyTorchDocs] PyTorch Team. (2024). PyTorch documentation: Neural network modules. *PyTorch*. https://pytorch.org/docs/stable/nn.html

.. [DevNag2017] Nag, D. (2017). Generative Adversarial Networks (GANs) in 50 lines of code (PyTorch). *GitHub*. https://github.com/devnag/pytorch-generative-adversarial-networks
