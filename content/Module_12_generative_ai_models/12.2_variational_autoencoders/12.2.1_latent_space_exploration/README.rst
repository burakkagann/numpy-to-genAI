.. _module-12-2-1-latent-space-exploration:

================================
12.2.1 Latent Space Exploration
================================

:Duration: 35-40 minutes
:Level: Intermediate

Overview
========

Variational Autoencoders (VAEs) are powerful generative models that learn compressed representations of data in a continuous latent space. Unlike traditional autoencoders that simply compress and reconstruct, VAEs learn a probabilistic mapping that enables generation of new samples by sampling from the latent distribution [Kingma2014]_.

In this exercise, you will explore the latent space of a trained VAE, discovering how these models organize learned representations and how to navigate the latent space to generate new content. Understanding latent spaces is fundamental to modern generative AI, from image synthesis to creative applications. The concepts you learn here apply directly to more advanced models like StyleGAN, diffusion models, and multimodal AI systems.

Learning Objectives
-------------------

By the end of this exercise, you will be able to:

* Understand the encoder-decoder architecture of Variational Autoencoders
* Explain what a latent space represents and how it encodes data characteristics
* Implement latent space visualization and interpolation techniques
* Generate new samples by sampling from learned latent distributions


Quick Start: See It In Action
=============================

Run this code to train a VAE and visualize its latent space:

.. code-block:: python
   :caption: Train a VAE on simple geometric patterns
   :linenos:

   import numpy as np
   import torch
   import torch.nn as nn

   # Simple VAE architecture
   class VAE(nn.Module):
       def __init__(self, input_dim=256, hidden_dim=128, latent_dim=8):
           super().__init__()
           # Encoder
           self.encoder = nn.Sequential(
               nn.Linear(input_dim, hidden_dim), nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
           )
           self.fc_mu = nn.Linear(hidden_dim, latent_dim)
           self.fc_var = nn.Linear(hidden_dim, latent_dim)
           # Decoder
           self.decoder = nn.Sequential(
               nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
               nn.Linear(hidden_dim, input_dim), nn.Sigmoid()
           )

       def encode(self, x):
           h = self.encoder(x)
           return self.fc_mu(h), self.fc_var(h)

       def reparameterize(self, mu, log_var):
           std = torch.exp(0.5 * log_var)
           eps = torch.randn_like(std)
           return mu + std * eps

       def forward(self, x):
           mu, log_var = self.encode(x)
           z = self.reparameterize(mu, log_var)
           return self.decoder(z), mu, log_var

   # Create and test VAE
   vae = VAE()
   test_input = torch.randn(4, 256)
   recon, mu, log_var = vae(test_input)
   print(f"Input shape: {test_input.shape}")
   print(f"Latent shape: {mu.shape}")
   print(f"Output shape: {recon.shape}")

.. figure:: reconstruction_comparison.png
   :width: 700px
   :align: center
   :alt: Comparison showing original geometric patterns on top and VAE reconstructions on bottom

   VAE reconstruction comparison. The top row shows original patterns; the bottom row shows reconstructions after encoding and decoding through the 8-dimensional latent space.

The VAE compresses 256-pixel images into just 8 latent dimensions, then reconstructs them. The slight blurriness in reconstructions reflects the probabilistic nature of VAEs, which prioritize learning a smooth, meaningful latent space over pixel-perfect reconstruction.


Core Concepts
=============

Concept 1: What is a Latent Space?
----------------------------------

A **latent space** is a compressed, learned representation where the essential features of data are encoded in fewer dimensions. The term "latent" means hidden or underlying, since these representations capture abstract properties not explicitly present in the raw data [Goodfellow2016]_.

Consider a collection of face images. While each image has millions of pixels, the underlying variation can be described by fewer factors: pose, expression, lighting, identity. The latent space captures these factors:

* **Dimensionality Reduction**: 256 pixels compressed to 8 latent dimensions
* **Semantic Organization**: Similar data points cluster together
* **Continuity**: Nearby points in latent space produce similar outputs

Unlike traditional dimensionality reduction (like PCA), neural network latent spaces are learned through training. The network discovers which features matter for reconstruction, creating representations tailored to the data [Bank2020]_.

.. figure:: latent_space_visualization.png
   :width: 600px
   :align: center
   :alt: 2D scatter plot showing encoded patterns clustered by type with colored points

   Latent space visualization showing how the VAE organizes different pattern types. Each color represents a pattern type (diagonal, horizontal, vertical, cross). Notice how similar patterns cluster together.

.. admonition:: Did You Know?

   The term "latent variable" comes from statistics and was used long before deep learning. Factor analysis in psychology (1900s) used latent variables to explain observed correlations between test scores. Modern VAEs extend this idea using neural networks to learn arbitrarily complex latent structures [Blei2017]_.


Concept 2: VAE Architecture
---------------------------

A **Variational Autoencoder** consists of three components working together [Kingma2014]_:

**The Encoder**

The encoder neural network maps input data to latent distribution parameters:

* **Input**: Original data (e.g., 16x16 image = 256 values)
* **Output**: Mean (mu) and variance (log_var) of a Gaussian distribution
* **Purpose**: Learn to compress data into meaningful latent representations

**The Reparameterization Trick**

To sample from the latent distribution while allowing gradient backpropagation:

.. code-block:: python
   :caption: The reparameterization trick

   def reparameterize(mu, log_var):
       std = torch.exp(0.5 * log_var)  # Convert log-variance to std
       epsilon = torch.randn_like(std)  # Random noise from N(0,1)
       z = mu + std * epsilon          # Sample: z = mu + std * noise
       return z

This trick expresses the random variable ``z`` as a deterministic function of ``mu``, ``log_var``, and random noise ``epsilon``. Since gradients can flow through ``mu`` and ``log_var``, the network can be trained with backpropagation [Rezende2014]_.

**The Decoder**

The decoder neural network reconstructs data from latent samples:

* **Input**: Sampled latent vector (z)
* **Output**: Reconstructed data
* **Purpose**: Learn to generate data from latent representations

.. figure:: vae_architecture.png
   :width: 700px
   :align: center
   :alt: Diagram showing VAE architecture with encoder, latent space, and decoder components

   VAE architecture diagram. The encoder maps input to distribution parameters (mu, log_var). The reparameterization trick samples z from this distribution. The decoder reconstructs the input from z.

**The Loss Function**

VAE training minimizes two terms:

1. **Reconstruction Loss**: How well does the output match the input?
2. **KL Divergence**: How close is the learned distribution to a standard normal?

.. code-block:: python
   :caption: VAE loss function

   # Reconstruction loss (how accurate is the output?)
   recon_loss = F.binary_cross_entropy(output, input, reduction='sum')

   # KL divergence (how close to N(0,1) is the latent distribution?)
   kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

   # Total loss
   total_loss = recon_loss + kl_loss

The KL divergence term regularizes the latent space, encouraging it to be smooth and continuous. This is what allows meaningful interpolation between points [Higgins2017]_.


Concept 3: Exploring the Latent Space
-------------------------------------

Once trained, the VAE's latent space can be explored in several ways:

**Sampling: Generating New Data**

Generate new samples by sampling from the prior distribution and decoding:

.. code-block:: python
   :caption: Generating new samples

   # Sample from standard normal (the prior)
   z = torch.randn(10, latent_dim)

   # Decode to generate new patterns
   with torch.no_grad():
       generated = vae.decoder(z)

**Interpolation: Walking Through Latent Space**

Create smooth transitions between two data points by interpolating their latent representations:

.. code-block:: python
   :caption: Latent space interpolation

   # Encode two patterns
   z_start, _ = vae.encode(pattern_start)
   z_end, _ = vae.encode(pattern_end)

   # Linear interpolation
   for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
       z_interp = (1 - alpha) * z_start + alpha * z_end
       output = vae.decoder(z_interp)

This works because the latent space is continuous and semantically meaningful. Moving along a straight path between two points produces gradual, sensible transformations [Doersch2016]_.

.. figure:: interpolation_sequence.png
   :width: 700px
   :align: center
   :alt: Sequence of 8 images showing smooth transition from diagonal line to cross pattern

   Latent space interpolation between a diagonal line pattern and a cross pattern. Each step represents a different interpolation weight (alpha), showing how the VAE smoothly morphs between patterns.

**Latent Space Structure**

The latent space often exhibits meaningful structure:

* **Clustering**: Similar inputs map to nearby regions
* **Directions**: Moving along certain directions changes specific attributes
* **Arithmetic**: Vector arithmetic can combine features (e.g., "smiling" direction + "glasses" direction)

.. figure:: training_progression.png
   :width: 500px
   :align: center
   :alt: Grid showing VAE reconstructions improving from epoch 1 to epoch 200

   Training progression showing how reconstruction quality improves. Early epochs produce blurry outputs as the network learns the data structure. Later epochs show clearer, more accurate reconstructions.

.. important::

   VAE reconstructions are often slightly blurry compared to GANs. This is because VAEs optimize for the entire distribution of possible outputs, while GANs focus on generating sharp samples. The trade-off is that VAE latent spaces tend to be smoother and more predictable.


Hands-On Exercises
==================

Exercise 1: Execute and Explore
-------------------------------

Run the complete VAE training script:

.. code-block:: python
   :caption: Run latent_space_exploration.py

   # From the exercise directory, run:
   python latent_space_exploration.py

This script trains a VAE on geometric patterns and generates all visualizations. After running, examine the generated images and answer these reflection questions:

**Reflection Questions:**

1. How does reconstruction quality change from epoch 1 to epoch 200?
2. In the latent space visualization, why do similar patterns cluster together?
3. Why are the VAE reconstructions slightly blurry compared to the originals?
4. What happens to the latent space structure if you increase the latent dimension from 8 to 32?

.. dropdown:: Answers and Explanation
   :class-title: sd-font-weight-bold

   **1. Reconstruction quality over epochs**

   At epoch 1, reconstructions are nearly random noise because the network has not learned the data structure. By epoch 50, basic shapes emerge. By epoch 200, patterns are clearly recognizable though slightly smoothed. The loss decreases rapidly at first, then plateaus as the network converges.

   **2. Why similar patterns cluster**

   The VAE learns to encode similar inputs to nearby latent points because this minimizes reconstruction loss. If two inputs require similar decoder outputs, mapping them to nearby latent points makes the decoder's job easier. The KL divergence term also encourages a smooth, organized latent space rather than scattered, random mappings.

   **3. Why VAE outputs are blurry**

   VAEs model the output distribution rather than producing a single deterministic output. The decoder learns to output the mean of all plausible reconstructions for a given latent point. This averaging effect produces smoother, blurrier outputs. GANs avoid this by using adversarial training to produce sharp samples.

   **4. Effect of increasing latent dimension**

   Increasing latent dimension from 8 to 32 gives the VAE more capacity to represent variation. Reconstructions may become sharper because more details can be encoded. However, the latent space becomes harder to visualize and may have more "dead" dimensions that the network does not use effectively.


Exercise 2: Modify Parameters
-----------------------------

Experiment with the VAE by modifying key parameters. Each modification reveals different aspects of latent space learning.

**Goal 1**: Change the latent dimension

Modify ``LATENT_DIM`` in ``vae_model.py`` and observe effects:

.. code-block:: python
   :caption: Experiment with latent dimensions

   LATENT_DIM = 2    # Minimal: easy to visualize, limited capacity
   LATENT_DIM = 8    # Default: good balance
   LATENT_DIM = 32   # Large: more capacity, harder to interpret
   LATENT_DIM = 64   # Very large: may be redundant for simple patterns

.. dropdown:: Solution: Latent Dimension Effects
   :class-title: sd-font-weight-bold

   * **LATENT_DIM = 2**: The latent space can be visualized directly without projection. However, reconstruction quality suffers because 2 dimensions cannot capture all pattern variations. Similar patterns may overlap in latent space.

   * **LATENT_DIM = 8**: Good balance between visualization and reconstruction. Four pattern types can be well-separated while maintaining reasonable reconstruction quality.

   * **LATENT_DIM = 32+**: Reconstructions may improve slightly, but many dimensions remain unused. The network learns to ignore extra dimensions, and visualization requires dimensionality reduction.

**Goal 2**: Implement beta-VAE by adjusting KL weight

The beta-VAE modifies the loss function to use a weighted KL term [Higgins2017]_:

.. code-block:: python
   :caption: Beta-VAE modification in vae_model.py

   def vae_loss(recon_x, x, mu, log_var, beta=1.0):
       recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
       kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
       return recon_loss + beta * kl_loss  # beta controls KL weight

   # Try different beta values:
   # beta = 0.5  # Less regularization, sharper reconstructions
   # beta = 1.0  # Standard VAE
   # beta = 4.0  # More regularization, more disentangled latent space

.. dropdown:: Solution: Beta-VAE Effects
   :class-title: sd-font-weight-bold

   * **beta < 1**: The network prioritizes reconstruction over latent space regularization. Outputs become sharper, but the latent space may become less organized and interpolation quality decreases.

   * **beta = 1**: Standard VAE behavior. Balanced trade-off between reconstruction and latent space structure.

   * **beta > 1**: Strong regularization pushes the latent space closer to a standard normal. This encourages disentanglement (each latent dimension controls one feature) but reconstruction quality decreases. This is the beta-VAE approach for learning interpretable representations.

**Goal 3**: Modify encoder/decoder depth

Add or remove layers to change network capacity:

.. code-block:: python
   :caption: Shallow vs deep networks

   # Shallow network (faster, less capacity)
   self.encoder = nn.Sequential(
       nn.Linear(input_dim, hidden_dim), nn.ReLU(),
       nn.Linear(hidden_dim, latent_dim)
   )

   # Deep network (slower, more capacity)
   self.encoder = nn.Sequential(
       nn.Linear(input_dim, hidden_dim), nn.ReLU(),
       nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
       nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
       nn.Linear(hidden_dim, latent_dim)
   )


Exercise 3: Create Your Own
---------------------------

Complete the starter code to implement latent space interpolation yourself.

**Requirements**:

* Implement the ``encode()`` method to map input to latent space
* Implement the ``decode()`` method to reconstruct from latent vector
* Implement the ``interpolate()`` function to create smooth transitions

**Starter Code**:

Open :download:`latent_starter.py <latent_starter.py>` and complete the TODO sections:

.. code-block:: python
   :caption: latent_starter.py (complete the TODOs)
   :linenos:

   def encode(self, x):
       """
       Encode input to latent space.

       TODO: Implement the encoding process
       1. Pass x through encoder_fc1 and apply ReLU
       2. Pass result through encoder_fc2 and apply ReLU
       3. Pass result through fc_mu to get latent mean
       """
       # TODO: Your implementation here
       pass

   def decode(self, z):
       """
       Decode latent vector to reconstructed image.

       TODO: Implement the decoding process
       1. Pass z through decoder_fc1 and apply ReLU
       2. Pass result through decoder_fc2 and apply ReLU
       3. Pass result through decoder_out
       4. Apply Sigmoid to get values in [0, 1]
       """
       # TODO: Your implementation here
       pass

   def interpolate(vae, pattern_a, pattern_b, num_steps=8):
       """
       TODO: Implement latent space interpolation
       1. Encode both patterns to get their latent representations
       2. For each step, compute: z = (1-alpha)*z_a + alpha*z_b
       3. Decode each interpolated latent vector
       """
       # TODO: Your implementation here
       pass

.. dropdown:: Hint 1: Encoding Process
   :class-title: sd-font-weight-bold

   The encoder passes data through sequential layers with ReLU activations:

   .. code-block:: python

      def encode(self, x):
          h = torch.relu(self.encoder_fc1(x))
          h = torch.relu(self.encoder_fc2(h))
          mu = self.fc_mu(h)
          return mu

.. dropdown:: Hint 2: Interpolation Logic
   :class-title: sd-font-weight-bold

   Linear interpolation computes weighted averages between two points:

   .. code-block:: python

      alphas = np.linspace(0, 1, num_steps)  # [0.0, 0.14, 0.29, ..., 1.0]
      for alpha in alphas:
          z = (1 - alpha) * z_a + alpha * z_b  # Weighted average
          # When alpha=0, z=z_a; when alpha=1, z=z_b

.. dropdown:: Complete Solution
   :class-title: sd-font-weight-bold

   .. code-block:: python
      :linenos:

      def encode(self, x):
          """Encode input to latent space."""
          h = torch.relu(self.encoder_fc1(x))
          h = torch.relu(self.encoder_fc2(h))
          mu = self.fc_mu(h)
          return mu

      def decode(self, z):
          """Decode latent vector to reconstructed image."""
          h = torch.relu(self.decoder_fc1(z))
          h = torch.relu(self.decoder_fc2(h))
          out = self.decoder_out(h)
          return torch.sigmoid(out)

      def interpolate(vae, pattern_a, pattern_b, num_steps=8):
          """Interpolate between two patterns in latent space."""
          vae.eval()
          interpolated = []

          with torch.no_grad():
              # Step 1: Encode both patterns
              z_a = vae.encode(pattern_a)
              z_b = vae.encode(pattern_b)

              # Step 2: Create interpolation alphas
              alphas = np.linspace(0, 1, num_steps)

              # Step 3: Interpolate and decode
              for alpha in alphas:
                  z = (1 - alpha) * z_a + alpha * z_b
                  decoded = vae.decode(z)
                  interpolated.append(decoded.numpy().reshape(IMAGE_SIZE, IMAGE_SIZE))

          return interpolated

**Challenge Extension**: Create an animated GIF that smoothly walks through the latent space, visiting all four pattern types in sequence:

.. dropdown:: Challenge Solution
   :class-title: sd-font-weight-bold

   .. code-block:: python

      import imageio

      def create_latent_walk_animation(vae, patterns, fps=10):
          """Create animated GIF walking through latent space."""
          frames = []

          # Encode all patterns
          with torch.no_grad():
              latents = [vae.encode(p) for p in patterns]

          # Create smooth path visiting all patterns
          for i in range(len(latents)):
              z_start = latents[i]
              z_end = latents[(i + 1) % len(latents)]

              # Interpolate between consecutive patterns
              for alpha in np.linspace(0, 1, 15):
                  z = (1 - alpha) * z_start + alpha * z_end
                  decoded = vae.decode(z).numpy().reshape(16, 16)
                  # Convert to uint8 for GIF
                  frame = (decoded * 255).astype(np.uint8)
                  frames.append(frame)

          imageio.mimsave('latent_walk.gif', frames, fps=fps)
          print("Saved: latent_walk.gif")


Summary
=======

Key Takeaways
-------------

* **Latent spaces** are compressed, learned representations where data is encoded in fewer dimensions
* **VAEs** learn probabilistic latent spaces using encoder-decoder architecture with the reparameterization trick
* The **reparameterization trick** enables gradient-based training by expressing sampling as a deterministic function of learnable parameters plus noise
* **KL divergence** regularizes the latent space, making it smooth and enabling meaningful interpolation
* **Interpolation** in latent space produces smooth transitions because nearby points generate similar outputs
* VAE outputs are slightly **blurry** because they model the mean of the output distribution rather than producing sharp samples

Common Pitfalls
---------------

* **Posterior collapse**: The VAE ignores the latent code and decoder learns to produce average outputs. Remedies include KL annealing or reducing decoder capacity
* **Blurry outputs**: Expected behavior for VAEs. If sharpness is critical, consider using VAE-GAN hybrids or diffusion models
* **Latent dimension too small**: Reconstruction quality suffers. Increase latent dimension if patterns are underfitting
* **Latent dimension too large**: Many dimensions go unused. Monitor which dimensions have variance during training
* **Forgetting to use** ``torch.no_grad()`` **during inference**: Wastes memory and computation on unnecessary gradient tracking


References
==========

.. [Kingma2014] Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. In *Proceedings of the 2nd International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1312.6114

.. [Doersch2016] Doersch, C. (2016). Tutorial on Variational Autoencoders. *arXiv preprint arXiv:1606.05908*. https://arxiv.org/abs/1606.05908

.. [Goodfellow2016] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 20: Deep Generative Models. MIT Press. https://www.deeplearningbook.org/

.. [Higgins2017] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., & Lerchner, A. (2017). beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. In *Proceedings of the 5th International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=Sy2fzU9gl

.. [Rezende2014] Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). Stochastic Backpropagation and Approximate Inference in Deep Generative Models. In *Proceedings of the 31st International Conference on Machine Learning (ICML)* (pp. 1278-1286). https://arxiv.org/abs/1401.4082

.. [Blei2017] Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational Inference: A Review for Statisticians. *Journal of the American Statistical Association*, 112(518), 859-877. https://doi.org/10.1080/01621459.2017.1285773

.. [Bank2020] Bank, D., Koenigstein, N., & Giryes, R. (2020). Autoencoders. *arXiv preprint arXiv:2003.05991*. https://arxiv.org/abs/2003.05991

.. [PyTorchDocs] PyTorch Team. (2024). PyTorch documentation: Neural network modules. *PyTorch*. https://docs.pytorch.org/docs/stable/nn.html

.. [NumPyDocs] NumPy Developers. (2024). NumPy array creation routines. *NumPy Documentation*. https://numpy.org/doc/stable/reference/routines.array-creation.html
