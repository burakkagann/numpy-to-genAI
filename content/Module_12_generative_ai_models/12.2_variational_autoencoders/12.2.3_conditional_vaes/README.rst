.. _module-12-2-3-conditional-vaes:

================================
12.2.3 Conditional VAEs
================================

:Duration: 35-40 minutes
:Level: Intermediate

Overview
========

Variational Autoencoders (VAEs) are powerful generative models that learn to map data into a structured latent space and generate new samples by decoding random points from that space. However, standard VAEs have a limitation: you cannot control *what* they generate. The output is essentially random, determined only by the sampled latent vector.

**Conditional Variational Autoencoders (CVAEs)** solve this problem by incorporating class labels into both the encoder and decoder [Sohn2015]_. By conditioning the model on labels, you gain explicit control over generation. Want to generate the digit "7"? Simply provide label 7 to the decoder. This conditioning mechanism is the foundation for many controlled generation applications, from style transfer to attribute manipulation.

In this exercise, you will implement a Conditional VAE from scratch using PyTorch, train it on the MNIST handwritten digit dataset, and explore how conditioning enables targeted generation. This hands-on experience builds intuition for the more advanced conditional generation techniques used in modern AI art systems.

Learning Objectives
-------------------

By the end of this exercise, you will be able to:

* Understand the VAE architecture including encoder, decoder, and reparameterization trick
* Implement label conditioning by concatenating one-hot encoded labels to network inputs
* Train a Conditional VAE on MNIST and visualize the results
* Generate specific digit classes on demand using learned conditional distributions


Quick Start: See It In Action
=============================

Run this code to train a Conditional VAE and generate digits on demand:

.. code-block:: python
   :caption: Train a CVAE to generate specific digits
   :linenos:

   import torch
   import torch.nn as nn

   # Device selection
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # Simple Conditional VAE architecture
   class CVAE(nn.Module):
       def __init__(self, latent_dim=16, num_classes=10):
           super().__init__()
           self.latent_dim = latent_dim
           self.num_classes = num_classes

           # Encoder: image (784) + label (10) -> latent distribution
           self.encoder = nn.Sequential(
               nn.Linear(784 + num_classes, 256),
               nn.ReLU(),
               nn.Linear(256, 256),
               nn.ReLU()
           )
           self.fc_mu = nn.Linear(256, latent_dim)
           self.fc_logvar = nn.Linear(256, latent_dim)

           # Decoder: latent (16) + label (10) -> image (784)
           self.decoder = nn.Sequential(
               nn.Linear(latent_dim + num_classes, 256),
               nn.ReLU(),
               nn.Linear(256, 256),
               nn.ReLU(),
               nn.Linear(256, 784),
               nn.Sigmoid()
           )

       def encode(self, x, labels_onehot):
           combined = torch.cat([x, labels_onehot], dim=1)
           h = self.encoder(combined)
           return self.fc_mu(h), self.fc_logvar(h)

       def decode(self, z, labels_onehot):
           combined = torch.cat([z, labels_onehot], dim=1)
           return self.decoder(combined)

       def generate(self, labels, num_samples=1):
           """Generate samples for specified digit labels."""
           self.eval()
           with torch.no_grad():
               labels_onehot = torch.zeros(num_samples, 10, device=device)
               for i, label in enumerate(labels):
                   labels_onehot[i, label] = 1
               z = torch.randn(num_samples, self.latent_dim, device=device)
               return self.decode(z, labels_onehot)

   # Create model and generate a digit
   model = CVAE().to(device)
   sample = model.generate([7])  # Generate a "7"
   print(f"Generated digit shape: {sample.shape}")

.. figure:: cvae_generated_samples.png
   :width: 600px
   :align: center
   :alt: Grid of CVAE-generated MNIST digits organized by class, showing 10 samples per digit from 0 to 9

   CVAE-generated digits after training. Each row shows 10 samples conditioned on a specific digit class (0-9). The model learns to generate recognizable digits while maintaining variation within each class.

The key insight is that the same random noise produces different outputs depending on which label you provide. This is the power of conditional generation: explicit control over the output while preserving the generative variety of the latent space.


Core Concepts
=============

Concept 1: VAE Fundamentals
---------------------------

Before diving into conditioning, let us review the core architecture of Variational Autoencoders [Kingma2014]_.

**The Encoder**

The encoder takes an input (such as an image) and maps it to a probability distribution in latent space, rather than a single point. It outputs two vectors:

* **mu (mean)**: The center of the distribution
* **logvar (log-variance)**: The spread of the distribution

.. code-block:: python
   :caption: Encoder outputs distribution parameters

   def encode(self, x):
       h = self.encoder_layers(x)
       mu = self.fc_mu(h)        # Mean of latent distribution
       logvar = self.fc_logvar(h)  # Log-variance
       return mu, logvar

**The Reparameterization Trick**

To sample from the latent distribution while allowing gradients to flow during training, we use the reparameterization trick [Rezende2014]_:

.. math::

   z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)

Instead of sampling directly from the learned distribution, we sample noise from a standard normal and transform it using the learned parameters.

.. code-block:: python
   :caption: Reparameterization enables gradient flow

   def reparameterize(self, mu, logvar):
       std = torch.exp(0.5 * logvar)  # Convert log-variance to std
       epsilon = torch.randn_like(std)  # Random noise
       z = mu + std * epsilon  # Sample from learned distribution
       return z

**The Decoder**

The decoder takes a point from latent space and reconstructs the original input:

.. code-block:: python
   :caption: Decoder reconstructs from latent space

   def decode(self, z):
       reconstruction = self.decoder_layers(z)
       return reconstruction

**VAE Loss Function**

The VAE objective balances two goals:

1. **Reconstruction Loss**: How well does the decoder recreate the input?
2. **KL Divergence**: How close is the latent distribution to a standard normal?

.. math::

   \mathcal{L} = \mathbb{E}[\log p(x|z)] - D_{KL}(q(z|x) || p(z))

This loss encourages the model to both reconstruct accurately and maintain a structured latent space.

.. figure:: cvae_training_history.png
   :width: 600px
   :align: center
   :alt: Three plots showing total loss, reconstruction loss, and KL divergence during CVAE training

   Training dynamics of the Conditional VAE. The reconstruction loss (center) decreases as the model learns to reconstruct digits. The KL divergence (right) stabilizes as the latent space becomes structured. Visualization created with Matplotlib [MatplotlibDocs]_.


Concept 2: Label Conditioning
-----------------------------

The key innovation in Conditional VAEs is incorporating class labels into both the encoder and decoder [Sohn2015]_. This allows the model to learn class-specific representations and generate targeted outputs.

**Why Condition?**

In a standard VAE, the latent space mixes all classes together. You sample randomly and get a random output. With conditioning:

* The encoder learns what features are relevant for each class
* The decoder knows which class to generate
* The latent space can focus on within-class variation (style, thickness, rotation)

Think of it as giving an artist instructions: instead of saying "draw something," you can say "draw a 7."

**How to Condition: Concatenation**

The simplest conditioning approach concatenates the class label (as a one-hot vector) to the network inputs:

.. code-block:: python
   :caption: Conditioning through concatenation
   :linenos:
   :emphasize-lines: 6,12

   def encode(self, x, labels_onehot):
       # x: flattened image (batch_size, 784)
       # labels_onehot: one-hot labels (batch_size, 10)

       # Concatenate image with label information
       combined = torch.cat([x, labels_onehot], dim=1)  # (batch_size, 794)

       # Process through encoder
       h = self.encoder_layers(combined)
       return self.fc_mu(h), self.fc_logvar(h)

   def decode(self, z, labels_onehot):
       # Concatenate latent vector with label
       combined = torch.cat([z, labels_onehot], dim=1)  # (batch_size, 26)

       # Generate conditioned output
       return self.decoder_layers(combined)

The one-hot encoding represents each class as a binary vector:

* Label 0: ``[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]``
* Label 7: ``[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]``

.. figure:: cvae_architecture_diagram.png
   :width: 700px
   :align: center
   :alt: Diagram showing CVAE architecture with label concatenation at encoder input and decoder input

   Conditional VAE architecture. Labels are concatenated to both the encoder input (image + label) and decoder input (latent + label). This allows the model to learn class-aware representations.

.. admonition:: Did You Know?

   The Conditional VAE was introduced in 2015 by Sohn, Lee, and Yan [Sohn2015]_. They showed that by conditioning on auxiliary information, VAEs could perform structured prediction tasks like image segmentation and future frame prediction. The same conditioning principle now powers modern text-to-image systems, where text embeddings act as the "label" guiding generation.


Concept 3: Training and Generation
----------------------------------

Training a CVAE follows the same principles as a standard VAE, but with labels provided at each step.

**Training Loop**

.. code-block:: python
   :caption: CVAE training step
   :linenos:

   def train_step(model, images, labels, optimizer):
       optimizer.zero_grad()

       # Convert labels to one-hot
       labels_onehot = torch.zeros(labels.size(0), 10, device=device)
       labels_onehot.scatter_(1, labels.unsqueeze(1), 1)

       # Forward pass with labels
       mu, logvar = model.encode(images, labels_onehot)
       z = model.reparameterize(mu, logvar)
       reconstruction = model.decode(z, labels_onehot)

       # Compute loss
       recon_loss = F.binary_cross_entropy(reconstruction, images, reduction='sum')
       kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
       loss = recon_loss + kl_loss

       # Backward pass
       loss.backward()
       optimizer.step()

       return loss.item()

**Conditional Generation**

After training, generation is straightforward: sample from the latent prior and decode with the desired label:

.. code-block:: python
   :caption: Generate specific digits

   def generate(model, digit, num_samples=10):
       model.eval()
       with torch.no_grad():
           # Create one-hot label for desired digit
           labels_onehot = torch.zeros(num_samples, 10, device=device)
           labels_onehot[:, digit] = 1

           # Sample from prior and decode
           z = torch.randn(num_samples, model.latent_dim, device=device)
           samples = model.decode(z, labels_onehot)

       return samples

.. figure:: cvae_conditional_generation.png
   :width: 700px
   :align: center
   :alt: Two rows of digits 0-9 showing same latent vector with different labels (top) vs different latent vectors (bottom)

   Demonstration of conditional control. Top row: The same latent vector decoded with labels 0-9 produces different digits. Bottom row: Different random latent vectors produce variation within each class.

**The Power of Conditioning**

The top row of the figure above demonstrates the core capability: a single random vector ``z`` produces 10 different digits depending on the label. This is impossible with standard VAEs, where you have no control over what gets generated.

.. important::

   The latent space in a CVAE captures *within-class variation* rather than *between-class variation*. The label handles class identity, while the latent vector controls style, thickness, rotation, and other attributes that vary within a class.


Hands-On Exercises
==================

Exercise 1: Execute and Explore
-------------------------------

Run the :download:`conditional_vae.py <conditional_vae.py>` script to train a Conditional VAE on MNIST:

.. code-block:: bash

   python conditional_vae.py

The training takes approximately 5 minutes on CPU or 2-3 minutes on GPU. After training, examine the generated images.

**Reflection Questions:**

1. Looking at ``cvae_generated_samples.png``, which digits appear most realistic? Which are most challenging for the model?
2. In ``cvae_conditional_generation.png``, the top row uses the same latent vector with different labels. What aspects of the digits remain similar across classes?
3. How do the CVAE-generated digits compare to the GAN-generated patterns from Module 12.1.1?
4. What happens to the KL divergence during training? Why does it stabilize rather than continuing to decrease?

.. dropdown:: Answers and Explanation
   :class-title: sd-font-weight-bold

   1. **Digit quality**: Digits like 1, 0, and 7 typically look most realistic because they have simpler, more consistent shapes. Digits like 8, 9, and 4 are often more challenging because they have more complex curves or easily confused features.

   2. **Shared characteristics**: When using the same latent vector, you may notice similar stroke thickness, slight rotation angles, or overall positioning across different digit classes. The latent space captures these style attributes while the label controls the digit identity.

   3. **CVAE vs GAN comparison**: CVAEs often produce slightly blurrier outputs compared to GANs but offer more stable training and explicit control through conditioning. GANs may produce sharper images but lack the structured latent space that enables interpolation and manipulation.

   4. **KL divergence behavior**: The KL term encourages the latent distribution to match a standard normal. It stabilizes because the model finds a balance between reconstruction quality (which benefits from complex distributions) and regularization (which prefers simple distributions). Too much KL pressure destroys useful information; too little creates an unstructured latent space.


Exercise 2: Modify Parameters
-----------------------------

Experiment with the CVAE by modifying key hyperparameters. Compare results to understand their effects.

**Goal 1**: Change the latent dimension

Modify ``LATENT_DIM`` in the script:

.. code-block:: python
   :caption: Try different latent dimensions

   LATENT_DIM = 2    # Very small: limited variation, but visualizable
   LATENT_DIM = 16   # Default: good balance
   LATENT_DIM = 64   # Large: more capacity for variation

.. dropdown:: Hint: Latent Dimension Effects
   :class-title: sd-font-weight-bold

   * **Small (2)**: Limited capacity to encode variation. All digits of the same class may look identical. However, you can easily visualize the 2D latent space.
   * **Medium (16)**: Good balance between expressiveness and regularization.
   * **Large (64)**: More capacity but harder to regularize. May see posterior collapse if not careful.

**Goal 2**: Generate only even numbers

Modify the generation code to produce only digits 0, 2, 4, 6, 8:

.. code-block:: python
   :caption: Generate specific digit subsets

   even_digits = [0, 2, 4, 6, 8]
   for digit in even_digits:
       samples = model.generate([digit] * 5)
       # Visualize samples

**Goal 3**: Increase training epochs

Change ``NUM_EPOCHS`` from 50 to 100 or 200. Observe:

* Does reconstruction quality improve?
* Do the generated digits look sharper?
* Is there a point of diminishing returns?

.. dropdown:: Solution: Extended Training
   :class-title: sd-font-weight-bold

   With more epochs:

   * Reconstruction loss continues to decrease slowly
   * Generated digits become sharper, especially complex ones like 8 and 9
   * After ~100-150 epochs, improvements become marginal
   * Training for too long risks overfitting, though VAEs are relatively robust

**Goal 4**: Try Fashion-MNIST

Replace MNIST with Fashion-MNIST (clothing items):

.. code-block:: python
   :caption: Switch to Fashion-MNIST

   from torchvision import datasets

   train_dataset = datasets.FashionMNIST(
       root='./data',
       train=True,
       transform=transforms.ToTensor(),
       download=True
   )

How does the model perform on more complex images?


Exercise 3: Re-code from Scratch
--------------------------------

Build your own Conditional VAE using the starter code below. Complete the TODO sections.

**Requirements**:

* Implement the encoder with label concatenation
* Implement the decoder with label concatenation
* Implement the reparameterization trick
* Complete the loss function

**Starter Code**:

.. code-block:: python
   :caption: cvae_starter.py (complete the TODO sections)
   :linenos:

   import torch
   import torch.nn as nn
   import torch.nn.functional as F

   class SimpleCVAE(nn.Module):
       def __init__(self, latent_dim=16, num_classes=10):
           super().__init__()
           self.latent_dim = latent_dim
           self.num_classes = num_classes

           # TODO: Define encoder layers
           # Input: 784 (image) + 10 (label) = 794
           self.encoder_fc1 = nn.Linear(???, 256)
           self.encoder_fc2 = nn.Linear(256, 128)
           self.fc_mu = nn.Linear(128, latent_dim)
           self.fc_logvar = nn.Linear(128, latent_dim)

           # TODO: Define decoder layers
           # Input: latent_dim + 10 (label)
           self.decoder_fc1 = nn.Linear(???, 128)
           self.decoder_fc2 = nn.Linear(128, 256)
           self.decoder_fc3 = nn.Linear(256, 784)

       def encode(self, x, labels_onehot):
           # TODO: Concatenate x and labels_onehot
           combined = ???
           h = F.relu(self.encoder_fc1(combined))
           h = F.relu(self.encoder_fc2(h))
           return self.fc_mu(h), self.fc_logvar(h)

       def reparameterize(self, mu, logvar):
           # TODO: Implement reparameterization trick
           # z = mu + std * epsilon, where std = exp(0.5 * logvar)
           std = ???
           epsilon = ???
           z = ???
           return z

       def decode(self, z, labels_onehot):
           # TODO: Concatenate z and labels_onehot
           combined = ???
           h = F.relu(self.decoder_fc1(combined))
           h = F.relu(self.decoder_fc2(h))
           return torch.sigmoid(self.decoder_fc3(h))

       def forward(self, x, labels):
           # Convert labels to one-hot
           labels_onehot = F.one_hot(labels, self.num_classes).float()
           mu, logvar = self.encode(x, labels_onehot)
           z = self.reparameterize(mu, logvar)
           return self.decode(z, labels_onehot), mu, logvar

   def cvae_loss(reconstruction, original, mu, logvar):
       # TODO: Compute reconstruction loss (binary cross entropy)
       recon_loss = ???

       # TODO: Compute KL divergence
       # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
       kl_loss = ???

       return recon_loss + kl_loss

.. dropdown:: Hint 1: Layer Dimensions
   :class-title: sd-font-weight-bold

   For the encoder:

   .. code-block:: python

      # Image (784) + one-hot label (10) = 794
      self.encoder_fc1 = nn.Linear(784 + 10, 256)

   For the decoder:

   .. code-block:: python

      # Latent (16) + one-hot label (10) = 26
      self.decoder_fc1 = nn.Linear(latent_dim + 10, 128)

.. dropdown:: Hint 2: Reparameterization
   :class-title: sd-font-weight-bold

   .. code-block:: python

      def reparameterize(self, mu, logvar):
          std = torch.exp(0.5 * logvar)
          epsilon = torch.randn_like(std)
          z = mu + std * epsilon
          return z

.. dropdown:: Complete Solution
   :class-title: sd-font-weight-bold

   .. code-block:: python
      :linenos:

      import torch
      import torch.nn as nn
      import torch.nn.functional as F

      class SimpleCVAE(nn.Module):
          def __init__(self, latent_dim=16, num_classes=10):
              super().__init__()
              self.latent_dim = latent_dim
              self.num_classes = num_classes

              # Encoder: 784 (image) + 10 (label) = 794 input features
              self.encoder_fc1 = nn.Linear(784 + 10, 256)
              self.encoder_fc2 = nn.Linear(256, 128)
              self.fc_mu = nn.Linear(128, latent_dim)
              self.fc_logvar = nn.Linear(128, latent_dim)

              # Decoder: latent_dim + 10 (label) input features
              self.decoder_fc1 = nn.Linear(latent_dim + 10, 128)
              self.decoder_fc2 = nn.Linear(128, 256)
              self.decoder_fc3 = nn.Linear(256, 784)

          def encode(self, x, labels_onehot):
              combined = torch.cat([x, labels_onehot], dim=1)
              h = F.relu(self.encoder_fc1(combined))
              h = F.relu(self.encoder_fc2(h))
              return self.fc_mu(h), self.fc_logvar(h)

          def reparameterize(self, mu, logvar):
              std = torch.exp(0.5 * logvar)
              epsilon = torch.randn_like(std)
              z = mu + std * epsilon
              return z

          def decode(self, z, labels_onehot):
              combined = torch.cat([z, labels_onehot], dim=1)
              h = F.relu(self.decoder_fc1(combined))
              h = F.relu(self.decoder_fc2(h))
              return torch.sigmoid(self.decoder_fc3(h))

          def forward(self, x, labels):
              labels_onehot = F.one_hot(labels, self.num_classes).float()
              mu, logvar = self.encode(x, labels_onehot)
              z = self.reparameterize(mu, logvar)
              return self.decode(z, labels_onehot), mu, logvar

      def cvae_loss(reconstruction, original, mu, logvar):
          recon_loss = F.binary_cross_entropy(
              reconstruction, original, reduction='sum'
          )
          kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
          return recon_loss + kl_loss

**Challenge Extension**: Add a second conditioning variable for digit thickness. Modify the architecture to accept both a digit label (0-9) and a thickness level (thin, medium, thick). This requires augmenting your training data with thickness labels.

.. dropdown:: Challenge Hint
   :class-title: sd-font-weight-bold

   Multi-attribute conditioning works by concatenating multiple one-hot vectors:

   .. code-block:: python

      # Digit (10 classes) + Thickness (3 classes) = 13 conditioning dims
      combined = torch.cat([x, digit_onehot, thickness_onehot], dim=1)

   You would need to:

   1. Create thickness labels for training data (e.g., based on stroke width)
   2. Modify encoder input: 784 + 10 + 3 = 797
   3. Modify decoder input: latent_dim + 10 + 3
   4. At generation time, specify both digit and thickness


Summary
=======

Key Takeaways
-------------

* **VAEs** learn a probabilistic latent space using an encoder (to distribution parameters) and decoder (from samples)
* The **reparameterization trick** enables gradient flow through the sampling operation: z = mu + std * epsilon
* **Conditional VAEs** add label information to both encoder and decoder via concatenation
* Labels control **what** to generate; the latent space controls **how** (style, variation)
* The VAE loss balances **reconstruction quality** with **latent space regularization** (KL divergence)
* Trained CVAEs can generate specific classes on demand while maintaining within-class diversity

Common Pitfalls
---------------

* **Posterior collapse**: If KL weight is too high, the model ignores the latent space and relies only on the label. Try annealing the KL weight during training.
* **Blurry outputs**: VAEs tend to produce blurrier outputs than GANs. This is a known limitation of the reconstruction loss.
* **Label leakage**: If the model learns to encode label information in the latent space, conditioning becomes redundant. The KL term helps prevent this.
* **Wrong concatenation dimension**: Always concatenate along dim=1 (feature dimension), not dim=0 (batch dimension).


References
==========

.. [Kingma2014] Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. In *Proceedings of the 2nd International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1312.6114

.. [Sohn2015] Sohn, K., Lee, H., & Yan, X. (2015). Learning structured output representation using deep conditional generative models. In *Advances in Neural Information Processing Systems* (Vol. 28, pp. 3483-3491). https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models

.. [Rezende2014] Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). Stochastic backpropagation and approximate inference in deep generative models. In *Proceedings of the 31st International Conference on Machine Learning* (pp. 1278-1286). https://arxiv.org/abs/1401.4082

.. [Doersch2016] Doersch, C. (2016). Tutorial on variational autoencoders. *arXiv preprint arXiv:1606.05908*. https://arxiv.org/abs/1606.05908

.. [Goodfellow2016] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 20: Deep Generative Models. MIT Press. https://www.deeplearningbook.org/

.. [LeCun1998] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324. https://doi.org/10.1109/5.726791

.. [PyTorchDocs] PyTorch Team. (2024). PyTorch documentation: Neural network modules. *PyTorch*. https://pytorch.org/docs/stable/nn.html

.. [Sweller1988] Sweller, J. (1988). Cognitive load during problem solving: Effects on learning. *Cognitive Science*, 12(2), 257-285. https://doi.org/10.1207/s15516709cog1202_4

.. [NumPyDocs] NumPy Developers. (2024). NumPy documentation. *NumPy*. https://numpy.org/doc/stable/

.. [MatplotlibDocs] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90-95. https://matplotlib.org/
