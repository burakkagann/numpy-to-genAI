.. _module-12-2-2-interpolation:

=====================================
12.2.2 Interpolation Animations
=====================================

:Duration: 35-40 minutes
:Level: Intermediate-Advanced
:Prerequisites: Module 12.2.1 (Latent Space Exploration), PyTorch basics

Overview
========

One of the most compelling demonstrations of what generative models have learned is the ability to create smooth morphing animations between generated images. By interpolating between points in the latent space, we can produce seamless transitions that reveal how the model organizes its internal representations.

Variational Autoencoders (VAEs) are particularly well-suited for interpolation because their latent spaces are explicitly regularized to be smooth and continuous [Kingma2014]_. Unlike GANs, where nearby latent points might produce very different outputs, VAE latent spaces are constrained to follow a Gaussian distribution. This regularization ensures that every point in the latent space maps to a plausible output, enabling high-quality morphing animations.

In this exercise, you will implement interpolation techniques to create animated morphing sequences through the VAE latent space.

Learning Objectives
-------------------

By the end of this exercise, you will be able to:

* Understand how VAE encoder-decoder architecture enables smooth latent space interpolation
* Implement linear and spherical (slerp) interpolation techniques between latent vectors
* Generate animated morphing sequences by walking through latent space paths
* Compare VAE interpolation characteristics with GAN-based approaches


Quick Start: See It In Action
=============================

Run this code to create a morphing animation between two randomly generated patterns:

.. code-block:: python
   :caption: Create a VAE morphing animation
   :linenos:

   import torch
   import imageio.v2 as imageio
   from vae_model import VAE, LATENT_DIM

   # Load pre-trained VAE
   vae = VAE(latent_dim=LATENT_DIM)
   vae.load_state_dict(torch.load('vae_weights.pth', map_location='cpu'))
   vae.eval()

   # Generate two random latent vectors
   torch.manual_seed(42)
   z_start = torch.randn(LATENT_DIM)
   z_end = torch.randn(LATENT_DIM)

   # Create interpolation frames
   frames = []
   for i in range(30):
       t = i / 29
       z = (1 - t) * z_start + t * z_end  # Linear interpolation
       z = z.unsqueeze(0)
       with torch.no_grad():
           img = vae.decoder(z)
       img = ((img[0] + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()
       frames.append((img * 255).astype('uint8'))

   imageio.mimsave('my_animation.gif', frames, fps=15)
   print("Animation saved!")

.. figure:: interpolation_animation.gif
   :width: 300px
   :align: center
   :alt: Animated GIF showing smooth morphing between abstract patterns

   Latent space interpolation animation. Notice how the patterns smoothly transform from one form to another, demonstrating the continuous structure of the VAE latent space.

The animation reveals that the VAE has learned a structured representation where nearby points produce similar outputs. This is a fundamental property that makes VAEs ideal for creative applications like morphing and style transfer.


Core Concepts
=============

Concept 1: VAE Architecture for Interpolation
---------------------------------------------

A Variational Autoencoder consists of two networks working together: an **encoder** that maps images to a probability distribution in latent space, and a **decoder** that generates images from latent vectors [Kingma2014]_.

.. figure:: vae_architecture.png
   :width: 700px
   :align: center
   :alt: Diagram showing VAE encoder-decoder architecture with latent space in the middle

   The VAE architecture. The encoder outputs mean (mu) and variance parameters for a Gaussian distribution. The decoder generates images from samples drawn from this distribution.

**Why VAE Latent Spaces Are Smooth**

The key innovation of VAEs is the **KL divergence loss**, which regularizes the latent space to match a standard normal distribution. This has two important effects [Bowman2016]_:

1. **No "holes"**: Every point in the latent space is valid. Unlike GANs where some regions might produce garbage, VAEs ensure all latent vectors decode to plausible outputs.

2. **Smooth transitions**: Nearby points in latent space produce similar outputs. Moving continuously through the space produces continuous changes in the generated images.

.. code-block:: python
   :caption: The reparameterization trick enables gradient flow

   def reparameterize(mu, logvar):
       """Sample from N(mu, var) using N(0, 1)."""
       std = torch.exp(0.5 * logvar)
       epsilon = torch.randn_like(std)
       z = mu + std * epsilon  # Reparameterized sample
       return z

The **reparameterization trick** allows gradients to flow through the sampling operation, making end-to-end training possible. This technique separates the randomness (epsilon) from the learned parameters (mu, sigma), enabling backpropagation [Kingma2014]_.

.. admonition:: Did You Know?

   The term "variational" in VAE comes from variational inference, a technique from Bayesian statistics. The VAE training objective (ELBO - Evidence Lower Bound) is derived by approximating an intractable posterior distribution [Goodfellow2016]_.


Concept 2: Interpolation Techniques
-----------------------------------

When interpolating between two latent vectors, we have several choices for how to trace the path from start to end.

**Linear Interpolation**

The simplest approach is linear interpolation, which follows a straight line through latent space:

.. code-block:: python
   :caption: Linear interpolation formula

   def linear_interpolate(z1, z2, t):
       """
       Interpolate linearly between z1 and z2.

       Args:
           z1: Starting latent vector
           z2: Ending latent vector
           t: Interpolation parameter in [0, 1]

       Returns:
           Interpolated vector: (1-t)*z1 + t*z2
       """
       return (1 - t) * z1 + t * z2

Linear interpolation is intuitive and works well for most applications. However, in high-dimensional spaces, the midpoint of a linear path can have lower magnitude than the endpoints, potentially producing less vibrant outputs.

**Spherical Interpolation (Slerp)**

Spherical linear interpolation follows a great circle path on a hypersphere, preserving the magnitude of the latent vectors [Shoemake1985]_. This can produce more consistent outputs when the magnitude of the latent vector affects the output intensity:

.. code-block:: python
   :caption: Spherical interpolation for smoother transitions

   def slerp(z1, z2, t):
       """
       Spherical linear interpolation between z1 and z2.

       Follows a great circle path on the hypersphere,
       maintaining consistent magnitude throughout.
       """
       # Normalize vectors
       z1_norm = z1 / (torch.norm(z1) + 1e-8)
       z2_norm = z2 / (torch.norm(z2) + 1e-8)

       # Calculate angle between vectors
       omega = torch.acos(torch.clamp(
           torch.sum(z1_norm * z2_norm), -1, 1
       ))

       if omega < 1e-6:
           return linear_interpolate(z1, z2, t)

       # Spherical interpolation formula
       sin_omega = torch.sin(omega)
       coef1 = torch.sin((1 - t) * omega) / sin_omega
       coef2 = torch.sin(t * omega) / sin_omega

       return coef1 * z1 + coef2 * z2

.. figure:: interpolation_comparison.png
   :width: 600px
   :align: center
   :alt: Side-by-side comparison of linear vs slerp interpolation

   Comparison of linear (top) and spherical (bottom) interpolation. For VAEs trained with standard priors, the differences are often subtle, but slerp can produce slightly more consistent outputs.

**When to Use Each Method**

* **Linear**: Default choice. Simple, fast, and usually sufficient for VAEs
* **Slerp**: Use when latent magnitude matters, or when working with normalized latent vectors [White2016]_


Concept 3: Creating Morphing Animations
---------------------------------------

To create smooth animations, we generate a sequence of frames by interpolating between keypoints in latent space, then combine them into an animated GIF.

**Frame Generation Loop**

.. code-block:: python
   :caption: Generating animation frames

   def generate_frames(decoder, z_start, z_end, num_frames=30):
       """Generate frames by interpolating through latent space."""
       frames = []

       with torch.no_grad():
           for i in range(num_frames):
               t = i / (num_frames - 1)
               z = linear_interpolate(z_start, z_end, t)
               z = z.unsqueeze(0)  # Add batch dimension

               # Decode to image
               image = decoder(z)

               # Convert to displayable format [0, 255]
               image = (image + 1) / 2  # [-1,1] -> [0,1]
               image = image.clamp(0, 1)
               image = image[0].permute(1, 2, 0).numpy()
               image = (image * 255).astype(np.uint8)

               frames.append(image)

       return frames

**Multi-Keypoint Looping Animations**

For more interesting animations, interpolate through multiple random keypoints and return to the start to create a seamless loop:

.. code-block:: python
   :caption: Creating looping animations through multiple keypoints

   # Generate 4 random keypoints
   keypoints = [torch.randn(LATENT_DIM) for _ in range(4)]
   keypoints.append(keypoints[0])  # Return to start for loop

   all_frames = []
   for i in range(len(keypoints) - 1):
       frames = generate_frames(
           decoder, keypoints[i], keypoints[i+1],
           num_frames=15
       )
       all_frames.extend(frames[:-1])  # Avoid duplicate frames

   imageio.mimsave('loop.gif', all_frames, fps=15, loop=0)

**Frame Rate and Smoothness**

The number of frames per segment determines the animation smoothness:

* **5-10 frames**: Noticeable stepping, but small file size
* **15-20 frames**: Good balance of smoothness and size
* **30+ frames**: Very smooth but larger files

.. figure:: interpolation_strip.png
   :width: 700px
   :align: center
   :alt: Eight frames showing gradual interpolation from one pattern to another

   Static visualization of interpolation steps (t=0.00 to t=1.00). Each frame represents a sample along the path from z_start to z_end.

.. important::

   When creating looping animations, skip the last frame of each segment (except the final one) to avoid duplicate frames at the transition points. This prevents a "stutter" effect in the loop.


Hands-On Exercises
==================

Exercise 1: Execute and Explore
-------------------------------

Run the complete interpolation script:

.. code-block:: python
   :caption: vae_interpolate.py
   :linenos:

   import torch
   import numpy as np
   import matplotlib.pyplot as plt
   import imageio.v2 as imageio
   from vae_model import VAE, LATENT_DIM

   def linear_interpolate(z1, z2, t):
       return (1 - t) * z1 + t * z2

   # Load VAE
   vae = VAE(latent_dim=LATENT_DIM)
   vae.load_state_dict(torch.load('vae_weights.pth', map_location='cpu'))
   vae.eval()

   # Generate keypoints and interpolate
   torch.manual_seed(42)
   z1 = torch.randn(LATENT_DIM)
   z2 = torch.randn(LATENT_DIM)

   frames = []
   with torch.no_grad():
       for i in range(30):
           t = i / 29
           z = linear_interpolate(z1, z2, t).unsqueeze(0)
           img = vae.decoder(z)
           img = ((img[0] + 1) / 2).clamp(0, 1)
           img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
           frames.append(img)

   imageio.mimsave('interpolation_animation.gif', frames, fps=15)
   print(f"Created animation with {len(frames)} frames")

After running the code, answer these reflection questions:

1. How does the VAE interpolation differ from DCGAN interpolation (Module 12.1.2)?
2. Are there any discontinuities or "jumps" in the transitions?
3. What visual elements remain consistent across the interpolation path?
4. Why does the VAE latent space produce such smooth transitions?

.. dropdown:: Answers and Explanation
   :class-title: sd-font-weight-bold

   1. **VAE vs DCGAN**: VAE interpolations tend to be smoother because the KL divergence loss regularizes the latent space to be a continuous Gaussian. DCGANs can have "dead zones" in latent space where outputs are inconsistent, while VAEs guarantee every point is valid.

   2. **Discontinuities**: You should observe smooth, continuous transitions with no sudden jumps. This is the key advantage of VAE latent spaces. Any artifacts are more likely to be blurriness rather than discontinuities.

   3. **Consistent elements**: Overall color palette and composition tend to remain coherent, even as specific shapes and patterns transform. This shows the VAE has learned meaningful high-level structure.

   4. **Why so smooth**: The KL divergence term in the VAE loss forces the encoder to map similar inputs to nearby latent points, AND ensures the latent space has no "holes." Every direction you can walk in latent space leads to a valid output [Kingma2014]_.


Exercise 2: Modify Parameters
-----------------------------

Experiment with the interpolation by modifying these parameters:

**Goal 1**: Compare interpolation smoothness with different frame counts

.. code-block:: python
   :caption: Experiment with frame counts

   # Choppy animation (few frames)
   frames_5 = generate_frames(decoder, z1, z2, num_frames=5)
   imageio.mimsave('choppy.gif', frames_5, fps=5)

   # Smooth animation (many frames)
   frames_60 = generate_frames(decoder, z1, z2, num_frames=60)
   imageio.mimsave('smooth.gif', frames_60, fps=30)

.. dropdown:: Hint
   :class-title: sd-font-weight-bold

   With 5 frames, transitions will look jerky. With 60 frames at 30 fps, you get a butter-smooth 2-second animation. The perceptual difference is significant even though the start and end points are the same.

**Goal 2**: Compare linear vs slerp interpolation

.. code-block:: python
   :caption: Side-by-side comparison

   # Generate both types
   linear_frames = []
   slerp_frames = []

   for i in range(30):
       t = i / 29
       z_linear = linear_interpolate(z1, z2, t)
       z_slerp = slerp(z1, z2, t)
       # ... generate and save frames ...

.. dropdown:: Solution
   :class-title: sd-font-weight-bold

   In practice, for VAEs with standard Gaussian priors, the difference between linear and slerp is often subtle. Slerp becomes more important when:

   * Working with normalized latent vectors
   * The latent magnitude correlates with output intensity
   * Interpolating through very distant points in latent space

**Goal 3**: Create a 4-keypoint looping animation

Create an animation that visits 4 random points in latent space and loops back to the start.

.. dropdown:: Complete Solution
   :class-title: sd-font-weight-bold

   .. code-block:: python

      torch.manual_seed(123)
      keypoints = [torch.randn(LATENT_DIM) for _ in range(4)]
      keypoints.append(keypoints[0])  # Close the loop

      all_frames = []
      for i in range(len(keypoints) - 1):
          for j in range(15):  # 15 frames per segment
              if i == len(keypoints) - 2 or j < 14:  # Skip last frame except final
                  t = j / 14
                  z = linear_interpolate(keypoints[i], keypoints[i+1], t)
                  z = z.unsqueeze(0)
                  with torch.no_grad():
                      img = vae.decoder(z)
                  img = ((img[0] + 1) / 2).clamp(0, 1)
                  img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                  all_frames.append(img)

      imageio.mimsave('loop_4point.gif', all_frames, fps=15, loop=0)
      print(f"Created looping animation: {len(all_frames)} frames")


Exercise 3: Re-code from Scratch
--------------------------------

Implement both interpolation functions yourself. The starter code provides the structure; fill in the interpolation logic.

**Requirements**:

* Implement ``linear_interpolate()`` function
* Implement ``slerp()`` function with fallback to linear
* Handle edge cases (identical vectors, t outside [0,1])

**Starter Code**:

.. code-block:: python
   :caption: interpolation_starter.py (complete the TODO sections)
   :linenos:

   import torch
   import numpy as np

   def linear_interpolate(z1, z2, t):
       """
       Linear interpolation between two latent vectors.

       TODO: Implement the formula: (1-t)*z1 + t*z2

       Args:
           z1: Starting vector
           z2: Ending vector
           t: Parameter in [0, 1]

       Returns:
           Interpolated vector
       """
       # TODO: Your code here
       pass


   def slerp(z1, z2, t):
       """
       Spherical linear interpolation.

       TODO: Implement slerp with these steps:
       1. Normalize both vectors
       2. Calculate angle omega = arccos(dot product)
       3. If omega is very small, fall back to linear
       4. Apply slerp formula

       Returns:
           Interpolated vector
       """
       # TODO: Your code here
       pass


   # Test your implementation
   if __name__ == '__main__':
       z1 = torch.randn(64)
       z2 = torch.randn(64)

       # Test linear interpolation
       assert torch.allclose(linear_interpolate(z1, z2, 0.0), z1)
       assert torch.allclose(linear_interpolate(z1, z2, 1.0), z2)

       # Test slerp
       mid = slerp(z1, z2, 0.5)
       assert mid.shape == z1.shape

       print("All tests passed!")

.. dropdown:: Hint 1: Linear interpolation
   :class-title: sd-font-weight-bold

   This is a one-liner. The weighted average formula is:

   ``result = (1 - t) * z1 + t * z2``

   When t=0, result equals z1. When t=1, result equals z2.

.. dropdown:: Hint 2: Slerp angle calculation
   :class-title: sd-font-weight-bold

   The angle between normalized vectors is found using the dot product:

   .. code-block:: python

      omega = torch.acos(torch.sum(z1_norm * z2_norm))

   Use ``torch.clamp()`` to handle numerical edge cases where the dot product slightly exceeds [-1, 1].

.. dropdown:: Complete Solution
   :class-title: sd-font-weight-bold

   .. code-block:: python
      :linenos:

      import torch

      def linear_interpolate(z1, z2, t):
          """Linear interpolation: straight line path."""
          return (1 - t) * z1 + t * z2


      def slerp(z1, z2, t):
          """Spherical linear interpolation: great circle path."""
          # Normalize vectors
          z1_norm = z1 / (torch.norm(z1) + 1e-8)
          z2_norm = z2 / (torch.norm(z2) + 1e-8)

          # Calculate angle between vectors
          dot = torch.sum(z1_norm * z2_norm)
          dot = torch.clamp(dot, -1.0, 1.0)
          omega = torch.acos(dot)

          # Fall back to linear if vectors are nearly identical
          if omega < 1e-6:
              return linear_interpolate(z1, z2, t)

          # Slerp formula
          sin_omega = torch.sin(omega)
          coef1 = torch.sin((1 - t) * omega) / sin_omega
          coef2 = torch.sin(t * omega) / sin_omega

          return coef1 * z1 + coef2 * z2


      if __name__ == '__main__':
          z1 = torch.randn(64)
          z2 = torch.randn(64)

          # Test linear
          assert torch.allclose(linear_interpolate(z1, z2, 0.0), z1)
          assert torch.allclose(linear_interpolate(z1, z2, 1.0), z2)

          # Test slerp
          mid = slerp(z1, z2, 0.5)
          assert mid.shape == z1.shape

          print("All tests passed!")

**Challenge Extension**: Add easing functions to create more natural-feeling animations. Implement ease-in-out timing:

.. dropdown:: Challenge Solution
   :class-title: sd-font-weight-bold

   .. code-block:: python

      def ease_in_out(t):
          """Smooth ease-in-out curve (cubic)."""
          if t < 0.5:
              return 4 * t * t * t
          else:
              return 1 - pow(-2 * t + 2, 3) / 2

      # Use in frame generation
      for i in range(num_frames):
          t_linear = i / (num_frames - 1)
          t_eased = ease_in_out(t_linear)  # Apply easing
          z = linear_interpolate(z1, z2, t_eased)
          # ... generate frame ...

   With easing, the animation will start slowly, accelerate in the middle, and slow down at the end, creating a more polished feel.


Summary
=======

Key Takeaways
-------------

* **VAE latent spaces** are inherently smooth due to KL divergence regularization, making them ideal for interpolation
* **Linear interpolation** ``(1-t)*z1 + t*z2`` is simple and works well for most VAE applications
* **Spherical interpolation (slerp)** follows a great circle path and can produce more consistent results for normalized vectors
* **Multi-keypoint paths** create more interesting animations by visiting several random points in latent space
* **Frame rate** affects perceived smoothness: 15-30 frames per segment at 15-30 fps produces fluid motion
* The **reparameterization trick** enables gradient-based training while maintaining stochastic sampling

Common Pitfalls
---------------

* **Forgetting batch dimension**: VAE decoder expects input shape ``(batch, latent_dim)``, not just ``(latent_dim,)``
* **Output range mismatch**: VAE outputs are in [-1, 1]; convert to [0, 1] or [0, 255] for display
* **Duplicate frames in loops**: Skip the last frame of each segment to avoid "stuttering" at transitions
* **Large GIF files**: Too many frames or high resolution can create unwieldy file sizes; balance quality with practicality


Next Steps
==========

Continue to :doc:`../12.2.3_conditional_vaes/README` to learn how to add control over VAE generation using class labels and conditional inputs.


References
==========

.. [Kingma2014] Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. *arXiv preprint*. https://arxiv.org/abs/1312.6114

.. [White2016] White, T. (2016). Sampling Generative Networks. *arXiv preprint*. https://arxiv.org/abs/1609.04468

.. [Bowman2016] Bowman, S. R., Vilnis, L., Vinyals, O., Dai, A. M., Jozefowicz, R., & Bengio, S. (2016). Generating Sentences from a Continuous Space. *Proceedings of CoNLL*. https://arxiv.org/abs/1511.06349

.. [Shoemake1985] Shoemake, K. (1985). Animating Rotation with Quaternion Curves. *ACM SIGGRAPH Computer Graphics*, 19(3), 245-254. https://doi.org/10.1145/325165.325242

.. [Higgins2017] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., ... & Lerchner, A. (2017). beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. *ICLR*.

.. [Goodfellow2016] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 20: Deep Generative Models (pp. 651-716). MIT Press. https://www.deeplearningbook.org/

.. [PyTorchDocs] PyTorch Contributors. (2024). PyTorch Documentation (Version 2.5). Retrieved December 26, 2025, from https://pytorch.org/docs/stable/

.. [Bransford2000] Bransford, J. D., Brown, A. L., & Cocking, R. R. (Eds.). (2000). *How People Learn: Brain, Mind, Experience, and School*. National Academy Press.
