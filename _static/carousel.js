/* ==========================================================================
   3D Cube Gallery - Random Image Selection
   Selects 6 random images from pool on each page load
   ========================================================================== */

document.addEventListener('DOMContentLoaded', function() {
    // All available images (22 total) - named by module origin
    const allImages = [
        // Module 2: Geometry & Mathematics
        { src: '_static/carousel/m2.1.4_star_fields.png', alt: 'Star Fields (Module 2.1.4)' },
        { src: '_static/carousel/m2.2.2_color_spiral.png', alt: 'Color Spiral (Module 2.2.2)' },
        { src: '_static/carousel/m2.2.4_distance_fields.png', alt: 'Distance Fields (Module 2.2.4)' },
        { src: '_static/carousel/m2.3.3_harmonograph.png', alt: 'Harmonograph (Module 2.3.3)' },
        { src: '_static/carousel/m2.3.3_colored_harmonograph.png', alt: 'Colored Harmonograph (Module 2.3.3)' },

        // Module 3: Transformations & Effects
        { src: '_static/carousel/m3.1.3_nonlinear_distortions.png', alt: 'Nonlinear Distortions (Module 3.1.3)' },
        { src: '_static/carousel/m3.1.4_kaleidoscope.png', alt: 'Kaleidoscope (Module 3.1.4)' },
        { src: '_static/carousel/m3.1.4_simple_kaleidoscope.png', alt: 'Simple Kaleidoscope (Module 3.1.4)' },
        { src: '_static/carousel/m3.3.5_delaunay_short.gif', alt: 'Delaunay Triangulation (Module 3.3.5)' },
        { src: '_static/carousel/m3.3.5_delaunay_full.gif', alt: 'Delaunay Triangulation Full (Module 3.3.5)' },

        // Module 4: Fractals & Recursion
        { src: '_static/carousel/m4.1.1_fractal_square.png', alt: 'Fractal Square (Module 4.1.1)' },

        // Module 5: Simulation & Emergent Behavior
        { src: '_static/carousel/m5.2.1_boids_predator.gif', alt: 'Boids Predator (Module 5.2.1)' },
        { src: '_static/carousel/m5.3.3_double_pendulum.gif', alt: 'Double Pendulum (Module 5.3.3)' },

        // Module 6: Noise & Procedural Generation
        { src: '_static/carousel/m6.4.2_wave_interference.png', alt: 'Wave Interference (Module 6.4.2)' },
        { src: '_static/carousel/m6.4.2_wave_patterns.png', alt: 'Wave Patterns (Module 6.4.2)' },

        // Module 8: Animation & Time
        { src: '_static/carousel/m8.4.3_animated_fractal.gif', alt: 'Animated Fractal (Module 8.4.3)' },

        // Module 9: Neural Networks
        { src: '_static/carousel/m9.1.3_activation_functions.png', alt: 'Activation Functions (Module 9.1.3)' },

        // Module 10: TouchDesigner Fundamentals
        { src: '_static/carousel/m10.1.1_color_flow.gif', alt: 'TouchDesigner Color Flow (Module 10.1.1)' },
        { src: '_static/carousel/m10.1.1_touchdesigner.gif', alt: 'TouchDesigner Exercise (Module 10.1.1)' },

        // Module 12: Generative AI Models
        { src: '_static/carousel/m12.1.2_dcgan_fabric.gif', alt: 'DCGAN Fabric (Module 12.1.2)' },
        { src: '_static/carousel/m12.1.3_stylegan.gif', alt: 'StyleGAN (Module 12.1.3)' },
        { src: '_static/carousel/m12.1.4_pix2pix.png', alt: 'Pix2Pix (Module 12.1.4)' },
        { src: '_static/carousel/m12.3.1_ddpm_fabric_morph.gif', alt: 'DDPM Fabric Morph (Module 12.3.1)' }
    ];

    // Get the cube gallery container
    const gallery = document.querySelector('.cube-gallery');
    if (!gallery) return;

    // Fisher-Yates shuffle algorithm
    function shuffle(array) {
        const arr = [...array];
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
        return arr;
    }

    // Select 6 random images
    const selected = shuffle(allImages).slice(0, 6);

    // Create and append image elements
    selected.forEach(imgData => {
        const img = document.createElement('img');
        img.src = imgData.src;
        img.alt = imgData.alt;
        img.loading = 'eager'; // Load all cube faces immediately
        gallery.appendChild(img);
    });
});
