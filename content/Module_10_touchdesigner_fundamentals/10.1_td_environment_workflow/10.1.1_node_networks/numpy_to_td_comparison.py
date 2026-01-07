"""
NumPy to TouchDesigner Comparison Script

This educational script demonstrates the conceptual bridge between
NumPy array operations (which you learned in Modules 1-9) and
TouchDesigner node operations.

The goal is to show that:
- NumPy arrays ≈ TOP texture data
- NumPy operations ≈ TOP operators
- Python for-loops ≈ Node networks (but parallel!)

Run this script to see how familiar NumPy code maps to TD concepts.
"""

import numpy as np
from PIL import Image


def numpy_image_pipeline():
    """
    A typical NumPy image processing pipeline.

    In TouchDesigner, this same pipeline would be:
        Noise TOP → Level TOP → Blur TOP → Null TOP

    Each step here corresponds to a node in TD!
    """

    print("=" * 60)
    print("NumPy Image Pipeline vs TouchDesigner Nodes")
    print("=" * 60)

    # Step 1: Generate noise (equivalent to Noise TOP)
    print("\n[NumPy] Step 1: Generate noise")
    print("         noise = np.random.rand(256, 256)")
    print("         TD equivalent: Noise TOP")
    noise = np.random.rand(256, 256)

    # Step 2: Adjust levels (equivalent to Level TOP)
    print("\n[NumPy] Step 2: Adjust brightness/contrast")
    print("         adjusted = noise * 0.5 + 0.25")
    print("         TD equivalent: Level TOP (Brightness/Contrast params)")
    adjusted = noise * 0.5 + 0.25

    # Step 3: Apply blur (equivalent to Blur TOP)
    print("\n[NumPy] Step 3: Simple blur (3x3 averaging)")
    print("         blurred = uniform_filter(adjusted, size=3)")
    print("         TD equivalent: Blur TOP")
    # Simple averaging blur for demonstration
    from scipy.ndimage import uniform_filter
    blurred = uniform_filter(adjusted, size=3)

    # Step 4: Output (equivalent to Null TOP)
    print("\n[NumPy] Step 4: Store result")
    print("         output = blurred")
    print("         TD equivalent: Null TOP (reference node)")
    output = blurred

    # Key difference explanation
    print("\n" + "=" * 60)
    print("KEY DIFFERENCE:")
    print("=" * 60)
    print("""
    NumPy:  Code runs ONCE, top to bottom
            Each line executes and completes
            Result is static until you run again

    TD:     Network runs CONTINUOUSLY (60+ fps)
            All nodes evaluate every frame
            Change any parameter -> instant update!

    The structure is similar, but TD is ALWAYS RUNNING.
    """)

    return output


def show_data_types():
    """
    Demonstrate how data types map between NumPy and TD.
    """

    print("\n" + "=" * 60)
    print("Data Type Mapping: NumPy -> TouchDesigner")
    print("=" * 60)

    mappings = [
        ("2D NumPy array (H x W)",      "TOP", "Grayscale image"),
        ("3D NumPy array (H x W x 3)",  "TOP", "RGB image"),
        ("3D NumPy array (H x W x 4)",  "TOP", "RGBA image"),
        ("1D NumPy array",              "CHOP", "Channel samples"),
        ("Pandas DataFrame",            "DAT", "Table data"),
        ("Python string",               "DAT", "Text data"),
        ("3D point array (N x 3)",      "SOP", "Point cloud"),
    ]

    print(f"\n{'NumPy Type':<30} {'TD Operator':<10} {'Description':<20}")
    print("-" * 60)
    for numpy_type, td_op, desc in mappings:
        print(f"{numpy_type:<30} {td_op:<10} {desc:<20}")

    print("""
    Notice: TOPs handle what you've been doing with NumPy image arrays!
    This is why TOPs feel most familiar coming from Modules 1-9.
    """)


def demonstrate_continuous_vs_static():
    """
    Illustrate the conceptual difference between static and continuous.
    """

    print("\n" + "=" * 60)
    print("Static vs Continuous Evaluation")
    print("=" * 60)

    # Static (NumPy way)
    print("\n--- NumPy (Static) ---")
    print("parameter = 0.5")
    print("result = image * parameter  # Calculates once")
    print("# If you change 'parameter' later, 'result' stays the same!")

    parameter = 0.5
    image = np.ones((10, 10))
    result = image * parameter

    print(f"\nResult mean: {result.mean():.2f}")

    parameter = 0.8  # Change parameter
    print(f"Changed parameter to 0.8")
    print(f"Result mean still: {result.mean():.2f}  <- Doesn't update!")

    # Continuous (TD way - simulated)
    print("\n--- TouchDesigner (Continuous) ---")
    print("Math CHOP outputs: parameter = 0.5")
    print("Level TOP uses parameter for brightness")
    print("# If Math CHOP changes, Level TOP IMMEDIATELY sees new value!")
    print("""
    In TD, changing the Math CHOP to 0.8 would instantly
    change the Level TOP output - no re-running needed!

    This is the power of node-based, real-time programming.
    """)


def generate_comparison_output():
    """
    Generate a visual output showing the NumPy pipeline result.
    """

    print("\n" + "=" * 60)
    print("Generating Visual Comparison Output")
    print("=" * 60)

    # Create a simple noise -> adjust -> blur pipeline
    np.random.seed(42)  # Reproducible

    # Noise generation
    noise = np.random.rand(512, 512)

    # Level adjustment (simulate Level TOP)
    adjusted = np.clip(noise * 1.2 - 0.1, 0, 1)

    # Simple blur (simulate Blur TOP)
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(adjusted, sigma=2)

    # Convert to 8-bit for saving
    output = (blurred * 255).astype(np.uint8)

    # Save result
    img = Image.fromarray(output, mode='L')
    img.save('numpy_pipeline_output.png')

    print("\nSaved: numpy_pipeline_output.png")
    print("""
    This image was created by:
        1. np.random.rand()       -> Like Noise TOP
        2. Array math (* and -)   -> Like Level TOP
        3. gaussian_filter()      -> Like Blur TOP

    In TouchDesigner, this would be 3 nodes connected by wires,
    updating in real-time as you adjust parameters!
    """)


if __name__ == '__main__':
    # Run all demonstrations
    numpy_image_pipeline()
    show_data_types()
    demonstrate_continuous_vs_static()
    generate_comparison_output()

    print("\n" + "=" * 60)
    print("Summary: NumPy -> TouchDesigner Mental Model")
    print("=" * 60)
    print("""
    Key takeaways for Module 10:

    1. NumPy arrays ARE what TOPs process (images as 2D/3D arrays)

    2. NumPy operations BECOME nodes (blur(), adjust -> Blur TOP, Level TOP)

    3. Sequential code BECOMES parallel network (runs every frame)

    4. Variables BECOME wires (data flows through connections)

    5. Re-running script BECOMES unnecessary (TD is always running)

    Your NumPy skills transfer directly to understanding TOPs!
    """)
