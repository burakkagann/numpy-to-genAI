.. _module-14-1-4-environmental-data:

================================
14.1.4 Environmental Data
================================

:Duration: 35-40 minutes
:Level: Intermediate
:Prerequisites: NumPy fundamentals, basic Python, understanding of APIs

Overview
========

Environmental data offers a rich source of creative material for generative art. Weather patterns, temperature gradients, wind flows, and atmospheric conditions contain inherent visual rhythms that can be transformed into compelling artwork. In this exercise, you will fetch real-time weather data from a public API and transform it into artistic visualizations using NumPy.

This exercise introduces the fundamental workflow of data-driven generative art: acquiring real-world data, processing it into numerical arrays, and mapping those values to visual properties like color, position, and intensity. The techniques learned here form the foundation for more complex data visualizations and demonstrate how the natural world can become a source of algorithmic creativity.

Learning Objectives
-------------------

By the end of this exercise, you will be able to:

* Fetch real-time environmental data from REST APIs using Python
* Parse JSON responses and convert them to NumPy arrays for processing
* Apply data-to-visual mapping strategies (temperature to color, wind to direction)
* Create multi-layered visualizations combining multiple data dimensions


Quick Start: See It In Action
=============================

Run this code to create your first environmental data visualization:

.. code-block:: python
   :caption: Generate a temperature heatmap from live weather data
   :linenos:

   import numpy as np
   from PIL import Image
   import requests
   from scipy.ndimage import zoom

   # Fetch temperature for a single location
   url = "https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&current_weather=true"
   response = requests.get(url)
   data = response.json()
   temp = data['current_weather']['temperature']
   print(f"Current temperature in Berlin: {temp}C")

   # Create a simple temperature-based color
   normalized = (temp + 10) / 50  # Normalize -10 to 40 range
   r = int(min(255, normalized * 2 * 255))
   b = int(min(255, (1 - normalized) * 2 * 255))
   color = (r, 100, b)

   # Generate colored square
   image = np.full((200, 200, 3), color, dtype=np.uint8)
   Image.fromarray(image).save("quick_start.png")

.. figure:: environmental_art.png
   :width: 500px
   :align: center
   :alt: Temperature heatmap showing Europe with blue regions in the north (cold) transitioning to red regions in the south (warm)

   A temperature heatmap generated from live Open-Meteo API data. Cooler temperatures appear blue, warmer temperatures appear red, with smooth interpolation between data points.

The visualization above was created by fetching temperature data for a grid of geographic points across Europe, then mapping each temperature value to a color on a blue-white-red gradient. The smooth appearance comes from interpolating between the discrete data points.


Core Concepts
=============

Concept 1: Fetching Environmental Data from APIs
------------------------------------------------

Modern weather services provide free APIs that return real-time measurements in machine-readable formats. An **API** (Application Programming Interface) allows your Python code to request data from remote servers using HTTP requests [REST2000]_.

The **Open-Meteo API** provides global weather data without requiring registration or API keys, making it ideal for educational projects [OpenMeteo2024]_. A typical API request looks like this:

.. code-block:: python
   :caption: Fetching weather data from Open-Meteo
   :linenos:

   import requests

   # Construct the API URL with parameters
   latitude = 48.85    # Paris
   longitude = 2.35
   url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"

   # Send HTTP GET request
   response = requests.get(url, timeout=5)

   # Parse JSON response
   data = response.json()
   weather = data['current_weather']

   print(f"Temperature: {weather['temperature']}C")
   print(f"Wind Speed: {weather['windspeed']} km/h")
   print(f"Wind Direction: {weather['winddirection']} degrees")

The response contains structured data including temperature, wind speed, wind direction, and weather codes. For generative art, we typically fetch data for multiple locations to create spatial patterns.

.. figure:: api_data_flow_diagram.png
   :width: 700px
   :align: center
   :alt: Diagram showing data pipeline from API request through JSON parsing to NumPy array to color mapping to final PNG image

   The environmental data art pipeline: HTTP requests fetch JSON data, which is parsed into NumPy arrays, mapped to colors, and rendered as images.

.. admonition:: Did You Know?

   The Open-Meteo API aggregates data from national weather services worldwide, including NOAA (USA), DWD (Germany), Meteo France, ECMWF, and many others, providing a unified interface to global weather data without requiring API keys [OpenMeteo2024]_.


Concept 2: Data-to-Visual Mapping Strategies
--------------------------------------------

The core creative challenge in data visualization is choosing how to map data values to visual properties. Edward Tufte, a pioneer in data visualization, emphasizes that effective mappings should reveal patterns while maintaining data integrity [Tufte2001]_.

**Temperature to Color**: The most intuitive mapping for temperature uses a **diverging color scale** with blue for cold, white for neutral, and red for hot. This leverages our cultural associations with color and follows perceptual guidelines for effective map design [Brewer2003]_ [Harrower2003]_.

.. code-block:: python
   :caption: Temperature to color mapping function
   :linenos:
   :emphasize-lines: 6,11,16

   def temperature_to_color(temp, temp_min=-10, temp_max=35):
       """Map temperature to blue-white-red gradient."""
       # Normalize to 0-1 range
       normalized = (temp - temp_min) / (temp_max - temp_min)
       normalized = max(0, min(1, normalized))  # Clamp to valid range

       if normalized < 0.5:
           # Cold: Blue to White
           t = normalized * 2
           r, g, b = int(t * 255), int(t * 255), 255
       else:
           # Hot: White to Red
           t = (normalized - 0.5) * 2
           r, g, b = 255, int((1 - t) * 255), int((1 - t) * 255)

       return (r, g, b)

**Wind to Direction**: Wind data includes both speed (scalar) and direction (angle). Direction maps naturally to arrow orientation, while speed can control arrow length or color intensity.

.. figure:: color_gradient_reference.png
   :width: 500px
   :align: center
   :alt: Color gradient bar showing temperature scale from -10C (blue) through 12C (white) to 35C (red)

   The temperature color scale used in our visualizations. Cold temperatures appear blue, mild temperatures white, and hot temperatures red.

.. important::

   Always **normalize** your data before mapping to visual properties. Raw temperature values might range from -30 to 45 degrees, but color channels require 0-255. Normalization ensures consistent visual output regardless of the actual data range.


Concept 3: Compositing Multi-Dimensional Weather Art
----------------------------------------------------

Real weather systems involve multiple interacting variables: temperature, humidity, wind, pressure, and precipitation. Compelling data art often layers multiple dimensions to create rich, informative visualizations [Cleveland1994]_.

The **layering approach** builds images from bottom to top:

1. **Base layer**: Temperature gradient provides the background color field
2. **Overlay layer**: Wind arrows or flow lines show atmospheric movement
3. **Effect layer**: Humidity affects color saturation or adds cloud-like effects
4. **Detail layer**: Precipitation adds particle effects or texture

.. code-block:: python
   :caption: Multi-layer composition approach
   :linenos:

   # Layer 1: Temperature background
   base_image = create_temperature_heatmap(temperatures)

   # Layer 2: Wind visualization
   base_image = draw_wind_arrows(base_image, wind_speeds, wind_directions)

   # Layer 3: Humidity effect (affects saturation)
   base_image = apply_humidity_saturation(base_image, humidity)

   # Layer 4: Cloud overlay for high humidity
   final_image = add_cloud_effect(base_image, humidity)

Data artists like Jer Thorp and Nicholas Felton have pioneered approaches to transforming personal and environmental data into art, demonstrating that data visualization can be both informative and aesthetically compelling [Thorp2012]_ [Felton2014]_.

.. figure:: weather_dashboard_art.png
   :width: 500px
   :align: center
   :alt: Multi-layered weather visualization showing temperature colors with flowing white wind lines and subtle cloud effects

   A multi-dimensional weather visualization combining temperature (color), wind (flowing lines), humidity (saturation), and clouds (white overlay).


Hands-On Exercises
==================

These exercises follow the PACL scaffolding model: first **Execute** existing code to understand the concepts, then **Modify** parameters to explore variations, and finally **Re-code** from scratch to demonstrate mastery.

Exercise 1: Execute and Explore
-------------------------------

Run the complete environmental art generator script:

.. code-block:: python
   :caption: environmental_art.py
   :linenos:

   import numpy as np
   from PIL import Image
   import requests
   from scipy.ndimage import zoom

   # Configuration
   GRID_SIZE = 10       # 10x10 grid of data points
   IMAGE_SIZE = 512     # Output image size

   # Geographic bounds (Europe)
   LAT_MIN, LAT_MAX = 35.0, 60.0
   LON_MIN, LON_MAX = -10.0, 25.0

   def fetch_temperature_grid(lat_min, lat_max, lon_min, lon_max, grid_size):
       """Fetch temperature data for a grid of geographic points."""
       temperatures = np.zeros((grid_size, grid_size), dtype=np.float32)
       latitudes = np.linspace(lat_max, lat_min, grid_size)
       longitudes = np.linspace(lon_min, lon_max, grid_size)

       for i, lat in enumerate(latitudes):
           for j, lon in enumerate(longitudes):
               url = f"https://api.open-meteo.com/v1/forecast?latitude={lat:.2f}&longitude={lon:.2f}&current_weather=true"
               try:
                   response = requests.get(url, timeout=5)
                   data = response.json()
                   temperatures[i, j] = data['current_weather']['temperature']
               except:
                   temperatures[i, j] = 10  # Fallback value

       return temperatures

   def temperature_to_color(temp, temp_min=-10, temp_max=35):
       """Map temperature to blue-white-red gradient."""
       normalized = np.clip((temp - temp_min) / (temp_max - temp_min), 0, 1)
       if normalized < 0.5:
           t = normalized * 2
           return (int(t * 255), int(t * 255), 255)
       else:
           t = (normalized - 0.5) * 2
           return (255, int((1-t) * 255), int((1-t) * 255))

   def create_heatmap(temperatures, image_size):
       """Create heatmap from temperature grid."""
       scale = image_size / temperatures.shape[0]
       upscaled = zoom(temperatures, scale, order=1)[:image_size, :image_size]

       image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
       temp_min, temp_max = upscaled.min() - 5, upscaled.max() + 5

       for y in range(image_size):
           for x in range(image_size):
               image[y, x] = temperature_to_color(upscaled[y, x], temp_min, temp_max)

       return image

   # Main execution
   print("Fetching weather data...")
   temperatures = fetch_temperature_grid(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, GRID_SIZE)
   print(f"Temperature range: {temperatures.min():.1f}C to {temperatures.max():.1f}C")

   image = create_heatmap(temperatures, IMAGE_SIZE)
   Image.fromarray(image).save("environmental_art.png")
   print("Saved environmental_art.png")

After running the code, answer these reflection questions:

1. How does the temperature distribution reflect the geography of Europe (latitude, coastlines)?
2. Why do we use ``scipy.ndimage.zoom`` instead of simply repeating pixels?
3. What would happen if we changed the color gradient to green-yellow-red instead?

.. dropdown:: Answers and Explanation
   :class-title: sd-font-weight-bold

   1. **Geographic patterns**: The visualization typically shows cooler temperatures in northern Europe (Scandinavia) and warmer temperatures in southern Europe (Mediterranean). Coastal areas often show more moderate temperatures than continental interiors due to ocean thermal regulation.

   2. **Smooth interpolation**: ``scipy.ndimage.zoom`` with ``order=1`` performs bilinear interpolation, creating smooth gradients between data points. Simple pixel repetition would create blocky, pixelated output that doesn't represent the continuous nature of temperature fields.

   3. **Green-yellow-red**: This palette would create a different aesthetic, often used for vegetation health or air quality indices. The visual impact depends on cultural associations; blue-red feels more intuitive for temperature because we associate blue with cold and red with heat.


Exercise 2: Modify Parameters
-----------------------------

Experiment with the visualization by modifying these parameters:

**Goal 1**: Change the geographic region to view different areas

.. code-block:: python
   :caption: Try different regions

   # North America
   LAT_MIN, LAT_MAX = 25.0, 50.0
   LON_MIN, LON_MAX = -125.0, -70.0

   # Asia
   LAT_MIN, LAT_MAX = 20.0, 50.0
   LON_MIN, LON_MAX = 70.0, 140.0

   # Your home region
   # LAT_MIN, LAT_MAX = ?, ?
   # LON_MIN, LON_MAX = ?, ?

**Goal 2**: Add wind visualization as an overlay

Modify the script to fetch wind data and display arrows:

.. code-block:: python
   :caption: Fetch wind data from API

   # In the fetch function, also get wind:
   weather = data['current_weather']
   wind_speed = weather['windspeed']
   wind_direction = weather['winddirection']

.. dropdown:: Hint: Drawing wind arrows
   :class-title: sd-font-weight-bold

   Use the PIL ImageDraw module to draw arrows:

   .. code-block:: python

      from PIL import ImageDraw
      import math

      def draw_arrow(draw, cx, cy, speed, direction):
          length = 10 + speed * 0.5
          angle = math.radians(direction + 180)
          dx = length * math.sin(angle)
          dy = -length * math.cos(angle)
          draw.line([(cx, cy), (cx + dx, cy + dy)], fill=(50, 50, 50), width=2)

**Goal 3**: Change the color palette for different data types

.. code-block:: python
   :caption: Alternative color palettes

   # Precipitation: Blue gradient
   def precipitation_to_color(mm, max_mm=50):
       t = min(1, mm / max_mm)
       return (int((1-t) * 255), int((1-t) * 255), int(200 + t * 55))

   # Humidity: Green gradient
   def humidity_to_color(percent):
       t = percent / 100
       return (int((1-t) * 200), int(100 + t * 155), int((1-t) * 200))

.. dropdown:: Complete Exercise 2 Solution
   :class-title: sd-font-weight-bold

   See the ``wind_overlay.py`` script in this directory for a complete implementation that combines temperature heatmap with wind direction arrows.

   .. figure:: wind_overlay.png
      :width: 400px
      :align: center
      :alt: Temperature heatmap with wind direction arrows overlaid

      Temperature heatmap with wind direction arrows showing atmospheric flow patterns.


Exercise 3: Create Weather Dashboard Art
----------------------------------------

Create your own multi-layered weather visualization from scratch. Your artwork should combine at least two weather variables in a visually cohesive composition.

**Requirements**:

* Fetch at least 2 different weather variables (temperature, wind, humidity, etc.)
* Apply appropriate normalization to each variable
* Use layered composition to combine the data dimensions
* Create a visually appealing output (consider color harmony, visual balance)

**Starter Code**:

.. code-block:: python
   :caption: weather_dashboard_starter.py
   :linenos:

   import numpy as np
   from PIL import Image
   import requests
   from scipy.ndimage import zoom

   GRID_SIZE = 8
   IMAGE_SIZE = 512

   def fetch_weather_data(lat_min, lat_max, lon_min, lon_max, grid_size):
       """
       TODO: Fetch temperature AND one other variable (wind, humidity, etc.)
       Return a dictionary with arrays for each variable.
       """
       data = {
           'temperature': np.zeros((grid_size, grid_size)),
           # TODO: Add another variable
       }

       # TODO: Implement API fetching loop
       # Hint: Add &hourly=relativehumidity_2m to get humidity

       return data

   def create_visualization(data, image_size):
       """
       TODO: Create a multi-layered visualization.
       Layer 1: Temperature background
       Layer 2: Your chosen overlay (wind arrows, humidity effect, etc.)
       """
       image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

       # TODO: Implement your visualization

       return image

   # Main
   weather = fetch_weather_data(40, 55, -5, 20, GRID_SIZE)
   result = create_visualization(weather, IMAGE_SIZE)
   Image.fromarray(result).save("my_weather_art.png")

.. dropdown:: Hint 1: Fetching humidity data
   :class-title: sd-font-weight-bold

   Add ``&hourly=relativehumidity_2m`` to your API URL:

   .. code-block:: python

      url = f"...&current_weather=true&hourly=relativehumidity_2m"
      data = response.json()
      humidity = data['hourly']['relativehumidity_2m'][0]

.. dropdown:: Hint 2: Layering approach
   :class-title: sd-font-weight-bold

   Build your image layer by layer:

   .. code-block:: python

      # Start with temperature background
      image = create_temperature_layer(temps)

      # Modify saturation based on humidity
      image = adjust_saturation(image, humidity)

      # Add wind flow lines on top
      image = draw_wind_lines(image, winds)

.. dropdown:: Complete Solution
   :class-title: sd-font-weight-bold

   See ``weather_dashboard_art.py`` for a complete implementation that combines:

   * Temperature gradient background (sunset color palette)
   * Humidity-based saturation adjustment
   * Flowing wind lines
   * Cloud overlay effect for high humidity areas

   .. code-block:: python
      :linenos:

      # Key components of the solution:

      def temperature_to_color(temp, temp_min=-5, temp_max=20):
          """Sunset palette: blue -> purple -> orange -> red"""
          normalized = np.clip((temp - temp_min) / (temp_max - temp_min), 0, 1)
          if normalized < 0.33:
              t = normalized * 3
              return (int(50 + t * 100), int(50 + t * 50), int(200 - t * 80))
          elif normalized < 0.66:
              t = (normalized - 0.33) * 3
              return (int(150 + t * 80), int(100 + t * 50), int(120 - t * 80))
          else:
              t = (normalized - 0.66) * 3
              return (int(230 + t * 25), int(150 - t * 100), int(40 - t * 40))

      def add_humidity_effect(image, humidity):
          """High humidity = more saturated colors"""
          saturation = 0.5 + (humidity / 100) * 0.5
          gray = np.mean(image, axis=2, keepdims=True)
          return gray + saturation * (image - gray)

**Challenge Extension**: Add temporal animation by fetching the 24-hour forecast and creating a GIF that shows how weather patterns evolve over time.


Summary
=======

In 35 minutes, you have learned how to transform real-world environmental data into generative art.

Key Takeaways
-------------

* **REST APIs** provide access to real-time environmental data in JSON format
* **Data normalization** is essential before mapping values to visual properties
* **Diverging color scales** (blue-white-red) effectively communicate temperature ranges
* **Multi-layer composition** allows combining multiple data dimensions artistically
* **Fallback strategies** ensure code works even when APIs are unavailable
* Environmental data contains inherent visual patterns that can inspire generative art

Common Pitfalls
---------------

* **Rate limiting**: Making too many API requests too quickly can result in blocked requests
* **Timeout handling**: Always use timeouts when making network requests
* **Data range assumptions**: Temperature ranges vary by season and location; use dynamic normalization
* **Color overflow**: Ensure RGB values stay within 0-255 range using ``np.clip``
* **Missing data**: APIs may return incomplete data; always have fallback values


Next Steps
==========

Continue to :doc:`../../14.2_visualization_techniques/14.2.1_network_graphs/README` to learn about visualizing relationships and connections in data, or explore :doc:`../../14.2_visualization_techniques/14.2.4_time_series_art/README` to create temporal visualizations from sequential data.


References
==========

.. [Tufte2001] Tufte, E. R. (2001). *The Visual Display of Quantitative Information* (2nd ed.). Graphics Press. ISBN: 978-0-9613921-4-7 [Classic reference for data visualization principles and best practices]

.. [Brewer2003] Brewer, C. A. (2003). ColorBrewer: Color advice for cartography. *Pennsylvania State University*. https://colorbrewer2.org/ [Tool for selecting effective color schemes for maps and data visualization]

.. [OpenMeteo2024] Open-Meteo. (2024). Open-Meteo Free Weather API Documentation. https://open-meteo.com/en/docs [Free, open-source weather API used in this exercise]

.. [Cleveland1994] Cleveland, W. S. (1994). *The Elements of Graphing Data* (Revised ed.). Hobart Press. ISBN: 978-0963488411 [Scientific visualization principles and perceptual guidelines]

.. [Harrower2003] Harrower, M., & Brewer, C. A. (2003). ColorBrewer.org: An online tool for selecting colour schemes for maps. *The Cartographic Journal*, 40(1), 27-37. https://doi.org/10.1179/000870403235002042 [Research on perceptually-based color selection]

.. [Thorp2012] Thorp, J. (2012). Make data more human. *TED Talk*. https://www.ted.com/talks/jer_thorp_make_data_more_human [Data artist discussing humanizing data visualization]

.. [Felton2014] Felton, N. (2014). *Feltron Annual Report*. http://feltron.com/ [Pioneer of personal data visualization as art form]

.. [REST2000] Fielding, R. T. (2000). Architectural Styles and the Design of Network-based Software Architectures. *Doctoral dissertation, University of California, Irvine*. https://ics.uci.edu/~fielding/pubs/dissertation/fielding_dissertation.pdf [Original REST architecture dissertation]

.. [NumPyDocs] NumPy Developers. (2024). NumPy User Guide. *NumPy Documentation*. https://numpy.org/doc/stable/user/index.html

.. [PillowDocs] Clark, A., et al. (2024). *Pillow: Python Imaging Library* (Version 10.x). https://pillow.readthedocs.io/
