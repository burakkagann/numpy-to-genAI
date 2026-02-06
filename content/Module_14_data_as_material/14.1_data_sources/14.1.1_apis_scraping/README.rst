.. _module-14-1-1-apis-scraping:

================================
14.1.1 APIs and Data Scraping
================================

:Duration: 30-35 minutes
:Level: Intermediate

Overview
========

APIs (Application Programming Interfaces) allow your code to request data from remote servers, transforming the entire internet into a source of creative material. In this exercise, you will fetch real-time asteroid data from NASA's Near Earth Object Web Service and transform it into abstract generative art.

This exercise introduces the fundamental workflow for data-driven art: fetch structured data from an API, parse the response, extract meaningful properties, and map those properties to visual elements. Each asteroid passing near Earth becomes a particle in your visualization, with its distance, velocity, and hazard status determining position, color, and appearance.

The result is a unique artwork generated from live scientific data, with the potential to create different visualizations each time you run the code, since different asteroids pass by Earth each day.

Learning Objectives
-------------------

By the end of this exercise, you will be able to:

* Understand REST APIs and how they provide structured data
* Fetch data from web APIs using Python's requests library
* Parse JSON responses and navigate nested data structures
* Transform numerical data into visual parameters (position, size, color)
* Create abstract visualizations from real-world scientific data


Quick Start: See It In Action
=============================

Run this code to fetch asteroid data from NASA and display basic information:

.. code-block:: python
   :caption: Fetch asteroid data from NASA API
   :linenos:

   import requests

   # NASA API endpoint for Near Earth Objects
   url = "https://api.nasa.gov/neo/rest/v1/feed"

   # API parameters (DEMO_KEY works for learning - 30 requests/hour)
   params = {
       "start_date": "2025-12-25",
       "end_date": "2025-12-26",
       "api_key": "DEMO_KEY"
   }

   # Make the API request
   print("Fetching asteroid data from NASA...")
   response = requests.get(url, params=params)
   data = response.json()

   # Display results
   print(f"Total asteroids found: {data['element_count']}")

   # Show first 3 asteroids
   for date, asteroids in data['near_earth_objects'].items():
       print(f"\nAsteroids passing near Earth on {date}:")
       for asteroid in asteroids[:3]:
           name = asteroid['name']
           diameter = asteroid['estimated_diameter']['meters']['estimated_diameter_max']
           hazardous = asteroid['is_potentially_hazardous_asteroid']
           print(f"  - {name}: {diameter:.1f}m {'(HAZARDOUS!)' if hazardous else ''}")

Now run the full visualization script to create abstract art from this data:

.. code-block:: bash

   python asteroid_art.py

:download:`Download asteroid_art.py <asteroid_art.py>`

.. figure:: asteroid_visualization.png
   :width: 600px
   :align: center
   :alt: Abstract orbital visualization showing asteroids as organic blobs radiating from a central Earth, with hazardous asteroids highlighted in red glow

   Abstract visualization of near-Earth asteroids. Each organic blob represents an asteroid, with position showing distance from Earth, color indicating velocity (cyan=slow, white=medium, pink=fast), and soft red glows highlighting potentially hazardous objects. Dense star field adds atmospheric depth.

The visualization transforms raw numbers from NASA's database into an orbital artwork that changes every time new asteroids pass by Earth.


Core Concepts
=============

Concept 1: What is an API?
--------------------------

An **API** (Application Programming Interface) is a structured way for programs to communicate with each other [Fielding2000]_. When you visit a website, your browser receives HTML designed for human viewing. When your code calls an API, it receives structured data designed for programmatic processing.

**REST** (Representational State Transfer) is the most common API architecture on the web. REST APIs use standard HTTP methods and return data in predictable formats, typically JSON. NASA provides numerous free REST APIs for accessing space data [NASAOpenAPIs]_.

NASA's Near Earth Object Web Service (NeoWs) provides data about asteroids passing close to Earth [NASANEO2024]_:

.. code-block:: python
   :caption: Understanding API endpoints

   # Base URL - the server address
   base_url = "https://api.nasa.gov/neo/rest/v1/feed"

   # Parameters modify the request
   params = {
       "start_date": "2025-12-25",    # Filter by date range
       "end_date": "2025-12-26",
       "api_key": "DEMO_KEY"          # Authentication (DEMO_KEY is free)
   }

   # The full request URL becomes:
   # https://api.nasa.gov/neo/rest/v1/feed?start_date=2025-12-25&end_date=2025-12-26&api_key=DEMO_KEY

**Key API Concepts**:

* **Endpoint**: The URL path for a specific resource (``/neo/rest/v1/feed`` for asteroid data)
* **Parameters**: Key-value pairs that filter or modify the response
* **API Key**: A credential that identifies your application (rate limits apply)
* **Response**: Structured data returned by the server, usually JSON

.. admonition:: Did You Know?

   NASA tracks over 2,000 potentially hazardous asteroids passing within 7.5 million kilometers of Earth. The data you access through the API is the same data used by planetary defense scientists to monitor potential impact threats [NASANEO2024]_.


Concept 2: Parsing JSON Data
----------------------------

**JSON** (JavaScript Object Notation) is the standard format for API responses [JSONOrg]_. JSON maps directly to Python dictionaries and lists, making it easy to navigate.

NASA's asteroid response has a nested structure:

.. code-block:: python
   :caption: Navigating nested JSON structure
   :linenos:
   :emphasize-lines: 4,8,12

   # The response structure (simplified)
   data = {
       "element_count": 25,
       "near_earth_objects": {
           "2025-12-25": [
               {
                   "name": "(2023 AB)",
                   "estimated_diameter": {
                       "meters": {"estimated_diameter_min": 100, "estimated_diameter_max": 200}
                   },
                   "is_potentially_hazardous_asteroid": False,
                   "close_approach_data": [{
                       "relative_velocity": {"kilometers_per_second": "15.5"},
                       "miss_distance": {"lunar": "10.5"}
                   }]
               }
               # ... more asteroids
           ]
       }
   }

   # Accessing nested data
   for date, asteroids in data["near_earth_objects"].items():
       for asteroid in asteroids:
           # Navigate through nested dictionaries
           velocity = asteroid["close_approach_data"][0]["relative_velocity"]["kilometers_per_second"]
           distance = asteroid["close_approach_data"][0]["miss_distance"]["lunar"]
           hazardous = asteroid["is_potentially_hazardous_asteroid"]

* **Line 4**: ``near_earth_objects`` is a dictionary with dates as keys
* **Line 8**: ``estimated_diameter`` contains nested objects for different units
* **Line 12**: ``close_approach_data`` is a list (asteroids may have multiple approaches)

The ``requests`` library handles JSON parsing automatically:

.. code-block:: python

   response = requests.get(url, params=params)
   data = response.json()  # Automatically parses JSON to Python dict


Concept 3: Data-to-Art Transformation
-------------------------------------

The creative challenge is mapping raw numbers to visual properties [Tufte2001]_. Each asteroid has numerical attributes that can become aesthetic parameters:

.. list-table:: Data Mapping Strategy
   :widths: 25 25 50
   :header-rows: 1

   * - Data Property
     - Visual Property
     - Mapping Logic
   * - Distance (lunar)
     - Radial position
     - Closer asteroids near center, distant ones at edges
   * - Diameter (meters)
     - Shape size
     - Larger asteroids = larger organic blobs
   * - Velocity (km/s)
     - Color
     - Cool Nebula palette: cyan=slow, white=medium, pink=fast
   * - Hazard status
     - Glow effect
     - Soft multi-layer red glow for potentially dangerous objects

**Normalization** converts raw values to usable visual ranges:

.. code-block:: python
   :caption: Normalizing data for visualization
   :linenos:

   def normalize(value, min_val, max_val, target_min=0, target_max=1):
       """Scale a value from its original range to a target range."""
       if max_val == min_val:
           return (target_min + target_max) / 2
       normalized = (value - min_val) / (max_val - min_val)
       return target_min + normalized * (target_max - target_min)

   # Example: Map velocity (3-30 km/s) to color intensity (0-255)
   velocity = 15.5  # km/s
   color_intensity = normalize(velocity, 3, 30, 0, 255)
   # Result: ~128 (middle of the color range)

**Polar coordinates** create the orbital aesthetic:

.. code-block:: python
   :caption: Converting polar to Cartesian coordinates

   import math

   # Asteroid position in polar coordinates
   radius = 200      # Distance from center (pixels)
   angle = 45        # Degrees

   # Convert to Cartesian (x, y) for drawing
   x = center_x + radius * math.cos(angle * math.pi / 180)
   y = center_y + radius * math.sin(angle * math.pi / 180)

.. figure:: api_flow_diagram.png
   :width: 700px
   :align: center
   :alt: Flow diagram showing the data pipeline from Python script through NASA API to JSON response to NumPy array to final visualization

   The data-to-art pipeline: each step transforms the data closer to its final visual form.


Hands-On Exercises
==================

Exercise 1: Execute and Explore
-------------------------------

Run the complete asteroid visualization script:

.. code-block:: bash

   python asteroid_art.py

Examine the output and answer these reflection questions:

1. How many asteroids passed near Earth in the date range?
2. What are the minimum and maximum asteroid diameters in the dataset?
3. How many asteroids are classified as potentially hazardous?


.. dropdown:: Answers and Explanation
   :class-title: sd-font-weight-bold

   1. **Asteroid count**: The script reports the total count after fetching. A typical week has 80-150 asteroids passing near Earth.

   2. **Diameter range**: The data ranges are printed during execution. Asteroid diameters typically range from a few meters to over 1km for the largest near-Earth objects.

   3. **Hazardous count**: Potentially hazardous asteroids (PHAs) are those larger than 140m that pass within 7.5 million km of Earth. Usually 5-15 per week.

Exercise 2: Modify the Visualization
------------------------------------

Experiment with the visualization by modifying parameters in ``asteroid_art.py``.

**Goal 1**: Change the date range to get more asteroids

In the ``fetch_asteroid_data()`` function, modify the date range:

.. code-block:: python
   :caption: Fetching more days of data

   # Original: 7 days
   end_date = datetime.now()
   start_date = end_date - timedelta(days=6)

   # Modified: 14 days (more asteroids!)
   start_date = end_date - timedelta(days=13)

**Goal 2**: Create a "danger mode" color scheme

In the ``create_orbital_visualization()`` function, change how colors are assigned:

.. code-block:: python
   :caption: Modified color mapping

   # Original: uses velocity_to_color() for Cool Nebula palette (cyan->white->pink)
   color = velocity_to_color(velocity_ratio)

   # Danger mode: hazardous = bright red, others = calm cyan
   if asteroid["is_hazardous"]:
       color = (255, 50, 50)
   else:
       color = (100, 200, 220)

.. figure:: hazard_highlight.png
   :width: 500px
   :align: center
   :alt: Visualization with hazardous asteroids prominently displayed in bright red with glowing effects

   Danger mode: Hazardous asteroids stand out with intense red coloring while safe objects fade to blue.

**Goal 3**: Experiment with different color schemes

.. figure:: velocity_variations.png
   :width: 600px
   :align: center
   :alt: 2x2 grid showing four different color mapping schemes for the asteroid visualization

   Different color mappings create distinct visual styles from the same data.

.. dropdown:: Hint: Modifying visual properties
   :class-title: sd-font-weight-bold

   The key variables to modify are in the ``create_orbital_visualization()`` function:

   * ``radius_ratio``: Controls how distance maps to radial position
   * ``size``: Controls circle size (based on diameter)
   * ``red``, ``green``, ``blue``: Control color values (0-255)
   * ``glow_size``: Controls the glow effect size for hazardous asteroids


Exercise 3: Create Your Own Space Art
-------------------------------------

Complete the starter code to create your own asteroid visualization from scratch.

**Requirements**:

* Extract asteroid velocities and distances from the NASA data
* Normalize values to appropriate visual ranges
* Create an abstract visualization using polar coordinates
* Highlight potentially hazardous asteroids

**Starter Code**:

.. code-block:: python
   :caption: asteroid_starter.py - complete the TODO sections
   :linenos:

   import numpy as np
   from PIL import Image, ImageDraw
   import json
   import math

   def fetch_asteroid_data():
       """Load cached asteroid data (provided for you)."""
       with open("asteroid_data.json", 'r') as f:
           return json.load(f)

   def normalize(value, min_val, max_val, target_min=0, target_max=1):
       """Normalize a value to a target range (provided for you)."""
       if max_val == min_val:
           return (target_min + target_max) / 2
       return target_min + (value - min_val) / (max_val - min_val) * (target_max - target_min)

   def transform_to_art(data, canvas_size=800):
       """Transform asteroid data into abstract art."""
       canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
       canvas[:, :] = [10, 10, 25]
       image = Image.fromarray(canvas)
       draw = ImageDraw.Draw(image)
       center = canvas_size // 2

       # TODO Step 1: Extract asteroid velocities, distances, and hazard flags
       # Loop through data['near_earth_objects'] items

       # TODO Step 2: Calculate min/max for normalization

       # TODO Step 3: Draw each asteroid using polar coordinates

       return np.array(image)

   data = fetch_asteroid_data()
   art = transform_to_art(data)
   Image.fromarray(art).save("my_space_art.png")

.. dropdown:: Hint 1: Extracting data from nested JSON
   :class-title: sd-font-weight-bold

   The near_earth_objects dictionary has dates as keys:

   .. code-block:: python

      for date, asteroid_list in data["near_earth_objects"].items():
          for asteroid in asteroid_list:
              velocity = float(asteroid["close_approach_data"][0]
                              ["relative_velocity"]["kilometers_per_second"])
              distance = float(asteroid["close_approach_data"][0]
                              ["miss_distance"]["lunar"])

.. dropdown:: Hint 2: Converting polar to Cartesian
   :class-title: sd-font-weight-bold

   Use the asteroid name hash for consistent angle placement:

   .. code-block:: python

      angle = (hash(asteroid_name) % 360) * (math.pi / 180)
      x = center + int(radius * math.cos(angle))
      y = center + int(radius * math.sin(angle))

.. dropdown:: Complete Solution
   :class-title: sd-font-weight-bold

   .. code-block:: python
      :linenos:

      def velocity_to_color(vel_ratio):
          """Cool Nebula palette: Cyan -> White -> Pink."""
          if vel_ratio < 0.5:
              t = vel_ratio * 2
              return (int(100+t*120), int(220+t*20), 255)
          t = (vel_ratio - 0.5) * 2
          return (int(220+t*35), int(240-t*60), int(255-t*35))

      def transform_to_art(data, canvas_size=800):
          canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
          canvas[:, :] = [10, 10, 25]
          image = Image.fromarray(canvas)
          draw = ImageDraw.Draw(image)
          center = canvas_size // 2

          # Step 1: Extract data
          velocities, distances, hazards, names = [], [], [], []
          for date, asteroid_list in data["near_earth_objects"].items():
              for asteroid in asteroid_list:
                  approach = asteroid["close_approach_data"][0]
                  velocities.append(float(approach["relative_velocity"]["kilometers_per_second"]))
                  distances.append(float(approach["miss_distance"]["lunar"]))
                  hazards.append(asteroid["is_potentially_hazardous_asteroid"])
                  names.append(asteroid["name"])

          # Step 2: Calculate ranges
          min_vel, max_vel = min(velocities), max(velocities)
          min_dist, max_dist = min(distances), max(distances)

          # Draw Earth (green, distinct from asteroids)
          draw.ellipse([center-16, center-16, center+16, center+16], fill=(30,60,45))
          draw.ellipse([center-12, center-12, center+12, center+12], fill=(100,200,150))

          # Step 3: Draw asteroids using Cool Nebula palette
          for i in range(len(velocities)):
              radius = normalize(distances[i], min_dist, max_dist, 0.2, 0.9) * (center-20)
              angle = (hash(names[i]) % 360) * (math.pi / 180)
              x = center + int(radius * math.cos(angle))
              y = center + int(radius * math.sin(angle))

              vel_ratio = normalize(velocities[i], min_vel, max_vel, 0, 1)
              color = velocity_to_color(vel_ratio)

              if hazards[i]:
                  draw.ellipse([x-10, y-10, x+10, y+10], fill=(80,20,20))
                  color = (255, 120, 130)

              draw.ellipse([x-5, y-5, x+5, y+5], fill=color)

          return np.array(image)

**Challenge Extension**: Combine data from multiple days to show asteroid "trails" or create an animated visualization showing asteroids moving through time.


Challenge: Add a Star Field Background
--------------------------------------

Enhance your visualization by adding a background of real stars, using the same
integer array indexing technique you learned in Module 2.1.4 (Creating Star Fields).

.. figure:: star_background_challenge.png
   :width: 600px
   :align: center
   :alt: Asteroid visualization with dense star field background, Earth shown in green at center

   Expected output: Asteroids with Cool Nebula colors overlaid on a dense star field.
   The procedural stars create a "milky way" atmospheric effect while catalog stars
   from bright_stars.json add real astronomical positions.

**Goal**: Combine the asteroid data visualization with a star field background
to create a more immersive "view from space" aesthetic.

**What You Will Learn**:

* Apply the integer array indexing technique from Module 2.1.4 to real astronomical data
* Convert celestial coordinates (RA/Dec) to canvas pixel positions
* Create visual hierarchy by layering (stars first, asteroids on top)
* Map star magnitude to brightness values
* Combine procedural and catalog-based stars for a dense "milky way" effect

**Requirements**:

1. Load the bundled star catalog (``bright_stars.json``)
2. Convert star positions (RA/Dec) to canvas coordinates
3. Draw stars with brightness based on magnitude (brighter stars = higher values)
4. Keep stars dimmer than asteroids to maintain visual hierarchy
5. Draw stars BEFORE asteroids so asteroids appear on top
6. **BONUS**: Add procedural random stars for a denser atmospheric effect

:download:`Download starter code <star_background_starter.py>`

:download:`Download star catalog <bright_stars.json>`

**Starter Code**:

.. code-block:: python
   :caption: star_background_starter.py

   import numpy as np
   from PIL import Image
   import json

   def load_stars():
       """Load bundled star catalog."""
       with open("bright_stars.json", 'r') as f:
           return json.load(f)["stars"]

   def add_star_background(canvas, stars):
       """
       Add stars to canvas using integer array indexing.

       This technique is the same as Module 2.1.4 (Creating Star Fields):
       canvas[y_coords, x_coords] = brightness
       """
       height, width = canvas.shape[:2]

       for star in stars:
           # TODO: Convert RA (0-360) to x coordinate (0-width)
           # TODO: Convert Dec (-90 to +90) to y coordinate (0-height)
           # TODO: Convert magnitude to brightness (brighter = lower magnitude)
           #       Magnitude range: -1.5 to 4.5 -> brightness: 200 to 50
           pass

       return canvas

.. dropdown:: Hint 1: Why Draw Stars First?
   :class-title: sd-font-weight-bold

   In computer graphics, objects drawn later appear "on top" of earlier objects.
   By drawing stars first (background layer), then Earth and asteroids, we ensure
   the data visualization remains prominent. This is the same layering principle
   used in Module 2.1.4 when creating star fields with foreground elements.

.. dropdown:: Hint 2: Celestial Coordinate System
   :class-title: sd-font-weight-bold

   **Right Ascension (RA)**: Like longitude, ranges 0 to 360 degrees around the celestial sphere.
   Maps directly to horizontal position:

   .. code-block:: python

      x = int((star["ra"] / 360) * width)

   **Declination (Dec)**: Like latitude, ranges -90 (south pole) to +90 (north pole).
   Maps to vertical position (inverted because y=0 is top of image):

   .. code-block:: python

      y = int(((90 - star["dec"]) / 180) * height)

.. dropdown:: Hint 3: Magnitude to Brightness (Inverted Scale)
   :class-title: sd-font-weight-bold

   Star magnitude is counter-intuitive: **brighter stars have LOWER magnitudes**.

   * Sirius (brightest star): magnitude -1.46
   * Faint visible stars: magnitude ~4.5

   To convert to brightness (where higher = brighter):

   .. code-block:: python

      # Invert and normalize: mag -1.5 to 4.5 becomes brightness 200 to 50
      brightness = int(200 - (star["magnitude"] + 1.5) * 25)
      brightness = max(30, min(180, brightness))  # Keep dimmer than asteroids

.. dropdown:: Hint 4: Visual Hierarchy
   :class-title: sd-font-weight-bold

   To keep stars as a subtle background:

   * Limit star brightness to 180 max (asteroids can be 255)
   * Add slight blue tint: ``[brightness, brightness, brightness + 20]``
   * Use integer array indexing for single-pixel stars (vs. ellipses for asteroids)

.. dropdown:: Complete Solution
   :class-title: sd-font-weight-bold

   .. code-block:: python

      def add_star_background(canvas, stars):
          height, width = canvas.shape[:2]

          for star in stars:
              # Convert celestial coordinates to canvas coordinates
              x = int((star["ra"] / 360) * width)
              y = int(((90 - star["dec"]) / 180) * height)

              # Skip if outside canvas
              if not (0 <= x < width and 0 <= y < height):
                  continue

              # Convert magnitude to brightness (inverted scale)
              # Brighter stars have lower magnitudes
              brightness = int(200 - (star["magnitude"] + 1.5) * 25)
              brightness = max(30, min(180, brightness))  # Keep dimmer than asteroids

              # Place star using integer array indexing (Module 2.1.4 technique)
              canvas[y, x] = [brightness, brightness, brightness + 20]  # Slight blue tint

          return canvas

   **Integration into transform_to_art()**:

   .. code-block:: python

      def transform_to_art(data, canvas_size=800):
          canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
          canvas[:, :] = [10, 10, 25]

          # Add star background FIRST (before Earth and asteroids)
          stars = load_stars()
          canvas = add_star_background(canvas, stars)

          # Then create PIL image and draw Earth/asteroids on top
          image = Image.fromarray(canvas)
          draw = ImageDraw.Draw(image)
          # ... rest of visualization code


Summary
=======

Key Takeaways
-------------

* **APIs provide structured data**: REST APIs return JSON that maps directly to Python dictionaries, making data access straightforward
* **DEMO_KEY for learning**: NASA's free demo key allows 30 requests per hour, sufficient for educational purposes
* **Nested data navigation**: API responses often have deeply nested structures requiring careful traversal
* **Normalization is essential**: Raw data values must be scaled to visual ranges (0-255 for colors, pixel coordinates for positions)
* **Polar coordinates create orbital aesthetics**: Converting distance to radius and using hash-based angles creates natural orbital patterns
* **Error handling matters**: APIs can fail due to network issues or rate limiting; always include fallback mechanisms

Common Pitfalls
---------------

* **Forgetting to parse JSON**: Use ``response.json()`` not ``response.text`` to get a Python dictionary
* **Rate limiting**: DEMO_KEY has limits; cache responses to avoid repeated API calls during development
* **Assuming data structure**: Always check that keys exist before accessing nested data (use ``.get()`` with defaults)
* **Integer division for coordinates**: Ensure pixel coordinates are integers when drawing shapes
* **Ignoring edge cases**: Handle empty data sets and min/max edge cases in normalization


References
==========

.. [Fielding2000] Fielding, R. T. (2000). *Architectural styles and the design of network-based software architectures* (Doctoral dissertation). University of California, Irvine. https://www.ics.uci.edu/~fielding/pubs/dissertation/top.htm

.. [NASAOpenAPIs] National Aeronautics and Space Administration. (2024). NASA Open APIs. *NASA API Portal*. Retrieved December 26, 2025, from https://api.nasa.gov/

.. [NASANEO2024] National Aeronautics and Space Administration. (2024). Asteroids - NeoWs (Near Earth Object Web Service). *NASA API Portal*. https://api.nasa.gov/

.. [JSONOrg] JSON. (2024). Introducing JSON. *JSON.org*. Retrieved December 26, 2025, from https://www.json.org/

.. [RequestsDocs] Reitz, K., & Contributors. (2024). *Requests: HTTP for Humans* (Version 2.31.0) [Computer software]. https://docs.python-requests.org/

.. [Tufte2001] Tufte, E. R. (2001). *The Visual Display of Quantitative Information* (2nd ed.). Graphics Press. ISBN: 978-0961392147

.. [NumPyDocs] NumPy Developers. (2024). NumPy documentation. *NumPy.org*. Retrieved December 26, 2025, from https://numpy.org/doc/stable/

.. [PillowDocs] Clark, A., & Contributors. (2024). *Pillow (PIL Fork) Documentation*. https://pillow.readthedocs.io/

.. [Galanter2016] Galanter, P. (2016). Generative art theory. In C. Paul (Ed.), *A Companion to Digital Art* (pp. 146-180). Wiley-Blackwell. https://doi.org/10.1002/9781118475249.ch5

.. [Manovich2020] Manovich, L. (2020). *Cultural Analytics*. MIT Press. ISBN: 978-0262037105
