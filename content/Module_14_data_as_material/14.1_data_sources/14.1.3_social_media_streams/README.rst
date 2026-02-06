.. _module-14-1-3-social-media-streams:

================================
14.1.3 Social Media Streams
================================

:Duration: 20-25 minutes
:Level: Intermediate

Overview
========

Social media platforms generate vast amounts of data every second, from tweets and posts to likes and shares. This data represents a rich source of material for generative art, allowing artists to transform the pulse of online communities into visual experiences. In this exercise, you will learn to process and visualize social media data, creating artwork that reflects collective human expression.

This exercise uses simulated social media data that mirrors the structure of real API responses. This approach allows you to focus on data processing and visualization techniques without requiring API authentication, while the skills you develop transfer directly to working with live social media streams.

Learning Objectives
-------------------

By the end of this exercise, you will be able to:

* Understand the structure of social media data (JSON format, timestamps, engagement metrics)
* Extract and process text content from social media posts
* Create word cloud visualizations that reveal content themes
* Generate temporal visualizations showing activity patterns and engagement trends


Quick Start: See It In Action
=============================

Run this code to generate a word cloud from simulated social media data:

.. code-block:: python
   :caption: Generate a word cloud from social media posts
   :linenos:

   import numpy as np
   import matplotlib.pyplot as plt
   from collections import Counter
   from wordcloud import WordCloud
   import random

   # Generate 50 simulated posts about creative coding
   random.seed(42)
   topics = ["generative art", "creative coding", "data visualization",
             "machine learning", "neural networks", "fractals"]
   posts = [f"Exploring {random.choice(topics)}! #creativecoding" for _ in range(50)]

   # Extract and count words
   words = " ".join(posts).lower().replace("#", " ")
   word_counts = Counter(words.split())

   # Create word cloud
   wordcloud = WordCloud(width=800, height=400, background_color="white")
   wordcloud.generate_from_frequencies(word_counts)

   plt.figure(figsize=(10, 5))
   plt.imshow(wordcloud, interpolation="bilinear")
   plt.axis("off")
   plt.savefig("word_cloud.png", dpi=150)

.. figure:: word_cloud.png
   :width: 600px
   :align: center
   :alt: Word cloud visualization showing frequently used terms in social media posts about creative coding

   Word cloud generated from simulated social media posts about creative coding. Larger words appear more frequently in the dataset.

The word cloud transforms raw text data into an immediate visual representation of community interests. Larger words indicate higher frequency, revealing the dominant themes in the conversation.


Core Concepts
=============

Concept 1: Social Media Data Structures
---------------------------------------

Social media platforms expose data through APIs (Application Programming Interfaces) that return structured data, typically in JSON format [Twitter2024]_. Understanding this structure is essential for processing social media as creative material.

A typical social media post contains several key components:

.. code-block:: python
   :caption: Structure of a social media post

   post = {
       "id": "1234567890",              # Unique identifier
       "created_at": "2024-01-15T14:32:00Z",  # ISO 8601 timestamp
       "text": "Exploring generative art with NumPy! #creativecoding",
       "author": {
           "username": "creative_coder",
           "followers": 1500
       },
       "engagement": {
           "likes": 42,
           "retweets": 12,
           "replies": 5
       },
       "hashtags": ["creativecoding"]
   }

The ``created_at`` field uses ISO 8601 format, which includes the date, time, and timezone. The ``engagement`` metrics provide quantitative measures of how the audience responded to the content. Hashtags serve as user-defined categorization, making them valuable for topic analysis [Bruns2016]_.

.. admonition:: Did You Know?

   The hashtag symbol (#) was first proposed for Twitter in 2007 by Chris Messina, borrowing from IRC chat channels. It was initially dismissed by Twitter but became a defining feature of social media communication [Parker2011]_.


Concept 2: Text Processing for Visualization
---------------------------------------------

Raw social media text requires processing before visualization. This involves several key steps: tokenization (splitting text into words), normalization (converting to lowercase, removing punctuation), and filtering (removing common words that carry little meaning) [Hearst1999]_.

.. code-block:: python
   :caption: Text processing pipeline
   :linenos:
   :emphasize-lines: 11,15,18

   import re
   from collections import Counter

   def process_text(posts, stop_words=None):
       """Extract meaningful words from posts."""
       if stop_words is None:
           stop_words = {"the", "a", "an", "is", "are", "to", "of", "in",
                        "for", "on", "with", "at", "by", "and", "or"}

       all_words = []
       for post in posts:
           # Convert to lowercase and remove hashtag symbols
           text = post["text"].lower().replace("#", " ")

           # Extract only alphabetic words (3+ characters)
           words = re.findall(r'\b[a-z]{3,}\b', text)

           # Filter out stop words
           filtered = [w for w in words if w not in stop_words]
           all_words.extend(filtered)

       return Counter(all_words)

* **Line 11**: Normalizes text by converting to lowercase and treating hashtags as regular words
* **Line 15**: Uses a regular expression to extract only alphabetic words of 3 or more characters
* **Line 18**: Removes common stop words that would dominate the visualization

The resulting ``Counter`` object maps each word to its frequency, which can be directly used by word cloud libraries.

.. important::

   Stop word lists should be customized for your domain. Social media text often contains platform-specific terms (like "RT" for retweet) that may or may not be meaningful for your visualization.


Concept 3: Temporal Visualization Techniques
--------------------------------------------

Social media data is inherently temporal, and visualizing activity patterns over time reveals when communities are most engaged [Wattenberg2008]_. Common approaches include:

**Activity Timelines**: Show the volume of posts over time periods (hours, days, weeks).

**Engagement Heatmaps**: Display engagement levels across two dimensions, such as hour of day versus day of week.

.. code-block:: python
   :caption: Creating an activity timeline
   :linenos:

   from datetime import datetime
   from collections import Counter
   import matplotlib.pyplot as plt

   def create_activity_timeline(posts):
       """Count posts by hour of day."""
       hours = []
       for post in posts:
           # Parse ISO 8601 timestamp
           time_str = post["created_at"].replace("Z", "")
           dt = datetime.fromisoformat(time_str)
           hours.append(dt.hour)

       hour_counts = Counter(hours)

       # Plot activity by hour
       plt.figure(figsize=(12, 5))
       plt.bar(range(24), [hour_counts.get(h, 0) for h in range(24)])
       plt.xlabel("Hour of Day")
       plt.ylabel("Number of Posts")
       plt.title("Posting Activity by Hour")
       plt.savefig("activity_timeline.png", dpi=150)

.. figure:: activity_timeline.png
   :width: 600px
   :align: center
   :alt: Bar chart showing posting activity distributed across 24 hours of the day

   Posting activity timeline showing when users are most active. Peaks often correspond to morning commutes, lunch breaks, and evening leisure time.

Temporal patterns in social media data often reflect human behavior cycles: work schedules, time zones, and cultural events [Golder2011]_. These patterns can be transformed into rhythmic visual elements in generative art.


Hands-On Exercises
==================

Exercise 1: Execute and Explore
-------------------------------

Run the complete social media visualization script:

.. code-block:: python
   :caption: social_media_visualization.py
   :linenos:

   import numpy as np
   import matplotlib.pyplot as plt
   from datetime import datetime, timedelta
   from collections import Counter
   import random
   import re
   from wordcloud import WordCloud

   def generate_simulated_posts(num_posts=100, seed=42):
       """Generate simulated social media posts."""
       random.seed(seed)
       np.random.seed(seed)

       topics = ["generative art", "creative coding", "data visualization",
                 "machine learning", "neural networks", "fractals"]
       hashtags_pool = ["creativecoding", "generativeart", "codeart",
                       "dataviz", "python", "aiart", "digitalart"]

       base_time = datetime(2024, 1, 15, 12, 0, 0)
       posts = []

       for i in range(num_posts):
           hours_ago = random.randint(0, 168)
           post_time = base_time - timedelta(hours=hours_ago)

           template = f"Exploring {random.choice(topics)}!"
           post_hashtags = random.sample(hashtags_pool, random.randint(1, 3))
           hashtag_text = " ".join(f"#{tag}" for tag in post_hashtags)

           posts.append({
               "id": str(1000000000 + i),
               "created_at": post_time.isoformat() + "Z",
               "text": f"{template} {hashtag_text}",
               "likes": int(np.random.pareto(1.5) * 10),
               "hashtags": post_hashtags
           })
       return posts

   # Generate data and create visualizations
   posts = generate_simulated_posts(100)

   # Word cloud
   text = " ".join(p["text"] for p in posts).lower().replace("#", " ")
   words = re.findall(r'\b[a-z]{3,}\b', text)
   stop_words = {"the", "and", "for", "with", "exploring"}
   word_counts = Counter(w for w in words if w not in stop_words)

   wc = WordCloud(width=800, height=600, background_color="white", colormap="viridis")
   wc.generate_from_frequencies(word_counts)

   plt.figure(figsize=(10, 7.5))
   plt.imshow(wc, interpolation="bilinear")
   plt.axis("off")
   plt.title("Social Media Content Themes", fontsize=16, fontweight="bold")
   plt.savefig("word_cloud.png", dpi=150, bbox_inches="tight")
   plt.close()
   print("Generated word_cloud.png")

After running the code, answer these reflection questions:

1. Which topics appear most prominently in the word cloud?
2. How do the hashtags compare to the general word frequencies?
3. What does the distribution of likes tell you about engagement patterns?

.. dropdown:: Answers and Explanation
   :class-title: sd-font-weight-bold

   1. **Topic prominence**: Terms like "generative", "creative", "coding", and "art" should appear largest, reflecting the domain focus of the simulated data. The specific ranking depends on the random selection in post generation.

   2. **Hashtag comparison**: Hashtags are more standardized than general text. While "creativecoding" appears as a hashtag, the words "creative" and "coding" appear separately in the main text, affecting their relative sizes.

   3. **Engagement distribution**: The ``np.random.pareto(1.5)`` function generates a power-law distribution, where most posts receive few likes but some receive many. This mimics real social media engagement patterns where viral content is rare but impactful.


Exercise 2: Modify Parameters
-----------------------------

Experiment with the visualization by modifying these parameters.

**Goal 1**: Change the word cloud color scheme

Modify the ``colormap`` parameter to create different visual styles:

.. code-block:: python
   :caption: Different color schemes

   # Cool blues
   wc = WordCloud(colormap="Blues")

   # Warm oranges
   wc = WordCloud(colormap="Oranges")

   # Rainbow gradient
   wc = WordCloud(colormap="rainbow")

   # Grayscale
   wc = WordCloud(colormap="gray")

.. dropdown:: Available colormaps
   :class-title: sd-font-weight-bold

   Matplotlib provides many colormaps. Some creative options include:

   - ``viridis``, ``plasma``, ``inferno``: Perceptually uniform
   - ``twilight``, ``twilight_shifted``: Circular colormaps
   - ``cool``, ``hot``: Temperature-based
   - ``Spectral``: Full rainbow spectrum

**Goal 2**: Add custom stop words

Create a domain-specific stop word filter:

.. code-block:: python
   :caption: Custom stop words for creative coding content

   stop_words = {
       # Standard English stop words
       "the", "a", "an", "is", "are", "to", "of", "in", "for",
       # Social media specific
       "just", "today", "day", "check", "out",
       # Domain specific (if you want to hide common terms)
       "art", "creative"  # Remove these to see other themes emerge
   }

**Goal 3**: Filter posts by hashtag before visualization

.. code-block:: python
   :caption: Filter by hashtag

   # Only include posts with #creativecoding
   filtered_posts = [p for p in posts if "creativecoding" in p["hashtags"]]

   # Or exclude certain hashtags
   filtered_posts = [p for p in posts if "aiart" not in p["hashtags"]]

.. dropdown:: Hint: Combining filters
   :class-title: sd-font-weight-bold

   You can combine multiple filters using logical operators:

   .. code-block:: python

      # Posts with creativecoding OR generativeart
      filtered = [p for p in posts
                  if "creativecoding" in p["hashtags"]
                  or "generativeart" in p["hashtags"]]


Exercise 3: Create a Hashtag Co-occurrence Visualization
--------------------------------------------------------

Create a visualization showing which hashtags frequently appear together in the same posts. This reveals thematic clusters in the social media conversation.

**Requirements**:

* Extract all pairs of hashtags that appear in the same post
* Count how often each pair co-occurs
* Visualize as a heatmap showing co-occurrence frequencies

**Starter Code**:

.. code-block:: python
   :caption: hashtag_cooccurrence.py (complete the TODO sections)
   :linenos:

   import numpy as np
   import matplotlib.pyplot as plt
   from collections import Counter
   from itertools import combinations

   # Sample posts with hashtags
   posts = [
       {"hashtags": ["creativecoding", "generativeart", "python"]},
       {"hashtags": ["creativecoding", "dataviz"]},
       {"hashtags": ["generativeart", "aiart", "digitalart"]},
       {"hashtags": ["python", "dataviz", "creativecoding"]},
       {"hashtags": ["aiart", "generativeart"]},
   ]

   # TODO 1: Get all unique hashtags
   all_hashtags = set()
   for post in posts:
       # Add each hashtag to the set
       ...

   hashtag_list = sorted(all_hashtags)
   n = len(hashtag_list)

   # TODO 2: Create co-occurrence matrix
   cooccurrence = np.zeros((n, n), dtype=int)

   for post in posts:
       # Get all pairs of hashtags in this post
       tags = post["hashtags"]
       for tag1, tag2 in combinations(tags, 2):
           # Find indices
           i = hashtag_list.index(tag1)
           j = hashtag_list.index(tag2)
           # TODO: Increment both [i,j] and [j,i] for symmetry
           ...

   # TODO 3: Create heatmap visualization
   plt.figure(figsize=(8, 8))
   # Use plt.imshow() to create the heatmap
   ...
   plt.xticks(range(n), hashtag_list, rotation=45, ha="right")
   plt.yticks(range(n), hashtag_list)
   plt.title("Hashtag Co-occurrence Matrix")
   plt.tight_layout()
   plt.savefig("hashtag_cooccurrence.png", dpi=150)

.. dropdown:: Hint 1: Collecting unique hashtags
   :class-title: sd-font-weight-bold

   Use set update to add multiple items:

   .. code-block:: python

      for post in posts:
          all_hashtags.update(post["hashtags"])

.. dropdown:: Hint 2: Incrementing the matrix
   :class-title: sd-font-weight-bold

   For a symmetric matrix, increment both positions:

   .. code-block:: python

      cooccurrence[i, j] += 1
      cooccurrence[j, i] += 1

.. dropdown:: Complete Solution
   :class-title: sd-font-weight-bold

   .. code-block:: python
      :linenos:

      import numpy as np
      import matplotlib.pyplot as plt
      from itertools import combinations

      posts = [
          {"hashtags": ["creativecoding", "generativeart", "python"]},
          {"hashtags": ["creativecoding", "dataviz"]},
          {"hashtags": ["generativeart", "aiart", "digitalart"]},
          {"hashtags": ["python", "dataviz", "creativecoding"]},
          {"hashtags": ["aiart", "generativeart"]},
      ]

      # Get unique hashtags
      all_hashtags = set()
      for post in posts:
          all_hashtags.update(post["hashtags"])

      hashtag_list = sorted(all_hashtags)
      n = len(hashtag_list)

      # Create co-occurrence matrix
      cooccurrence = np.zeros((n, n), dtype=int)

      for post in posts:
          tags = post["hashtags"]
          for tag1, tag2 in combinations(tags, 2):
              i = hashtag_list.index(tag1)
              j = hashtag_list.index(tag2)
              cooccurrence[i, j] += 1
              cooccurrence[j, i] += 1

      # Visualize
      plt.figure(figsize=(8, 8))
      plt.imshow(cooccurrence, cmap="YlOrRd")
      plt.colorbar(label="Co-occurrence Count")
      plt.xticks(range(n), hashtag_list, rotation=45, ha="right")
      plt.yticks(range(n), hashtag_list)
      plt.title("Hashtag Co-occurrence Matrix", fontsize=14, fontweight="bold")
      plt.tight_layout()
      plt.savefig("hashtag_cooccurrence.png", dpi=150)
      print("Saved hashtag_cooccurrence.png")

   .. figure:: hashtag_frequency.png
      :width: 500px
      :align: center
      :alt: Bar chart showing frequency of top hashtags in social media posts

      Hashtag frequency analysis reveals the most common topics in the social media stream.

**Challenge Extension**: Modify the visualization to show a network graph instead of a heatmap, where hashtags are nodes and edges represent co-occurrence strength.


Summary
=======

Key Takeaways
-------------

* **Social media data** is structured in JSON format with consistent fields for content, timestamps, and engagement metrics
* **Text processing** involves tokenization, normalization, and stop word filtering to extract meaningful content
* **Word clouds** transform word frequency data into immediate visual representations of community interests
* **Temporal visualizations** reveal when communities are most active, often reflecting human behavior patterns
* **Simulated data** allows learning data processing techniques without requiring API authentication
* **Hashtags** serve as user-defined categorization and provide valuable signals for topic analysis

Common Pitfalls
---------------

* **Ignoring time zones**: Social media timestamps are typically in UTC; convert appropriately for local analysis
* **Overloaded stop words**: Removing too many words can eliminate meaningful content; tailor lists to your domain
* **API rate limits**: Real social media APIs have strict rate limits; cache data and handle errors gracefully
* **Data bias**: Simulated data may not capture the full complexity of real social media conversations
* **Privacy considerations**: Real social media data may contain personal information; handle responsibly [Tufekci2014]_


References
==========

.. [Twitter2024] Twitter, Inc. (2024). Twitter API Documentation. *Twitter Developer Platform*. https://developer.twitter.com/en/docs

.. [Bruns2016] Bruns, A., & Burgess, J. (2016). Twitter hashtags from ad hoc to calculated publics. In N. Rambukkana (Ed.), *Hashtag Publics: The Power and Politics of Discursive Networks* (pp. 13-28). Peter Lang. https://doi.org/10.3726/978-1-4331-3197-5

.. [Parker2011] Parker, A. (2011, June 10). Twitter's secret handshake. *The New York Times*. https://www.nytimes.com/2011/06/12/fashion/hashtags-a-new-way-for-tweets-cultural-studies.html

.. [Hearst1999] Hearst, M. A. (1999). Untangling text data mining. In *Proceedings of the 37th Annual Meeting of the Association for Computational Linguistics* (pp. 3-10). ACL. https://doi.org/10.3115/1034678.1034679

.. [Wattenberg2008] Viegas, F. B., Wattenberg, M., & Feinberg, J. (2008). Participatory visualization with Wordle. *IEEE Transactions on Visualization and Computer Graphics*, 15(6), 1137-1144. https://doi.org/10.1109/TVCG.2009.171

.. [Golder2011] Golder, S. A., & Macy, M. W. (2011). Diurnal and seasonal mood vary with work, sleep, and daylength across diverse cultures. *Science*, 333(6051), 1878-1881. https://doi.org/10.1126/science.1202775

.. [Tufekci2014] Tufekci, Z. (2014). Big questions for social media big data: Representativeness, validity and other methodological pitfalls. In *Proceedings of the 8th International AAAI Conference on Weblogs and Social Media*. https://ojs.aaai.org/index.php/ICWSM/article/view/14517

.. [NumPyDocs] NumPy Developers. (2024). NumPy array creation routines. *NumPy Documentation*. https://numpy.org/doc/stable/reference/routines.array-creation.html

.. [MatplotlibDocs] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90-95. https://doi.org/10.1109/MCSE.2007.55

.. [WordcloudDocs] Mueller, A. (2024). *wordcloud: A little word cloud generator* (Version 1.9.3). https://github.com/amueller/word_cloud


Next Steps
==========

Continue to :doc:`../14.1.4_environmental_data/README` to learn about using weather, seismic, and astronomical data as material for generative art, or explore :doc:`../../14.2_visualization_techniques/14.2.1_network_graphs/README` to create more sophisticated visualizations of relational data.
