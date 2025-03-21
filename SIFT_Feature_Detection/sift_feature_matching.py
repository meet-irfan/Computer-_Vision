# -*- coding: utf-8 -*-
"""SIFT Feature Matching.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1a27NYgTzODweGT0vIvy3lLvBtfqbwTly
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load two images
img1 = cv2.imread('/content/1696407021566.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/content/WhatsApp Image 2025-03-07 at 19.11.44_1c036be3.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Initialize the SIFT detector
sift = cv2.SIFT_create()

# 2. Detect and compute keypoints and descriptors
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

print(f"Image 1 - {len(keypoints1)} keypoints detected")
print(f"Image 2 - {len(keypoints2)} keypoints detected")

# 3. Match features using Brute-Force Matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# 4. Sort matches by distance (smaller distances indicate better matches)
matches = sorted(matches, key=lambda x: x.distance)

# 5. Draw the top 50 matches
matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(15, 8))
plt.imshow(matched_img)
plt.title('SIFT Feature Matching')
plt.axis('off')
plt.show()