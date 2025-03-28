# -*- coding: utf-8 -*-
"""Point Set Alignment.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GAbLtfyM1vraLLBbboqoNAfwIjEtoajE
"""

import numpy as np
import matplotlib.pyplot as plt

# Original points (Source)
source = np.array([
    [1, 1], [2, 3], [3, 5], [4, 7]
])

# Transformed points (Target) with noise
target = np.array([
    [1.1, 1.2], [2.0, 3.1], [3.2, 5.1], [4.1, 7.2]
])

# Calculate transformation using Least Squares
A = np.hstack((source, np.ones((source.shape[0], 1))))
params, _, _, _ = np.linalg.lstsq(A, target, rcond=None)

rotation = params[:2, :2]
translation = params[2, :]

print("Rotation Matrix:\n", rotation)
print("Translation Vector:\n", translation)

# Apply transformation
aligned = (source @ rotation) + translation

# Visualize alignment
plt.scatter(source[:, 0], source[:, 1], color='blue', label='Source')
plt.scatter(target[:, 0], target[:, 1], color='red', label='Target')
plt.scatter(aligned[:, 0], aligned[:, 1], color='green', marker='x', label='Aligned')
plt.legend()
plt.title("Point Set Alignment using Least Squares")
plt.show()