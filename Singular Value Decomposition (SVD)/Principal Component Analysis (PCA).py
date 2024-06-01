import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import svd

# Load the point cloud data
point_cloud = pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\Singular Value Decomposition (SVD)\PC_sm.csv", header=None, names=['x', 'y', 'z'])

# Convert the data to numpy array
point_cloud_np = point_cloud.values

# Compute the centroid of the point cloud
centroid = np.mean(point_cloud_np, axis=0)

# Center the points by subtracting the centroid
centered_cloud = point_cloud_np - centroid

# Compute SVD of the centered point cloud
U, S, Vt = svd(centered_cloud, full_matrices=False)

# The columns of Vt are the Unit components
unit_axes = Vt

# Creating a 3D plot to visualize the data
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting the original points
ax.scatter(point_cloud_np[:, 0], point_cloud_np[:, 1], point_cloud_np[:, 2], color='blue', alpha=0.5, label='Point Cloud')

# Define a scaling factor for better visualization of the unit axes
scale_factor = 40

# Plotting the Unit axes
for i in range(3):
    ax.quiver(*centroid, *(unit_axes[i, :] * scale_factor), color=['r', 'g', 'b'][i], linewidth=2, label=f'Axis {i+1}')

# Setting labels and legend
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.legend()

# Setting the plot limits
ax.set_xlim(centroid[0] - 50, centroid[0] + 50)
ax.set_ylim(centroid[1] - 50, centroid[1] + 50)
ax.set_zlim(centroid[2] - 50, centroid[2] + 50)

# Add a title and display the plot
plt.title('3D Point Cloud with Unit Axes')
plt.show()