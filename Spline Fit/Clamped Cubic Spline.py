import numpy as np
import matplotlib.pyplot as plt

# Load data from the CSV file
data = np.genfromtxt('c:/Users/harsh/OneDrive/Desktop/Spline Fit/swing_keypoints.csv', delimiter=',', names=True)

# Extract time and x-coordinate values
time = data['time']
x_coords = data['keypoint_4_x']

# Find the indices of NaN values
nan_indices = np.where(np.isnan(x_coords))[0]

# Extract indices surrounding the NaN gap (The gap is between indices 35 and 55)
start_index = nan_indices[0] - 5  # Start a few points before the gap
end_index = nan_indices[-1] + 6   # End a few points after the gap

# Trim the data to only include the range around the NaN gap
time_range = time[start_index:end_index]
x_coords_range = x_coords[start_index:end_index]

# Remove NaN values for the initial guess
valid_indices = ~np.isnan(x_coords_range)
time_range = time_range[valid_indices]
x_coords_range = x_coords_range[valid_indices]

# Calculate the slopes on either side of the gap to set constraints
slope_start = (x_coords_range[1] - x_coords_range[0]) / (time_range[1] - time_range[0])
slope_end = (x_coords_range[-1] - x_coords_range[-2]) / (time_range[-1] - time_range[-2])

# Generate cubic spline coefficients
# Cubic spline equation: a*x^3 + b*x^2 + c*x + d
# Use clamped boundary conditions: set slope_start and slope_end as constraints
a, b, c, d = np.polyfit(time_range, x_coords_range, 3)
coefficients = [a, b, c, d]

# Generate x values for the spline fit
x_fit = np.linspace(time_range[0], time_range[-1], 100)

# Calculate y values using the cubic spline equation
y_fit = np.polyval(coefficients, x_fit)

# Plot the raw data and the cubic spline fit
plt.figure(figsize=(10, 6))
plt.plot(time, x_coords, 'bo', label='Raw Data')  # Plot raw data
plt.plot(time_range, x_coords_range, 'ro', label='Data Around Gap')  # Plot data around gap
plt.plot(x_fit, y_fit, 'g-', label='Cubic Spline Fit')  # Plot cubic spline fit
plt.xlabel('Time')
plt.ylabel('X Coordinate')
plt.title('Clamped Cubic Spline Fit to Fill Gap')
plt.legend()
plt.grid(True)
plt.show()
