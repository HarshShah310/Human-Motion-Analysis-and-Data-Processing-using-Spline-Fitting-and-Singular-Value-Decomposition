import numpy as np
import matplotlib.pyplot as plt

def splinecoeff(x, y, v1=0, vn=0):
    """
    Compute the coefficients of cubic splines.
    Args:
        x (array): Input x-values.
        y (array): Input y-values.
        v1 (float): First derivative at x[0].
        vn (float): First derivative at x[-1].
    Returns:
        array: Coefficients of cubic splines.
    """
    n = len(x)
    A = np.zeros((n, n))
    r = np.zeros(n)
    dx = np.zeros(n-1)
    dy = np.zeros(n-1)

    for i in range(n-1):
        dx[i] = x[i+1] - x[i]
        dy[i] = y[i+1] - y[i]

    for i in range(1, n-1):
        A[i, i-1:i+2] = [dx[i-1], 2*(dx[i-1] + dx[i]), dx[i]]
        r[i] = 3 * (dy[i]/dx[i] - dy[i-1]/dx[i-1])

    A[0, 0] = 1  # Natural spline conditions
    A[-1, -1] = 1
    coeff = np.zeros((n, 3))
    coeff[:, 1] = np.linalg.solve(A, r)

    for i in range(n-1):
        coeff[i, 2] = (coeff[i+1, 1] - coeff[i, 1]) / (3 * dx[i])
        coeff[i, 0] = dy[i]/dx[i] - dx[i] * (2 * coeff[i, 1] + coeff[i+1, 1]) / 3

    return coeff[:-1, :]  # Remove last row

def splineplot(x, y, k):
    """
    Generate points along the spline curve.
    Args:
        x (array): Input x-values.
        y (array): Input y-values.
        k (int): Number of points to generate between each pair of x-values.
    Returns:
        array, array: Generated x-values and y-values along the spline curve.
    """
    coeff = splinecoeff(x, y)
    x1 = []
    y1 = []

    for i in range(len(x)-1):
        xs = np.linspace(x[i], x[i+1], k+1)
        dx = xs - x[i]
        ys = coeff[i, 2] * dx
        ys = (ys + coeff[i, 1]) * dx
        ys = (ys + coeff[i, 0]) * dx + y[i]
        x1.extend(xs[:k])
        y1.extend(ys[:k])

    x1.append(x[-1])
    y1.append(y[-1])
    return x1, y1

# Load data from the CSV file
data = np.genfromtxt('c:/Users/harsh/OneDrive/Desktop/Spline Fit/swing_keypoints.csv', delimiter=',', skip_header=1)

# Extract relevant columns (time and keypoint_4_x)
time = data[:, 0]
keypoint_4_x = data[:, 1]

# Filter out NaN values
valid_indices = ~np.isnan(keypoint_4_x)
time = time[valid_indices]
keypoint_4_x = keypoint_4_x[valid_indices]

# Sample every 5th point
sampled_indices = np.arange(0, len(time), 5)
time_sampled = time[sampled_indices]
keypoint_4_x_sampled = keypoint_4_x[sampled_indices]

# Generate spline fit
x_plot, y_plot = splineplot(time_sampled, keypoint_4_x_sampled, 100)

# Plot raw data and spline fit
plt.figure(figsize=(10, 6))
plt.plot(time, keypoint_4_x, 'r-', label='Raw Data')  # Red is raw data
plt.plot(x_plot, y_plot, 'g-', label='Spline Fit')  # Green is spline fit
plt.xlabel('Time')
plt.ylabel('Keypoint_4_X')
plt.title('Raw Data and Piecewise Cubic Spline Fit (Keypoint 4)')
plt.legend()
plt.grid(True)
plt.show()