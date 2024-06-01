# Human Motion Analysis and Data Processing using Spline Fitting and Singular Value Decomposition
 
In this project, a comprehensive set of numerical analysis tools in Python was developed to process and analyze motion capture data and 3D point cloud data. The project focused on three main tasks:

● Spline Fitting for Noisy Data: Eighteen key points on a human body during actions such as a golf swing were tracked. Given the noisy and incomplete data, a custom piecewise cubic spline fitting algorithm was designed and implemented. This algorithm smooths the raw data and interpolates missing values without relying on pre-built functions like scipy.interpolate or spline(). Accurate reconstruction of motion trajectories from noisy datasets was achieved through this custom approach.

● Principal Component Analysis (PCA) using Singular Value Decomposition (SVD): Using a 3D point cloud of a building captured by a lidar system, PCA was applied to reorient the point cloud data. SVD was implemented from scratch to identify the principal axes of the data and align a new Cartesian reference frame with the shape of the point cloud. This process facilitated more intuitive analysis and manipulation of the 3D structure.

● Image Compression using SVD: A program was developed to perform image compression using Singular Value Decomposition. This technique reduces the dimensionality of image data by decomposing it into its singular values and vectors, effectively reducing storage requirements while maintaining image quality.

Skills: Python (Programming Language) · Numerical Analysis · Principal Component Analysis · Image processing and compression · Machine Learning