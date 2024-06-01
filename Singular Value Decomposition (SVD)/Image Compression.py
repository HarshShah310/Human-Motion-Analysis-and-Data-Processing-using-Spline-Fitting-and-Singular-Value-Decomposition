import numpy as np
from PIL import Image

def svd_image_compression(image_path, k):
    # Load the image
    image = Image.open(image_path)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # Convert image to numpy array
    A = np.array(image)
    
    # Perform Singular Value Decomposition
    U, S, V = np.linalg.svd(A, full_matrices=False)
    
    # Keep only the top k singular values to compress the image
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    V_k = V[:k, :]
    
    # Reconstruct the compressed image
    compressed_image = np.dot(U_k, np.dot(S_k, V_k))
    
    # Convert the reconstructed image to uint8
    compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)
    
    # Create a PIL Image object from the reconstructed image array
    compressed_image = Image.fromarray(compressed_image)
    
    return compressed_image

image_path = "C:/Users/harsh/OneDrive/Desktop/Singular Value Decomposition (SVD)/gray_cat.jpg"
k = 50  # Number of singular values to keep for compression
compressed_image = svd_image_compression(image_path, k)
compressed_image.show()
