from Resnet_feature_extractor import getResNet50Model
import numpy as np
import h5py
import os
from scipy.spatial import distance
from PIL import Image
import matplotlib.pyplot as plt

# Paths
feature_db_path = r"C:\Users\Saikrupa\source\repos\PythonApplication4\PythonApplication4\output\ResnetFeatures.h5"
query_folder = r"C:\Users\Saikrupa\source\repos\PythonApplication4\PythonApplication4\testsmall"
dataset_path = r"C:\Users\Saikrupa\source\repos\PythonApplication4\PythonApplication4\trainsmall"

# Load the features database (HDF5 file)
with h5py.File(feature_db_path, 'r') as h5f:
    feats = h5f['image_features'][:]  # Extract feature vectors
    imgNames = h5f['image_names'][:]  # Extract image names

# Convert image names from bytes to strings if necessary
if imgNames.ndim == 1:
    imgNames = np.array([name.decode('utf-8') for name in imgNames])
else:
    raise ValueError("Dataset 'image_names' is not in the expected format.")

# Initialize ResNet50 model
model = getResNet50Model()

# Function to compute distances and display images
def display_query_and_matches(query_img_path, threshold=5.0, max_matches=20):
    # Extract features for the query image
    X = model.extract_feat(query_img_path)
    
    # Compute Euclidean distance scores
    distances = np.array([distance.euclidean(X, feat) for feat in feats])
    
    # Find images with distances less than the threshold
    mask = distances < threshold
    if not np.any(mask):
        print(f"No matches found for query image '{query_img_path}' with a threshold of {threshold}.")
        return

    # Get matching image names and distances
    matching_indices = np.where(mask)[0]
    imlist = [imgNames[i] for i in matching_indices]
    distancelist = distances[matching_indices]
    
    # Sort matches by distance
    sorted_indices = np.argsort(distancelist)
    imlist = [imlist[i] for i in sorted_indices]
    distancelist = [distancelist[i] for i in sorted_indices]
    
    # Limit the number of matches to display
    if len(imlist) > max_matches:
        imlist = imlist[:max_matches]
        distancelist = distancelist[:max_matches]

    print(f"Query image: {query_img_path}")
    print(f"Found {len(imlist)} images with distance less than {threshold}:")
    for i in range(len(imlist)):
        print(f"Match {i + 1}: {imlist[i]} with distance {distancelist[i]:.4f}")

    # Display the query image and matches
    rows = 4  # Number of rows in the grid
    cols = (len(imlist) + rows - 1) // rows  # Calculate number of columns
    
    plt.figure(figsize=(15, 3 * rows))
    
    # Display query image
    query_img = Image.open(query_img_path).convert('RGB')
    plt.subplot(rows + 1, cols, 1)
    plt.imshow(query_img)
    plt.title("Query")
    plt.axis('off')
    
    # Display matching images in a grid layout
    for i, img_name in enumerate(imlist):
        img_path = os.path.join(dataset_path, img_name)
        if os.path.exists(img_path):
            match_img = Image.open(img_path).convert('RGB')
            plt.subplot(rows + 1, cols, i + 2)
            plt.imshow(match_img)
            plt.title(f"M {i + 1} (D: {distancelist[i]:.4f})")
            plt.axis('off')
        else:
            print(f"Image file '{img_path}' not found.")
    
    plt.tight_layout()
    plt.show()

# Process query images from a directory structure
for subdir, _, files in os.walk(query_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image files
            query_img_path = os.path.join(subdir, file)
            print(f"query: {query_img_path}")
            display_query_and_matches(query_img_path, threshold=0.6, max_matches=25)  # Set threshold and max matches as needed


