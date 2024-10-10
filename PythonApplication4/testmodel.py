from Resnet_feature_extractor import getResNet50Model
import numpy as np
import h5py
import os
from scipy.spatial import distance
from PIL import Image
import matplotlib.pyplot as plt

# Paths specifying training, testing and extracted features of image folders
feature_db = r"E:\Desktop\final project\PythonApplication4\PythonApplication4\output\ResnetFeatures.h5"
testing = r"E:\Desktop\final project\PythonApplication4\PythonApplication4\testsmall"
training = r"E:\Desktop\final project\PythonApplication4\PythonApplication4\trainsmall"

# Loading database with image features
with h5py.File(feature_db, 'r') as h5f:
    feats = h5f['image_features'][:]  # feature vectors are extracted
    imgNames = h5f['image_names'][:]  # image names are extracted

# Convert image names from bytes to strings 
if imgNames.ndim == 1:
    imgNames = np.array([name.decode('utf-8') for name in imgNames])
else:
    raise ValueError("Dataset 'image_names' is incorrect")

# Initializing ResNet50 model
model = getResNet50Model()

# Displays images and computes euclidean distances
def display_query_and_matches(test_path, threshold=5.0, max_matches=20):
    # features are extracted for test images
    X = model.extract_feat(test_path)
    
    # computing euclidean distance
    distances = np.array([distance.euclidean(X, feat) for feat in feats])
    
    # finding images based on threshold
    mask = distances < threshold
    if not np.any(mask):
        print(f"No matches  '{test_path}' with  {threshold}.")
        return

    # similar image names matching
    matching_indices = np.where(mask)[0]
    imlist = [imgNames[i] for i in matching_indices]
    distancelist = distances[matching_indices]
    
    # distance sorting images
    sorted_indices = np.argsort(distancelist)
    imlist = [imlist[i] for i in sorted_indices]
    distancelist = [distancelist[i] for i in sorted_indices]
    
    # Limit to image matches for clear display
    if len(imlist) > max_matches:
        imlist = imlist[:max_matches]
        distancelist = distancelist[:max_matches]

    print(f"Query: {test_path}")
    print(f"{len(imlist)} images with distance {threshold}:")
    for i in range(len(imlist)):
        print(f"Match {i + 1}: {imlist[i]} with distance {distancelist[i]:.4f}")

    # Display the query image and similar matches 
    rows = 4  # Number of rows in the grid
    cols = (len(imlist) + rows - 1) // rows  # Calculate number of columns
    
    plt.figure(figsize=(15, 3 * rows))
    
    # Display query image
    query_img = Image.open(test_path).convert('RGB')
    plt.subplot(rows + 1, cols, 1)
    plt.imshow(query_img)
    plt.title("Query")
    plt.axis('off')
    
    # Display matching images 
    for i, img_name in enumerate(imlist):
        img_path = os.path.join(training, img_name)
        if os.path.exists(img_path):
            match_img = Image.open(img_path).convert('RGB')
            plt.subplot(rows + 1, cols, i + 2)
            plt.imshow(match_img)
            plt.title(f"M {i + 1} (D: {distancelist[i]:.4f})")
            plt.axis('off')
        else:
            print(f"Image '{img_path}' not found.")
    
    plt.tight_layout()
    plt.show()

# Process query images 
for subdir, _, files in os.walk(testing):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # validating image formats
            test_path = os.path.join(subdir, file)
            print(f"query: {test_path}")
            display_query_and_matches(test_path, threshold=0.6, max_matches=25)  # set threshold and similar matches
