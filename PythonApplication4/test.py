from Resnet_feature_extractor import getResNet50Model
import numpy as np
import h5py
import os
from scipy.spatial import distance

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

# Performance metrics initialization
TP, FP, FN, TN = 0, 0, 0, 0
threshold = 1.2

# Function to compute distances and print image names with distances less than the threshold
def find_and_print_matches(query_img_path, threshold=0.6):
    global TP, FP, FN, TN

    # Extract features for the query image
    X = model.extract_feat(query_img_path)
    
    # Compute Euclidean distance scores
    distances = [distance.euclidean(X, feat) for feat in feats]
    distances = np.array(distances)
    
    # Find images with distances less than the threshold
    mask = distances < threshold
    matching_indices = np.where(mask)[0]
    
    # Simulated evaluation for performance metrics
    if len(matching_indices) > 0:
        TP += 1  # Assuming every close match is a true positive for simulation
        FP += len(matching_indices) - 1  # Extra matches assumed as false positives
    else:
        FN += 1  # No match found where there should be one
    
    TN += len(distances) - len(matching_indices) - 1  # Remaining images assumed as true negatives

    print(f"\nQuery image: {query_img_path}")
    print(f"Found {len(matching_indices)} images with distance less than {threshold}:")

    # Print matching image names and their corresponding distances
    for i in matching_indices:
        print(f"Match: {imgNames[i]} with distance {distances[i]:.4f}")

# Iterate through query images in the query folder
for subdir, _, files in os.walk(query_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image files
            query_img_path = os.path.join(subdir, file)
            print(f"Processing query image: {query_img_path}")
            find_and_print_matches(query_img_path, threshold=threshold)

# Compute precision, recall, and accuracy
precision = TP / (TP + FP) if TP + FP > 0 else 0
recall = TP / (TP + FN) if TP + FN > 0 else 0
accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN > 0 else 0

print("\nPerformance Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")


