import numpy as np
import h5py
import os
from scipy.spatial import distance
import matplotlib.pyplot as plt

# Paths specifying training folders containing images and features folder
features = r"C:\Users\Saikrupa\source\repos\PythonApplication4\PythonApplication4\output\ResnetFeatures.h5"
training = r"C:\Users\Saikrupa\source\repos\PythonApplication4\PythonApplication4\trainsmall"

# Loading  features database 
with h5py.File(features, 'r') as h5f:
    feats = h5f['image_features'][:]  # Extracting feature vectors
    imgNames = h5f['image_names'][:]  # Extracting image names

# Convert image names from bytes to strings 
if imgNames.ndim == 1:
    imgNames = np.array([name.decode('utf-8') for name in imgNames])
else:
    raise ValueError("Dataset 'image_names' incorrect.")

# Extract class labels from image names 
def class_label(img_name):
    # Assuming image name format: <class_label>_<image_id>.jpg
    return img_name.split('_')[0]

# Compute pairwise distances and categorize
num_img = len(imgNames)
intra = []
inter = []

for i in range(num_img):
    for j in range(i + 1, num_img):
        dist = distance.euclidean(feats[i], feats[j])
        label_i = class_label(imgNames[i])
        label_j = class_label(imgNames[j])
        
        if label_i == label_j:
            intra.append(dist)
        else:
            inter.append(dist)

# Plot the results
plt.figure(figsize=(14, 6))

# plotting Intra-class distances
plt.subplot(1, 2, 1)
plt.hist(intra, bins=50, color='blue', alpha=0.7)
plt.title('Intra-Class Distances')
plt.xlabel('Distance')
plt.ylabel('Frequency')

# plotting Inter-class distances
plt.subplot(1, 2, 2)
plt.hist(inter, bins=50, color='red', alpha=0.7)
plt.title('Inter-Class Distances')
plt.xlabel('Distance')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
