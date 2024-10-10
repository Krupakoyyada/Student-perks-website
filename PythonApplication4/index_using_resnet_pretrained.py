import os
import h5py
import numpy as np
from Resnet_feature_extractor import getResNet50Model

# Define paths
images_path = r"C:\Users\Saikrupa\source\repos\PythonApplication4\PythonApplication4\trainsmall"  # Path to the folder with subfolders of images
output_folder = r"C:\Users\Saikrupa\source\repos\PythonApplication4\PythonApplication4\output"  # Path to the output folder

print("Start feature extraction")

# Load the pre-trained ResNet model
model = getResNet50Model()

# Initialize lists to store features and names
feats = []
names = []

# Traverse through each subfolder and image file
for person_name in os.listdir(images_path):
    person_folder = os.path.join(images_path, person_name)
    
    if os.path.isdir(person_folder):  # Check if it is a directory (i.e., a person folder)
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image files
                print(f"Extracting features from image - {img_name} in folder - {person_name}")
                
                # Extract features from the image
                X = model.extract_feat(img_path)
                
                # Append features and names with subfolder information
                feats.append(X)
                names.append(os.path.join(person_name, img_name))

# Convert features list to a NumPy array
feats = np.array(feats)

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Define output path for storing extracted features
output_file = os.path.join(output_folder, "ResnetFeatures.h5")

print("Writing feature extraction results to h5 file")

# Save features and names to an H5 file with descriptive dataset names
with h5py.File(output_file, 'w') as h5f:
    h5f.create_dataset('image_features', data=feats)
    h5f.create_dataset('image_names', data=np.string_(names))
    
print("Feature extraction and saving completed.")


