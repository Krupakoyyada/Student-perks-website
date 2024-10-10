import os
import h5py
import numpy as np
from Resnet_feature_extractor import getResNet50Model

# Specifying our folder paths that contains images
training = r"E:\Desktop\final project\PythonApplication4\PythonApplication4\trainsmall"  # Path to folder containing training images
saving = r"E:\Desktop\final project\PythonApplication4\PythonApplication4\output"  # Path to save image features

print("extracting features")

# Loading resnet which is pre-trained model
model = getResNet50Model()

# Initialization if lists that stores features of images and also names of images
feats = []
names = []

# This will traverse model to the image folder containing sub-folders
for p_name in os.listdir(training):
    p_folder = os.path.join(training, p_name)
    
    if os.path.isdir(p_folder):  # Check for directories of persons images
        for image_name in os.listdir(p_folder):
            image_path = os.path.join(p_folder, image_name)
            
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # considers mentioned image formats that are valid
                print(f"Extracting features from image - {image_name} in folder - {p_name}")
                
                # Extracting features 
                X = model.extract_feat(image_path)
                
                # Appending names and features of the images
                feats.append(X)
                names.append(os.path.join(p_name, image_name))

# converting list containing features to numpy array
feats = np.array(feats)

# checks for folder to save features
os.makedirs(saving, exist_ok=True)

# Defining path for saving extracted features from images
output_file = os.path.join(saving, "ResnetFeatures.h5")

print("Extracted features are stored in this file")

# Save features and names to  H5 file with  dataset names
with h5py.File(output_file, 'w') as h5f:
    h5f.create_dataset('image_features', data=feats)
    h5f.create_dataset('image_names', data=np.string_(names))
    
print("extracted features from given images")
