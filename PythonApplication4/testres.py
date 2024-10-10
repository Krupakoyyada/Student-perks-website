import h5py

file_path = r"C:\Users\Saikrupa\source\repos\PythonApplication4\PythonApplication4\output\ResnetFeatures.h5"  # Update this path if needed

# Open the HDF5 file
with h5py.File(file_path, 'r') as h5f:
    # Check if 'image_features' dataset exists
    if 'image_features' in h5f:
        # Access the image features dataset
        image_features = h5f['image_features'][:]
        
        print("Image features:")
        for i, feature_vector in enumerate(image_features):
            print(f"Feature vector for image {i}: {feature_vector}")

    else:
        print("'image_features' dataset not found in the file.")
    
    # Check if 'image_names' dataset exists
    if 'image_names' in h5f:
        # Access the image names dataset
        image_names = h5f['image_names'][:]  # This is an array of byte strings
        
        # Convert byte strings to regular strings
        names = [name.decode('utf-8') for name in image_names]

        print("\nImage names:")
        for i, name in enumerate(names):
            print(f"Image {i}: {name}")
    else:
        print("'image_names' dataset not found in the file.")

