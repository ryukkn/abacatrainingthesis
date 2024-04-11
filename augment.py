import os
import cv2
import numpy as np
from random import sample

def preprocess_functions(p=1.0, size_list=[64, 72, 96], innerMove=False):
    def _preprocess_functions(x):
        height, width = x.shape[:2]
        # Inner move
        if(np.random.random() < p) and innerMove:
            _patch_size = sample(size_list, k=1)[0]
            patch_size = (_patch_size, _patch_size)
            # random offset
            offsetx = np.random.randint(1, 10)
            offsety = np.random.randint(2, height-_patch_size) - 1
            # Swap two patches
            patch_1 = x[offsety:patch_size[0]+offsety, offsetx:patch_size[1]+offsetx].copy()
            patch_2 = x[-(patch_size[0]+offsety):-offsety, -(patch_size[1]+offsetx):-offsetx].copy()
            # Swap pixels between patches
            x[offsety:patch_size[0]+offsety, offsetx:patch_size[1]+offsetx] = patch_2
            x[-(patch_size[0]+offsety):-offsety, -(patch_size[1]+offsetx):-offsetx] = patch_1
        return x
    return _preprocess_functions

# Define your image directory
image_dir = './dataset/augmented'

# Get list of all folders in the directory
folders = [f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))]

# Initialize an empty list to store image files
image_files = []

# Iterate through each folder
for folder in folders:
    # Get the path to the current folder
    folder_path = os.path.join(image_dir, folder)
    
    # Get list of all image files in the current folder
    folder_image_files = [os.path.join(folder_path,f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Extend the image_files list with the image files from the current folder
    image_files.extend(folder_image_files)


# Apply preprocessing to each image
for image_file in image_files:
    print(image_file)
    image = cv2.imread(image_file)
    resized_image = cv2.resize(image, (224, 224))
    # Apply preprocessing function
    preprocessed_image = preprocess_functions(innerMove=True)(resized_image)
    
    # Save the preprocessed image
    cv2.imwrite(image_file, preprocessed_image)

print("Preprocessing complete.")