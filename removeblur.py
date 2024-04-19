import cv2
import os
import cv2
import numpy as np
from random import sample
import shutil

def compute_laplacian_variance(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Laplacian filter to the image
    filtered_image = cv2.Laplacian(gray_image, cv2.CV_64F)
    
    # Compute the variance of the filtered image
    variance = np.var(filtered_image)
    
    return variance



def  remove(directory):
    files_remove =0
    print("Initializing  direcotry ("+directory+").....")
    # Define your image directory
    image_dir = directory

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



    print("Removing blurred Data.....")
    # Apply preprocessing to each image
    for image_file in image_files:
        image = cv2.imread(image_file)
        # Apply preprocessing function
        variance  = compute_laplacian_variance(image)
        
        if variance < 1200:
            print(image_file)
            files_remove +=1
            os.remove(image_file)

    print("Preprocessing complete.")
    print("Files Removed:")
    print(files_remove)


# remove('./dataset/train')
# remove('./dataset/val')
# remove('./dataset/test')
# remove('./dataset/augmented')
