import os
import cv2
import numpy as np
from random import sample
import shutil


def crop_center(image, crop_width, crop_height):
    """
    Crop the center portion of the image with specified width and height.
    """
    height, width = image.shape[:2]
    
    # Calculate starting and ending indices for cropping
    start_x = max(0, (width - crop_width) // 2)
    start_y = max(0, (height - crop_height) // 2)
    end_x = min(width, start_x + crop_width)
    end_y = min(height, start_y + crop_height)
    
    # Perform cropping
    cropped_image = image[start_y:end_y, start_x:end_x]
    
    # return    foreground_extractor(cv2.resize(cropped_image, (224,224),  interpolation=cv2.INTER_NEAREST)) 
    return    cv2.resize(cropped_image, (320,320),  interpolation=cv2.INTER_NEAREST)

def crop():
    def _crop(x):
        height, width = x.shape[:2]
        return  crop_center(x, int(width/2), int(width/2))
    return _crop



def  resize(directory):
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



    print("Resizing Data.....")
    # Apply preprocessing to each image
    for image_file in image_files:
        image = cv2.imread(image_file)
        # Apply preprocessing function
        preprocessed_image = crop()(image)
        
        # Save the preprocessed image
        cv2.imwrite(image_file, preprocessed_image)

    print("Preprocessing complete.")


resize('./dataset/train')
resize('./dataset/val')
resize('./dataset/test')
# resize('./dataset-test/test - Copy')