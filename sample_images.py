
import os
import shutil
from random import sample

def select_images_per_class(source_dir, target_dir, num_images=10):
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Loop through each class folder in the source directory
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name, 'images')
        if os.path.isdir(class_path):
            # Get a list of all image files in the class directory
            images = [img for img in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, img))]
            
            # Randomly select a specified number of images
            selected_images = sample(images, min(num_images, len(images)))
            
            # Create the target class directory
            target_class_dir = os.path.join(target_dir, class_name, 'images')
            os.makedirs(target_class_dir, exist_ok=True)
            
            # Copy the selected images to the target directory
            for img in selected_images:
                source_img_path = os.path.join(class_path, img)
                target_img_path = os.path.join(target_class_dir, img)
                shutil.copy2(source_img_path, target_img_path)

source_directory = 'data/tiny-imagenet-200/train'
target_directory = 'data/dataset'

select_images_per_class(source_directory, target_directory, num_images=10)