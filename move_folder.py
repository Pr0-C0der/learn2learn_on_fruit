import shutil
import os

def move_folder(src, dest):
    # Check if the source folder exists
    if not os.path.exists(src):
        print(f"Source folder {src} does not exist.")
        return

    # Check if the destination folder exists
    if not os.path.exists(dest):
        # Create destination folder if it does not exist
        os.makedirs(dest)

    # Move the folder
    shutil.move(src, dest)
    print(f"Moved folder {src} to {dest}")

# Example usage
src_folder = 'tiny-imagenet-200'  # Replace with the path to the source folder
dest_folder = 'data/tiny-imagenet-200'  # Replace with the path to the destination folder

move_folder(src_folder, dest_folder)
