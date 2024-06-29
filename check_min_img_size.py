import os
from PIL import Image

def check_image_dimensions(directory, min_size=(100, 100)):
    """
    Checks all images in the given directory and its subdirectories to see if their dimensions are less than the specified size.
    
    :param directory: Path to the root directory containing images.
    :param min_size: Minimum acceptable dimensions for images, default is (100, 100).
    """
    min_width, min_height = min_size
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.JPEG')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        if width < min_width or height < min_height:
                            print(f"Image {file_path} has dimensions {width}x{height}, which is less than {min_width}x{min_height}")
                except Exception as e:
                    print(f"Failed to process image {file_path}: {e}")


tinyimagenet_path = "data/tiny-imagenet-200" 
check_image_dimensions(tinyimagenet_path)