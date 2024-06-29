import os
from PIL import Image

def resize_images_in_directory(directory, size=(100, 100)):
    """
    Resizes all images in the given directory and its subdirectories to the specified size.
    
    :param directory: Path to the root directory containing images.
    :param size: Desired size for resizing images, default is (100, 100).
    """
    print(os.walk(directory))
    for root, _, files in os.walk(directory):
        # print(f"{root} + {files}")
        for file in files:
            # print(root)
            if file.endswith(('.png', '.jpg', '.jpeg', '.JPEG')):
                file_path = os.path.join(root, file)
                print(file_path)
                try:
                    with Image.open(file_path) as img:
                        img = img.resize(size, Image.LANCZOS)
                        img.save(file_path)
                    print(f"Resized and saved image: {file_path}")
                except Exception as e:
                    print(f"Failed to process image {file_path}: {e}")



tinyimagenet_path = "data/tiny-imagenet-200"
resize_images_in_directory(tinyimagenet_path)
