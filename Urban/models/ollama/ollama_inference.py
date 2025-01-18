import os
import io
import glob
from tqdm import tqdm

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from ollama import chat


def concat_images(image_path1, image_path2, axis='horizontal'):
    """
    Concatenate two images either horizontally or vertically.

    Args:
        image_path1 (str): Path to the first image.
        image_path2 (str): Path to the second image.
        axis (str): Direction of concatenation - 'horizontal' or 'vertical'.

    Returns:
        Image: Concatenated image.
    """
    # Load images
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    # Ensure both images have the same mode and size for concatenation
    if axis == 'horizontal':
        img2 = img2.resize((img2.width, img1.height))  # Resize img2 to match height of img1
        total_width = img1.width + img2.width
        max_height = max(img1.height, img2.height)
        new_image = Image.new('RGB', (total_width, max_height))
        new_image.paste(img1, (0, 0))
        new_image.paste(img2, (img1.width, 0))
    elif axis == 'vertical':
        img2 = img2.resize((img1.width, img2.height))  # Resize img2 to match width of img1
        total_height = img1.height + img2.height
        max_width = max(img1.width, img2.width)
        new_image = Image.new('RGB', (max_width, total_height))
        new_image.paste(img1, (0, 0))
        new_image.paste(img2, (0, img1.height))
    else:
        raise ValueError("Axis must be 'horizontal' or 'vertical'.")

    return np.array(new_image)


def numpy_to_bytes(np_image):
    """
    Convert a NumPy image array to bytes.

    Args:
        np_image (np.ndarray): NumPy array representing the image.

    Returns:
        bytes: The image encoded in byte format.
    """
    # Convert NumPy array to PIL Image
    image = Image.fromarray(np_image.astype('uint8'))

    # Save the image to a BytesIO stream
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")  # Specify format as needed (e.g., PNG, JPEG)
        image_bytes = buffer.getvalue()

    return image_bytes

dataset_folder = "F:\\Urban\\Preprocessed_Dataset"
image_folder = os.path.join(dataset_folder, "image_dataset")
pkl_folder = os.path.join(dataset_folder, "pkl_dataset")

satellite_image_folder = os.path.join(image_folder, "satellite_image")
bldg_bbox_image_folder = os.path.join(image_folder, "bldg_bbox_image")
bldg_polygon_image_folder = os.path.join(image_folder, "bldg_polygon_image")
blk_image_folder = os.path.join(image_folder, "blk_image")

satellite_image_paths = glob.glob(os.path.join(satellite_image_folder, "*.png"))
bldg_polygon_image_paths = glob.glob(os.path.join(bldg_polygon_image_folder, "*.png"))

for satellite_image_path, bldg_polygon_image_path in zip(tqdm(satellite_image_paths), bldg_polygon_image_paths):
    image = concat_images(satellite_image_path, bldg_polygon_image_path, axis='horizontal')
    image = numpy_to_bytes(image)
    vlm_prompt = """
    "On the left side of this image, a satellite photograph of an urban block is shown.
    On the right side is a segmented image of the same urban block, highlighting only the boundaries of the block and the buildings within it.
    Using this information, describe the arrangement of buildings in the urban block depicted in the image.
    """

    vlm_response = chat(
        model='llama3.2-vision',
        messages=[
            {
                'role': 'user',
                'content': vlm_prompt,
                'images': [image],
            }
        ],
    )
    vlm_output = vlm_response.message.content

    print(vlm_output)
    exit()