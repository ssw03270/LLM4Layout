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
Once we extract a design plan from a given urban satellite image, we want to explain the urban layout through a set of rules.
For example, an L-shaped building can be described independently as follows 
{"Category": "L-shaped", "x": 0.08, "y": 0.39, "z": -0.4, "width": 1.11, "height": 0.39, "angle": 1.57, 
However, in most cases, given the boundaries of the city blocks, this can be explained as follows:

'Boundaries: Rectangles, gently curved rectangles are diagonally inclined, and buildings are arranged in two rows.
For a clear design and aesthetic, it is desirable that the building is spaced apart from the boundary at regular intervals, and that the central point of the individual building rows is soft to a degree similar to the boundary.
Once you have roughly determined the number, type, and size of buildings that will be within the boundaries, you can establish an individual building placement strategy based on the spacing and boundaries between the buildings.
e.g. x location of building i = (x location of the neighborest building + x location of the nearest boundary)/2'

Natural language based design planning extracted above can be helpful for layout planning of urban planning in urban satellite images.
For the given satellite image, please create a full natural language based design plan as below
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
    print(satellite_image_path)
    exit()