import os
import glob
import re
import json
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm

from preprocessing.utils import *

# folder_path = './dataset_stats.txt'
room_lists = ["threed_front_diningroom", "threed_front_livingroom", "threed_front_bedroom"]
room = room_lists[0]

color_map = plt.cm.get_cmap('tab20', 35)
predefined_colors = [to_hex(color_map(i)) for i in range(35)]

stats = load_stats('./dataset_stats.txt')
class_labels = stats["class_labels"]
minx, miny, maxx, maxy = (stats["bounds_translations"][0],
                          stats["bounds_translations"][2],
                          stats["bounds_translations"][3],
                          stats["bounds_translations"][5])


def extract_json_objects_with_category(file_path):
    """
    Extracts JSON objects containing '{category' from a txt file and converts them to Python dictionaries.

    Args:
        file_path (str): Path to the txt file containing JSON-like objects.

    Returns:
        list: A list of dictionaries parsed from the JSON-like objects containing '{category'.
    """
    # Regex pattern to match JSON-like objects containing '{category'
    pattern = r"\{[^{}]*?\"furniture\"[^{}]*?\}"

    json_objects = []

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().replace('\'', "\"")
        matches = re.findall(pattern, content)  # Find all matches

        for match in matches:
            try:
                json_obj = json.loads(match)  # Convert JSON string to Python dict
                json_objects.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Matched content: {match}")

    return json_objects

datasets_folder = os.path.join(os.path.dirname(__file__), "output3")
output_paths = glob.glob(os.path.join(datasets_folder, "*_output.txt"))

for output_path in tqdm(output_paths):
    try:
        room_name = output_path.split('\\')[-1].replace("_vlm_output.txt", "")
        # dataset = load_dataset(os.path.join("../../datasets", room, "dataset.pkl"))
        # room_polygon = Polygon(dataset["test"]["polygons"][room_name]["room_polygon"])

        furniture_items = extract_json_objects_with_category(output_path)
        plt.clf()

        # Create a new plot
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

        # x_room, y_room = room_polygon.exterior.xy
        # ax.plot(x_room, y_room, color='black', alpha=1, label='Room Polygon')

        # Function to add rotated rectangle
        def add_rotated_rect(ax, x, z, width, depth, angle, color, label=None):
            # Calculate the lower-left corner based on center position
            lower_left_x = x - width / 2
            lower_left_z = z - depth / 2
            # Create a rectangle
            rect = patches.Rectangle((lower_left_x, lower_left_z), width, depth,
                                     linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
            # Apply rotation
            t = patches.transforms.Affine2D().rotate_around(x, z, angle) + ax.transData
            rect.set_transform(t)
            ax.add_patch(rect)
            if label:
                ax.text(x, z, label, ha='center', va='center', fontsize=8, color='black', clip_on=True)


        # Iterate through each furniture item and add to plot
        for item in furniture_items:
            x = item['x']
            z = item['z']
            width = item['width']
            depth = item['depth']
            angle = item['angle']
            furniture = item['furniture']

            if angle % 15 == 0:
                angle = math.radians(angle)

            label = 34
            if furniture in class_labels:
                label = class_labels.index(furniture)

            color = predefined_colors[label]

            add_rotated_rect(ax, x, z, width, depth, angle, color, furniture)

        # Set plot limits (adjust as needed)
        ax.set_xlim(minx - 0.1, maxx + 0.1)
        ax.set_ylim(miny - 0.1, maxy + 0.1)
        ax.set_aspect('equal')

        # Remove axes for clean output
        ax.axis('off')

        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig)
        buf.seek(0)

        # Resize the image to 128x128 and save
        img = Image.open(buf)
        img = img.resize((512, 512), Image.LANCZOS)
        buf.close()

        save_path = f"./output3/{room_name}_vlm_output.png"
        img.save(save_path)
    except:
        print("Failed")
        continue