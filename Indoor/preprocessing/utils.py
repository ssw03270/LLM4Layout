import csv
import json
import numpy as np
import pickle
import os

from PIL import Image
import io

from shapely.ops import unary_union
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import webcolors

def load_room_split_data(split_path):
    """
    Loads valid scene information from a given CSV file
    and returns it as a dictionary for easy lookup.
    """
    with open(split_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = {item[0]: item[1] for item in list(reader)}
    return data


def load_stats(stats_path):
    """
    Loads statistics information (e.g., object_types) from a JSON file.
    """
    with open(stats_path, "r") as file:
        stats = json.load(file)
    return stats

def load_dataset(dataset_path):
    """
    Loads statistics information (e.g., object_types) from a JSON file.
    """
    with open(dataset_path, "rb") as file:
        dataset = pickle.load(file)
    return dataset

def load_retrieval_results(dataset_path):
    """
    Loads statistics information (e.g., object_types) from a JSON file.
    """
    with open(dataset_path, "rb") as file:
        dataset = pickle.load(file)
    return dataset


def parse_boxes_npz(file_path):
    """
    Loads and parses box-related data (floor plan, object translations, sizes, angles, etc.)
    from an NPZ file.
    """
    boxes_npz = np.load(os.path.join(file_path, "boxes.npz"))
    boxes_data = {
        'uids': boxes_npz['uids'].tolist(),
        'jids': boxes_npz['jids'].tolist(),
        'scene_id': boxes_npz['scene_id'].item(),
        'scene_uid': boxes_npz['scene_uid'].item(),
        'scene_type': boxes_npz['scene_type'].item(),
        'json_path': boxes_npz['json_path'].item(),
        'room_layout': boxes_npz['room_layout'].tolist(),
        'floor_plan_vertices': boxes_npz['floor_plan_vertices'].tolist(),
        'floor_plan_faces': boxes_npz['floor_plan_faces'].tolist(),
        'floor_plan_centroid': boxes_npz['floor_plan_centroid'].tolist(),
        'class_labels': boxes_npz['class_labels'].tolist(),
        'translations': boxes_npz['translations'].tolist(),
        'sizes': boxes_npz['sizes'].tolist(),
        'angles': boxes_npz['angles'].tolist()
    }
    return boxes_data


def parse_descriptions_pkl(file_path):
    """
    Loads and parses descriptions data, including object class IDs, counts, and relations.
    """
    with open(os.path.join(file_path, "descriptions.pkl"), "rb") as f:
        descriptions_pkl = pickle.load(f)
    descriptions_data = {
        'obj_class_ids': descriptions_pkl['obj_class_ids'],
        'obj_counts': descriptions_pkl['obj_counts'],
        'obj_relations': descriptions_pkl['obj_relations']
    }
    return descriptions_data


def parse_models_info_pkl(file_path):
    """
    Loads and parses models_info data to retrieve metadata about each object.
    """
    with open(os.path.join(file_path, "models_info.pkl"), "rb") as f:
        models_info_pkl = pickle.load(f)

    models_info_data = []
    for model in models_info_pkl:
        models_info_data.append({
            'model_id': model['model_id'],
            'category': model['category'],
            'chatgpt_caption': model['chatgpt_caption']
        })
    return models_info_data


def parse_relations_npy(file_path):
    """
    Loads and parses relational information (e.g., pairwise relationships) from an NPY file.
    """
    relations_npy = np.load(os.path.join(file_path, "relations.npy"))
    relations_data = relations_npy.tolist()
    return relations_data

def parse_room_mask_png(file_path):
    """
    Loads a room mask image (e.g., .png file) and returns it as a numpy array
    formatted for use with ResNet or other deep learning models.
    """
    try:
        # Load the image using PIL
        img = Image.open(os.path.join(file_path, "room_mask.png"))

        # Convert to numpy array
        mask_array = np.array(img)

        # Normalize pixel values to [0, 1] (required for ResNet preprocessing)
        mask_array = mask_array.astype(np.float32) / 255.0

        # Return the mask as a numpy array
        return mask_array

    except Exception as e:
        print(f"Error loading mask image from {file_path}: {e}")
        return None

def parse_data_type(file_path, valid_split_data):
    file_path = file_path.split('_')[-1]
    return valid_split_data[file_path]

def construct_room_polygon(vertices, faces):
    """
    Constructs and returns the main room polygon by combining all face polygons
    and taking their union.
    """
    polygons = []
    for face in faces:
        poly = Polygon(np.array(vertices)[face][:, [0, 2]])
        polygons.append(poly)

    union_poly = unary_union(polygons)
    if union_poly.geom_type == 'MultiPolygon':
        room_polygon = max(union_poly.geoms, key=lambda p: p.area)
    else:
        room_polygon = Polygon(union_poly.exterior)
    return room_polygon


def center_polygon(polygon):
    """
    Translates the polygon so that its centroid is at (0, 0).
    """
    minx, miny, maxx, maxy = polygon.bounds

    # 중심점 계산
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2

    return translate(polygon, xoff=-center_x, yoff=-center_y)


def build_object_polygons(class_labels, sizes, angles, translations, stats):
    """
    Builds a list of shapely polygons representing objects in the room,
    along with metadata about each object's location, scale, and rotation.
    """
    object_polygons = []
    object_infos = []


    for label, size, angle, loc in zip(class_labels, sizes, angles, translations):
        label_idx = np.where(label)[0][0]
        category = stats['object_types'][label_idx]

        width, height, depth = size[0] * 2, size[1] * 2, size[2] * 2
        dx, dy, dz = loc[0], loc[1], loc[2]

        base_rect = Polygon([
            (-width / 2, -depth / 2),
            (width / 2, -depth / 2),
            (width / 2, depth / 2),
            (-width / 2, depth / 2)
        ])

        rotated_rect = rotate(base_rect, angle[0], use_radians=True)
        translated_rect = translate(rotated_rect, xoff=dx, yoff=dz)

        on_floor = True if round(dy - height / 2, 2) < 0.5 else False
        object_infos.append({
            'category': category,
            'location': [dx, dy, dz],
            'scale': [width / 2, depth / 2, height / 2],
            'rotation': angle[0],
            'polygon': list(translated_rect.exterior.coords),
            'on_floor': on_floor
        })
        object_polygons.append(translated_rect)

    return object_polygons, object_infos

def visualize_polygons(room_polygon, object_polygons, stats, draw_type="all", subfolder=""):
    """
    Displays the floor plan polygon and object polygons for visual inspection.
    """

    # Get bounds from the stats or room_polygon (depending on your preference)
    if "bounds_translations" in stats:
        minx, miny, maxx, maxy = stats["bounds_translations"][0], stats["bounds_translations"][2], stats["bounds_translations"][3], stats["bounds_translations"][5]
    else:
        minx, miny, maxx, maxy = room_polygon.bounds

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')  # Set the figure background color to black
    ax.set_facecolor('black')        # Set the axes background color to black

    x_room, y_room = room_polygon.exterior.xy
    ax.fill(x_room, y_room, color='white', alpha=1, label='Room Polygon')

    if draw_type != "boundary":
        for poly in object_polygons:
            x_obj, y_obj = poly.exterior.xy
            ax.plot(x_obj, y_obj)

    ax.set_xlim(minx - 0.1, maxx + 0.1)
    ax.set_ylim(miny - 0.1, maxy + 0.1)
    ax.set_aspect('equal')

    # Remove axes for clean output
    ax.axis('off')

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # Resize the image to 128x128 and save
    img = Image.open(buf)
    img = img.resize((256, 256), Image.LANCZOS)
    save_path = f"./temp/{subfolder}_{draw_type}.png"
    img.save(save_path)
    buf.close()
    print(f"Image saved to {save_path}")

def generate_polygons_image(room_polygon, object_polygons, layout, stats, draw_type="all"):
    """
    Displays the floor plan polygon and object polygons for visual inspection.
    """

    # Get bounds from the stats or room_polygon (depending on your preference)
    if "bounds_translations" in stats:
        minx, miny, maxx, maxy = stats["bounds_translations"][0], stats["bounds_translations"][2], stats["bounds_translations"][3], stats["bounds_translations"][5]
    else:
        minx, miny, maxx, maxy = room_polygon.bounds

    class_labels = stats["class_labels"]

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')  # Set the figure background color to black
    ax.set_facecolor('black')        # Set the axes background color to black

    x_room, y_room = room_polygon.exterior.xy
    ax.fill(x_room, y_room, color='white', alpha=1, label='Room Polygon')

    color_map = plt.cm.get_cmap('tab20', 35)
    predefined_colors = [to_hex(color_map(i)) for i in range(35)]

    if draw_type != "boundary":
        for idx, poly in enumerate(object_polygons):
            object_class_label = np.argmax(layout["class_labels"][idx])
            color = predefined_colors[object_class_label]

            x_obj, y_obj = poly.exterior.xy
            ax.plot(x_obj, y_obj, color=color)
            ax.fill(x_obj, y_obj, color=color, alpha=0.5)

    ax.set_xlim(minx - 0.1, maxx + 0.1)
    ax.set_ylim(miny - 0.1, maxy + 0.1)
    ax.set_aspect('equal')

    # Remove axes for clean output
    ax.axis('off')

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # Resize the image to 128x128 and save
    img = Image.open(buf)
    img = img.resize((256, 256), Image.LANCZOS)
    img_array = np.array(img)

    buf.close()
    return img_array

def save_image(image_array, file_name, save_path):
    save_path = f"{save_path}/{file_name}.png"
    image_array.save(save_path)

def closest_color(hex_color):
    """RGB 색상에 가장 가까운 색 이름을 반환"""
    rgb_color = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    min_diff = float('inf')
    closest_name = None
    for hex_value, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r, g, b = webcolors.hex_to_rgb(hex_value)
        diff = (r - rgb_color[0]) ** 2 + (g - rgb_color[1]) ** 2 + (b - rgb_color[2]) ** 2
        if diff < min_diff:
            min_diff = diff
            closest_name = name
    return closest_name


def layout_to_json(layout, stats):
    color_map = plt.cm.get_cmap('tab20', 35)
    predefined_colors = [to_hex(color_map(i)) for i in range(35)]
    color_names = [closest_color(color) for color in predefined_colors]
    class_labels = stats["class_labels"]

    translations = layout["translations"]
    sizes = layout["sizes"]
    angles = layout["angles"]
    labels = layout["class_labels"]

    furniture_list = []

    for trans, size, angle, label in zip(translations, sizes, angles, labels):
        # 라벨이 원-핫 인코딩된 경우
        label_index = np.argmax(label)
        class_label = class_labels[label_index]
        color_name = color_names[label_index]

        furniture = {
            'furniture': class_label,
            'x': round(trans[0], 2),
            'y': round(trans[1], 2),
            'z': round(trans[2], 2),
            'width': round(size[0], 2),
            'height': round(size[1], 2),
            'depth': round(size[2], 2),
            'angle': round(angle[0], 2),
            'color': color_name
        }
        furniture_list.append(furniture)

    json_output = json.dumps(furniture_list)
    json_output = json_output.replace("},", "}, \n")
    json_output = json_output.replace("[", "[\n")
    json_output = json_output.replace("]", "\n]")

    return json_output

def save_json(json_output, file_name, save_path):
    save_path = f"{save_path}/{file_name}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(json_output)