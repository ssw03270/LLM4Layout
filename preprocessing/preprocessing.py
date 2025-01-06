import os
import numpy as np
import pickle
import json
import random
import csv
from collections import defaultdict
from PIL import Image

from tqdm import tqdm
from shapely.ops import unary_union
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
import matplotlib.pyplot as plt

def compute_category_means(train, val, test, stats):
    """
    Computes mean + std for sizes, translations, and angles per category.

    Parameters:
        train (list): Training data.
        val (list): Validation data.
        test (list): Test data.
        stats (dict): Statistics data containing 'object_types'.

    Returns:
        dict: A dictionary containing mean and std values per category.
              예: {
                    'Chair': {
                        'sizes_mean': [...],
                        'sizes_std': [...],
                        'translations_mean': [...],
                        'translations_std': [...],
                        'angles_mean': float,
                        'angles_std': float
                    },
                    ...
                  }
    """
    from collections import defaultdict
    import numpy as np

    object_types = stats['object_types']  # 카테고리 이름 리스트 (라벨 인덱스 기준)

    # 카테고리별로 raw data를 수집할 딕셔너리
    sizes_dict = defaultdict(list)        # 카테고리 -> [[w, h, d], [w, h, d], ...]
    translations_dict = defaultdict(list) # 카테고리 -> [[x, y, z], [x, y, z], ...]
    angles_dict = defaultdict(list)       # 카테고리 -> [angle, angle, ...]

    # 모든 데이터셋(train, val, test)을 합쳐서 처리
    for dataset in [train, val, test]:
        for file_path, data_point in dataset.items():
            class_labels = data_point['class_labels']      # (N, one-hot)
            sizes = data_point['sizes']                    # (N, 3)
            translations = data_point['translations']      # (N, 3)
            angles = data_point['angles']                  # (N, 1) 형태

            for label, size, translation, angle in zip(class_labels, sizes, translations, angles):
                label_idx = np.where(label)[0][0]  # one-hot -> 정수 인덱스
                category = object_types[label_idx]

                sizes_dict[category].append(size)
                translations_dict[category].append(translation)
                angles_dict[category].append(angle[0])  # angle은 [val] 형태이므로 angle[0]

    # 카테고리별 (mean, std) 계산
    means_per_category = {}
    for category in object_types:
        # (n, 3)
        sizes_arr = np.array(sizes_dict[category])
        # (n, 3)
        translations_arr = np.array(translations_dict[category])
        # (n,)
        angles_arr = np.array(angles_dict[category])

        if sizes_arr.size > 0:
            sizes_mean = np.mean(sizes_arr, axis=0).tolist()  # [mean_w, mean_h, mean_d]
            sizes_std = np.std(sizes_arr, axis=0).tolist()     # [std_w, std_h, std_d]
        else:
            sizes_mean = [None, None, None]
            sizes_std = [None, None, None]

        if translations_arr.size > 0:
            translations_mean = np.mean(translations_arr, axis=0).tolist()  # [mean_x, mean_y, mean_z]
            translations_std = np.std(translations_arr, axis=0).tolist()    # [std_x, std_y, std_z]
        else:
            translations_mean = [None, None, None]
            translations_std = [None, None, None]

        if angles_arr.size > 0:
            angles_mean = float(np.mean(angles_arr))  # 스칼라
            angles_std = float(np.std(angles_arr))     # 스칼라
        else:
            angles_mean = None
            angles_std = None

        means_per_category[category] = {
            'sizes_mean': sizes_mean,
            'sizes_std': sizes_std,
            'translations_mean': translations_mean,
            'translations_std': translations_std,
            'angles_mean': angles_mean,
            'angles_std': angles_std
        }

    return means_per_category


def compute_overall_statistics(means_per_category):
    """
    Among the category means, finds:
     - min (with its std)
     - max (with its std)
     - median (with its std)
    for sizes (3 dim), translations (3 dim), and angles (1 dim).

    Returns:
        dict: 예시 구조
        {
          "sizes_mean": {
            "size_dim_0": {
              "min":  {"category": str, "mean": float, "std": float},
              "max":  {...},
              "median": {...}
            },
            "size_dim_1": { ... },
            "size_dim_2": { ... }
          },
          "translations_mean": {
            "translation_dim_0": { ... },
            ...
          },
          "angles_mean": {
            "min":  {"category": str, "mean": float, "std": float},
            "max":  {...},
            "median": {...}
          }
        }
    """
    import numpy as np

    # 최종 결과를 넣을 딕셔너리
    overall_stats = {
        'sizes_mean': {},
        'translations_mean': {},
        'angles_mean': {}
    }

    # ---------------------------------------------------------------------
    # (A) 사이즈, 위치는 3차원(dim=3)이므로 각 dim별로 처리
    # ---------------------------------------------------------------------
    # 1) size_dim_0,1,2를 위한 (카테고리, mean, std) 리스트 만들기
    size_dim_data = [[] for _ in range(3)]  # size_dim_data[0] -> dim0용
    trans_dim_data = [[] for _ in range(3)]  # trans_dim_data[0] -> dim0용

    for category, cat_stats in means_per_category.items():
        sizes_mean = cat_stats['sizes_mean']  # 예: [mean_w, mean_h, mean_d]
        sizes_std = cat_stats['sizes_std']  # 예: [std_w, std_h, std_d]
        trans_mean = cat_stats['translations_mean']  # 예: [mean_x, mean_y, mean_z]
        trans_std = cat_stats['translations_std']  # 예: [std_x, std_y, std_z]

        # sizes
        for i in range(3):
            if sizes_mean[i] is not None:
                size_dim_data[i].append((category, sizes_mean[i], sizes_std[i]))
        # translations
        for i in range(3):
            if trans_mean[i] is not None:
                trans_dim_data[i].append((category, trans_mean[i], trans_std[i]))

    # 2) 각 dim별로 sort 후, min/max/median를 추출
    def get_min_max_median(data_list):
        """
        data_list: List of (category, mean_val, std_val)
        return: {
          'min':    {'category':..., 'mean':..., 'std':...},
          'max':    {...},
          'median': {...}
        }
        """
        data_sorted = sorted(data_list, key=lambda x: x[1])  # mean_val 기준 정렬

        # min: 맨 앞
        min_cat, min_mean, min_std = data_sorted[0]
        # max: 맨 뒤
        max_cat, max_mean, max_std = data_sorted[-1]

        # median: 가운데
        mid_idx = len(data_sorted) // 2
        # 짝수라면 중간 2개 중 하나를 선택(여기서는 아래쪽 인덱스 mid_idx-1)
        if len(data_sorted) % 2 == 0:
            median_cat, median_mean, median_std = data_sorted[mid_idx - 1]
        else:
            median_cat, median_mean, median_std = data_sorted[mid_idx]

        return {
            'min': {
                'category': min_cat,
                'mean': float(min_mean),
                'std': float(min_std),
            },
            'max': {
                'category': max_cat,
                'mean': float(max_mean),
                'std': float(max_std),
            },
            'median': {
                'category': median_cat,
                'mean': float(median_mean),
                'std': float(median_std),
            }
        }

    # 3) 사이즈 결과 저장
    for i in range(3):
        dim_key = f'size_dim_{i}'
        if len(size_dim_data[i]) > 0:
            overall_stats['sizes_mean'][dim_key] = get_min_max_median(size_dim_data[i])
        else:
            overall_stats['sizes_mean'][dim_key] = None

    # 4) 위치 결과 저장
    for i in range(3):
        dim_key = f'translation_dim_{i}'
        if len(trans_dim_data[i]) > 0:
            overall_stats['translations_mean'][dim_key] = get_min_max_median(trans_dim_data[i])
        else:
            overall_stats['translations_mean'][dim_key] = None

    # ---------------------------------------------------------------------
    # (B) 각 카테고리의 angle_mean, angle_std는 스칼라이므로 별도 처리
    # ---------------------------------------------------------------------
    angle_data = []
    for category, cat_stats in means_per_category.items():
        if cat_stats['angles_mean'] is not None:
            angle_data.append((category, cat_stats['angles_mean'], cat_stats['angles_std']))

    if len(angle_data) > 0:
        overall_stats['angles_mean'] = get_min_max_median(angle_data)
    else:
        overall_stats['angles_mean'] = None

    return overall_stats


def save_overall_statistics_to_json(overall_stats, file_path):
    """
    Saves the overall statistics to a JSON file.

    Parameters:
        overall_stats (dict): The overall statistics dictionary.
        file_path (str): The path where the JSON file will be saved.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(overall_stats, f, indent=4)
        print(f"Overall statistics successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving overall statistics: {e}")

def visualize_polygons(room_polygon, object_polygons):
    """
    Displays the floor plan polygon and object polygons for visual inspection.
    """
    fig, ax = plt.subplots()
    x_room, y_room = room_polygon.exterior.xy
    ax.plot(x_room, y_room, color='black', linewidth=2)

    for poly in object_polygons:
        x_obj, y_obj = poly.exterior.xy
        ax.plot(x_obj, y_obj)

    minx, miny, maxx, maxy = room_polygon.bounds
    ax.set_xlim(minx - 0.1, maxx + 0.1)
    ax.set_ylim(miny - 0.1, maxy + 0.1)
    ax.set_aspect('equal')
    plt.show()


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
    centroid = polygon.centroid
    return translate(polygon, xoff=-centroid.x, yoff=-centroid.y)


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

def process_scene(folder_path, room, split_path):
    """
    Loads split data, statistics, and all relevant scene information for each subfolder.
    Constructs polygons for the room and the objects, then visualizes them.
    """
    valid_split_data = load_room_split_data(split_path)
    stats = load_stats(os.path.join(folder_path, room, "dataset_stats.txt"))

    target_path = os.path.join(folder_path, room)
    subfolders = [
        f for f in os.listdir(target_path)
        if os.path.isdir(os.path.join(target_path, f)) and "test" not in f and "train" not in f
    ]
    train_preprocessed_data = {}
    test_preprocessed_data = {}
    val_preprocessed_data = {}
    for subfolder in tqdm(subfolders):
        file_path = os.path.join(target_path, subfolder)
        file_data_type = parse_data_type(file_path, valid_split_data)

        # Load and parse various data files
        boxes_data = parse_boxes_npz(file_path)
        descriptions_data = parse_descriptions_pkl(file_path)
        models_info_data = parse_models_info_pkl(file_path)
        relations_data = parse_relations_npy(file_path)
        room_mask_data = parse_room_mask_png(file_path)

        # Build room polygon
        room_polygon_raw = construct_room_polygon(
            boxes_data['floor_plan_vertices'],
            boxes_data['floor_plan_faces']
        )
        room_polygon = center_polygon(room_polygon_raw)

        # Build object polygons
        object_polygons, object_infos = build_object_polygons(
            boxes_data['class_labels'],
            boxes_data['sizes'],
            boxes_data['angles'],
            boxes_data['translations'],
            stats
        )

        # Visualization
        # visualize_polygons(room_polygon, object_polygons)

        if file_data_type == "train":
            train_preprocessed_data[subfolder] = {
                    "class_labels": boxes_data['class_labels'],
                    "sizes": boxes_data['sizes'],
                    "angles": boxes_data['angles'],
                    "translations": boxes_data['translations'],
                    "room_mask": room_mask_data
                }
        elif file_data_type == "val":
            val_preprocessed_data[subfolder] = {
                    "class_labels": boxes_data['class_labels'],
                    "sizes": boxes_data['sizes'],
                    "angles": boxes_data['angles'],
                    "translations": boxes_data['translations'],
                    "room_mask": room_mask_data
                }

        elif file_data_type == "test":
            test_preprocessed_data[subfolder] = {
                    "class_labels": boxes_data['class_labels'],
                    "sizes": boxes_data['sizes'],
                    "angles": boxes_data['angles'],
                    "translations": boxes_data['translations'],
                    "room_mask": room_mask_data
                }

    return train_preprocessed_data, test_preprocessed_data, val_preprocessed_data

def save_dict_list_to_npz(train, val, test, file_path):
    """
    Saves a list of dictionaries to a .npz file.
    """
    try:
        with open(os.path.join(file_path, "train.pkl"), 'wb') as f:
            pickle.dump(train, f)
        with open(os.path.join(file_path, "val.pkl"), 'wb') as f:
            pickle.dump(val, f)
        with open(os.path.join(file_path, "test.pkl"), 'wb') as f:
            pickle.dump(test, f)
        print(f"Data successfully saved to {file_path}")

    except Exception as e:
        print(f"Error saving data: {e}")

def main():
    """
    Sets up the folder paths, rooms, seeds, and initiates processing for each room type.
    """
    random.seed(42)
    np.random.seed(42)

    folder_path = 'E:/Resources/IndoorSceneSynthesis/InstructScene'
    room_lists = ["threed_front_diningroom", "threed_front_livingroom", "threed_front_bedroom"]
    split_paths = [
        "E:/Resources/IndoorSceneSynthesis/InstructScene/valid_scenes/diningroom_threed_front_splits.csv",
        "E:/Resources/IndoorSceneSynthesis/InstructScene/valid_scenes/livingroom_threed_front_splits.csv",
        "E:/Resources/IndoorSceneSynthesis/InstructScene/valid_scenes/bedroom_threed_front_splits.csv",
    ]

    for room, split_path in zip(room_lists, split_paths):
        train, val, test = process_scene(folder_path, room, split_path)

        save_path = os.path.join("../datasets", room)
        os.makedirs(save_path, exist_ok=True)
        save_dict_list_to_npz(train, val, test, save_path)

        print(f"Saved train: {len(train)}, val: {len(val)}, test: {len(test)}")

        # 통계 계산을 위해 stats 로드
        stats = load_stats(os.path.join(folder_path, room, "dataset_stats.txt"))

        # 카테고리별 평균 계산
        means_per_category = compute_category_means(train, val, test, stats)

        # 전체 통계 계산
        overall_stats = compute_overall_statistics(means_per_category)

        # 전체 통계를 JSON으로 저장
        overall_stats_file_path = os.path.join(save_path, "overall_statistics.json")
        save_overall_statistics_to_json(overall_stats, overall_stats_file_path)

if __name__ == "__main__":
    main()