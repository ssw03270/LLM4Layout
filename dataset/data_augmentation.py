import os
import copy

import pickle
import json

from tqdm import tqdm

import numpy as np
np.random.seed(327)

from shapely.geometry import Polygon
from math import cos, sin

def check_collision(polygon_list):
    """Polygon 리스트에서 충돌(겹치는지) 여부를 확인"""
    for i in range(len(polygon_list)):
        for j in range(i + 1, len(polygon_list)):
            if polygon_list[i].intersects(polygon_list[j]):  # 두 다각형이 교차하면 True
                return True  # 충돌 발생
    return False  # 충돌 없음

def create_polygons(translations, sizes, angles, class_labels, obj_classes):
    """
    translations, sizes, angles 배열을 기반으로 Shapely Polygon 객체를 생성하는 함수.

    Args:
        translations (numpy.ndarray): [N, 3] 형태로 각 객체의 중심 좌표 (x, y, z).
        sizes (numpy.ndarray): [N, 3] 형태로 각 객체의 크기 (width, height, depth).
        angles (numpy.ndarray): [N, 1] 형태로 각 객체의 회전 각도 (라디안).

    Returns:
        list[Polygon]: Shapely Polygon 객체 리스트.
    """
    polygons = []
    for i in range(len(translations)):
        if obj_classes[np.argmax(class_labels[i])] in ["ceiling_lamp", "pendant_lamp"]:
            continue

        x, _, z = translations[i]  # z축은 무시
        width, _, depth = sizes[i]  # depth는 무시
        angle = angles[i][0]  # 라디안

        # 폴리곤의 네 꼭짓점 계산 (중심 기준)
        half_width, half_depth = width, depth
        corners = [
            (-half_width, -half_depth),
            (half_width, -half_depth),
            (half_width, half_depth),
            (-half_width, half_depth),
        ]

        # 회전 및 평행 이동 적용
        rotated_corners = [
            (
                x + cos(angle) * cx - sin(angle) * cz,
                z + sin(angle) * cx + cos(angle) * cz
            )
            for cx, cz in corners
        ]

        # Shapely Polygon 생성
        polygons.append(Polygon(rotated_corners))

    return polygons

def predicate_types():
    return [
        "above", "left of", "in front of",
        "closely left of", "closely in front of",
        "below", "right of", "behind",
        "closely right of", "closely behind"
    ]

def add_function(data, descriptions_data, models_info_data, obj_classes):
    '''
    keys: uids, jids, class_labels, translations, sizes, angles 에 영향을 줘야 함
    '''
    source_data = copy.deepcopy(data)
    target_data = copy.deepcopy(data)

    object_count = len(data["uids"])
    random_idx = np.random.randint(0, object_count)
    for key in source_data:
        if key in ["uids", "jids", "class_labels", "translations", "sizes", "angles"]:
            source_data[key] = np.delete(source_data[key], random_idx, axis=0)

    command = []
    for relation in descriptions_data["obj_relations"]:
        obj_1, rel, obj_2 = relation

        if obj_1 == random_idx:
            obj_2_idx = np.argmax(target_data["class_labels"][obj_2])
            command = [models_info_data[obj_1]["chatgpt_caption"], predicate_types()[rel], obj_classes[obj_2_idx].replace("_", " ")]
            break

    return source_data, target_data, command

def move_function(data, descriptions_data, models_info_data, obj_classes):
    '''
    위가 마이너스 z, 왼쪽이 마이너스 x
    [0: 상, 1: 하, 2: 좌, 3: 우]
    '''
    source_data = copy.deepcopy(data)
    target_data = copy.deepcopy(data)

    for try_count in range(10):
        source_data = copy.deepcopy(data)
        target_data = copy.deepcopy(data)

        object_count = len(data["uids"])
        random_obj_idx = np.random.randint(0, object_count)

        random_distance = np.random.choice(np.arange(0.1, 1.6, 0.1))
        random_direction = np.random.randint(0, 4)

        if random_direction == 0:
            target_data["translations"][random_obj_idx][2] -= random_distance
            text_direction = "back"
        elif random_direction == 1:
            target_data["translations"][random_obj_idx][2] += random_distance
            text_direction = "front"
        elif random_direction == 2:
            target_data["translations"][random_obj_idx][0] -= random_distance
            text_direction = "left"
        else:
            target_data["translations"][random_obj_idx][0] -= random_distance
            text_direction = "right"

        polygons = create_polygons(target_data["translations"], target_data["sizes"], target_data["angles"], target_data["class_labels"], obj_classes)
        collision = check_collision(polygons)

        if collision:
            continue

        command = [models_info_data[random_obj_idx]["chatgpt_caption"], text_direction, random_distance]
        return source_data, target_data, command
    return source_data, target_data, []

def rotate_function(data, descriptions_data, models_info_data, obj_classes):
    '''

    '''
    source_data = copy.deepcopy(data)
    target_data = copy.deepcopy(data)

    for try_count in range(10):
        source_data = copy.deepcopy(data)
        target_data = copy.deepcopy(data)

        object_count = len(data["uids"])
        random_obj_idx = np.random.randint(0, object_count)

        random_degree = np.random.choice(np.arange(15, 195, 15))
        target_data["angles"][random_obj_idx] += np.deg2rad(random_degree)
        target_data["angles"][random_obj_idx] = np.arctan2(
            np.sin(target_data["angles"][random_obj_idx]),
            np.cos(target_data["angles"][random_obj_idx])
        )

        polygons = create_polygons(target_data["translations"], target_data["sizes"], target_data["angles"], target_data["class_labels"], obj_classes)
        collision = check_collision(polygons)

        if collision:
            continue

        command = [models_info_data[random_obj_idx]["chatgpt_caption"], int(random_degree)]
        return source_data, target_data, command
    return source_data, target_data, []

def resize_function(data, descriptions_data, models_info_data, obj_classes):
    '''

    '''
    source_data = copy.deepcopy(data)
    target_data = copy.deepcopy(data)

    for try_count in range(10):
        source_data = copy.deepcopy(data)
        target_data = copy.deepcopy(data)

        object_count = len(data["uids"])
        random_obj_idx = np.random.randint(0, object_count)

        random_scaling = np.random.choice(np.arange(0.5, 1.6, 0.1))
        while 0.9 <= random_scaling and random_scaling <= 1.1:
            random_scaling = np.random.choice(np.arange(0.5, 1.6, 0.1))
        if random_scaling < 1:
            resize_type = "shrink"
        else:
            resize_type = "enlarge"

        target_data["sizes"][random_obj_idx] *= random_scaling

        polygons = create_polygons(target_data["translations"], target_data["sizes"], target_data["angles"], target_data["class_labels"], obj_classes)
        collision = check_collision(polygons)

        if collision:
            continue

        command = [models_info_data[random_obj_idx]["chatgpt_caption"], random_scaling, resize_type]
        return source_data, target_data, command
    return source_data, target_data, []

def replace_function(data, descriptions_data, models_info_data, obj_classes, current_subfolder, subfolders):
    '''

    '''
    source_data = copy.deepcopy(data)
    target_data = copy.deepcopy(data)

    for try_count in range(10):
        source_data = copy.deepcopy(data)
        target_data = copy.deepcopy(data)

        object_count = len(data["uids"])
        random_obj_idx = np.random.randint(0, object_count)

        subfolder_count = len(subfolders)
        retrieved_subfolder_idx = np.random.randint(0, subfolder_count)
        retrieved_subfolder = subfolders[retrieved_subfolder_idx]
        if current_subfolder == retrieved_subfolder or "rendered_scene_256" in retrieved_subfolder:
            continue

        retrieved_data = np.load(os.path.join(root_path, retrieved_subfolder, "boxes.npz"))
        retrieved_data = {key: retrieved_data[key] for key in retrieved_data.files}

        current_obj_class = target_data["class_labels"][random_obj_idx]
        for retrieved_obj_idx in range(len(retrieved_data["class_labels"])):
            if np.argmax(current_obj_class) == np.argmax(retrieved_data["class_labels"][retrieved_obj_idx]):
                target_data["sizes"][random_obj_idx] = retrieved_data["sizes"][retrieved_obj_idx]
                target_data["translations"][random_obj_idx][1] = retrieved_data["translations"][retrieved_obj_idx][1]
                target_data["uids"][random_obj_idx] = retrieved_data["uids"][retrieved_obj_idx]
                target_data["jids"][random_obj_idx] = retrieved_data["jids"][retrieved_obj_idx]


                polygons = create_polygons(target_data["translations"], target_data["sizes"], target_data["angles"], target_data["class_labels"], obj_classes)
                collision = check_collision(polygons)

                if collision:
                    target_data = copy.deepcopy(data)
                    continue

                with open(os.path.join(root_path, retrieved_subfolder, "models_info.pkl"), "rb") as f:
                    retrieved_models_info_data = pickle.load(f)

                command = [models_info_data[random_obj_idx]["chatgpt_caption"], retrieved_models_info_data[retrieved_obj_idx]["chatgpt_caption"]]
                return source_data, target_data, command
    return source_data, target_data, []


room_types = ["threed_front_bedroom"]
base_folder = "../../InstructScene/dataset/InstructScene"

for room_type in room_types:
    root_path = os.path.join(base_folder, room_type)
    with open(os.path.join(root_path, "dataset_stats.txt"), "r", encoding='utf-8') as f:
        stats = json.load(f)
    object_classes = stats["class_labels"]

    subfolders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]

    for subfolder in tqdm(subfolders):
        if "rendered_scene_256" in subfolder:
            continue

        boxes_data = np.load(os.path.join(root_path, subfolder, "boxes.npz"))
        boxes_data = {key: boxes_data[key] for key in boxes_data.files}

        with open(os.path.join(root_path, subfolder, "descriptions.pkl"), "rb") as f:
            descriptions_data = pickle.load(f)

        with open(os.path.join(root_path, subfolder, "models_info.pkl"), "rb") as f:
            models_info_data = pickle.load(f)

        source_boxes_data, target_boxes_data, command = add_function(boxes_data, descriptions_data, models_info_data, object_classes)
        if len(command) == 0:
            print(os.path.join(root_path, subfolder), "null add command")
        else:
            ##### Add Command Save
            os.makedirs(os.path.join(root_path, subfolder, "augmentation", "add_command"), exist_ok=True)
            np.savez(os.path.join(root_path, subfolder, "augmentation", "add_command", "source_boxes.npz"), **source_boxes_data)
            np.savez(os.path.join(root_path, subfolder, "augmentation", "add_command", "target_boxes.npz"), **target_boxes_data)
            text_command = {"command_type": "add",
                            "object_description": command[0],
                            "relative_description": command[1],
                            "reference_object_description": command[2]}
            with open(os.path.join(root_path, subfolder, "augmentation", "add_command", "text_command.json"), "w", encoding="utf-8") as f:
                json.dump(text_command, f, indent=4, ensure_ascii=False)

            ##### Remove Command Save
            os.makedirs(os.path.join(root_path, subfolder, "augmentation", "remove_command"), exist_ok=True)
            np.savez(os.path.join(root_path, subfolder, "augmentation", "remove_command", "source_boxes.npz"),
                     **target_boxes_data)
            np.savez(os.path.join(root_path, subfolder, "augmentation", "remove_command", "target_boxes.npz"),
                     **source_boxes_data)
            text_command = {"command_type": "remove",
                            "object_description": command[0],
                            "relative_description": command[1],
                            "reference_object_description": command[2]}
            with open(os.path.join(root_path, subfolder, "augmentation", "remove_command", "text_command.json"), "w",
                      encoding="utf-8") as f:
                json.dump(text_command, f, indent=4, ensure_ascii=False)

        source_boxes_data, target_boxes_data, command = move_function(boxes_data, descriptions_data, models_info_data, object_classes)
        if len(command) == 0:
            print(os.path.join(root_path, subfolder), "null move command")
        else:
            ##### Move Command Save
            os.makedirs(os.path.join(root_path, subfolder, "augmentation", "move_command"), exist_ok=True)
            np.savez(os.path.join(root_path, subfolder, "augmentation", "move_command", "source_boxes.npz"), **source_boxes_data)
            np.savez(os.path.join(root_path, subfolder, "augmentation", "move_command", "target_boxes.npz"), **target_boxes_data)
            text_command = {"command_type": "move",
                            "object_description": command[0],
                            "text_direction": command[1],
                            "move_distance": command[2]}
            with open(os.path.join(root_path, subfolder, "augmentation", "move_command", "text_command.json"), "w", encoding="utf-8") as f:
                json.dump(text_command, f, indent=4, ensure_ascii=False)

        source_boxes_data, target_boxes_data, command = rotate_function(boxes_data, descriptions_data, models_info_data, object_classes)
        if len(command) == 0:
            print(os.path.join(root_path, subfolder), "null rotate command")
        else:
            ##### Rotate Command Save
            os.makedirs(os.path.join(root_path, subfolder, "augmentation", "rotate_command"), exist_ok=True)
            np.savez(os.path.join(root_path, subfolder, "augmentation", "rotate_command", "source_boxes.npz"), **source_boxes_data)
            np.savez(os.path.join(root_path, subfolder, "augmentation", "rotate_command", "target_boxes.npz"), **target_boxes_data)
            text_command = {"command_type": "rotate",
                            "object_description": command[0],
                            "angle": command[1]}
            with open(os.path.join(root_path, subfolder, "augmentation", "rotate_command", "text_command.json"), "w", encoding="utf-8") as f:
                json.dump(text_command, f, indent=4, ensure_ascii=False)

        source_boxes_data, target_boxes_data, command = resize_function(boxes_data, descriptions_data, models_info_data, object_classes)
        if len(command) == 0:
            print(os.path.join(root_path, subfolder), "null resize command")
        else:
            ##### Resize Command Save
            os.makedirs(os.path.join(root_path, subfolder, "augmentation", "resize_command"), exist_ok=True)
            np.savez(os.path.join(root_path, subfolder, "augmentation", "resize_command", "source_boxes.npz"), **source_boxes_data)
            np.savez(os.path.join(root_path, subfolder, "augmentation", "resize_command", "target_boxes.npz"), **target_boxes_data)
            text_command = {"command_type": "resize",
                            "object_description": command[0],
                            "scaling_factor": command[1],
                            "scaling_type": command[2]}
            with open(os.path.join(root_path, subfolder, "augmentation", "resize_command", "text_command.json"), "w", encoding="utf-8") as f:
                json.dump(text_command, f, indent=4, ensure_ascii=False)

        source_boxes_data, target_boxes_data, command = replace_function(boxes_data, descriptions_data, models_info_data, object_classes, subfolder, subfolders)
        if len(command) == 0:
            print(os.path.join(root_path, subfolder), "null replace command")
        else:
            ##### Resize Command Save
            os.makedirs(os.path.join(root_path, subfolder, "augmentation", "replace_command"), exist_ok=True)
            np.savez(os.path.join(root_path, subfolder, "augmentation", "replace_command", "source_boxes.npz"), **source_boxes_data)
            np.savez(os.path.join(root_path, subfolder, "augmentation", "replace_command", "target_boxes.npz"), **target_boxes_data)
            text_command = {"command_type": "replace",
                            "object_description": command[0],
                            "target_object_description": command[1]}
            with open(os.path.join(root_path, subfolder, "augmentation", "replace_command", "text_command.json"), "w", encoding="utf-8") as f:
                json.dump(text_command, f, indent=4, ensure_ascii=False)