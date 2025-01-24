import os
from tqdm import tqdm
from typing import *

import nltk
from nltk.corpus import cmudict
nltk.download('cmudict')

import pickle
import json
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from shapely.geometry import Polygon, box
from shapely.affinity import translate

from math import cos, sin

def starts_with_vowel_sound(word, pronunciations=cmudict.dict()):
    for syllables in pronunciations.get(word, []):
        return syllables[0][-1].isdigit()


def get_article(word):
    word = word.split(" ")[0]
    article = "an" if starts_with_vowel_sound(word) else "a"
    return article

def reverse_rel(rel: str) -> str:
    return {
        "above": "below",
        "below": "above",
        "in front of": "behind",
        "behind": "in front of",
        "left of": "right of",
        "right of": "left of",
        "closely in front of": "closely behind",
        "closely behind": "closely in front of",
        "closely left of": "closely right of",
        "closely right of": "closely left of"
    }[rel]

def convert_to_past_participle(verb: str) -> str:
    """동사를 과거 분사 형태로 변환하는 간단한 함수."""
    irregular_verbs = {
        "Place": "placed",
        "Put": "put",
        "Position": "positioned",
        "Arrange": "arranged",
        "Add": "added",
        "Set up": "set up",
        "Hang": "hung",
        "Install": "installed"
    }
    return irregular_verbs.get(verb, verb + "ed")  # 기본적으로 -ed를 붙임

def fill_templates(
    room_type: str,
    desc: Dict[str, List],
    object_types: List[str], predicate_types: List[str],
    object_descs: Optional[List[str]]=None,
    seed: Optional[int]=None,
    return_obj_ids=False
) -> Tuple[str, Dict[int, int], List[Tuple[int, int, int]], List[Tuple[str, str]]]:
    room_type = room_type.replace("threed_front_", "")

    if object_descs is None:
        assert object_types is not None

    if seed is not None:
        np.random.seed(seed)

    obj_class_ids = desc["obj_class_ids"]  # map from object index to class id

    # Describe the relations between the main objects and others
    selected_relation_indices = np.random.choice(
        len(desc["obj_relations"]),
        min(np.random.choice([1, 2]), len(desc["obj_relations"])),  # select 1 or 2 relations
        replace=False
    )
    selected_relations = [desc["obj_relations"][idx] for idx in selected_relation_indices]
    selected_relations = [
        (int(obj_class_ids[s]), int(p), int(obj_class_ids[o]))
        for s, p, o in selected_relations
    ]  # e.g., [(4, 2, 18), ...]; 4, 18 are class ids; 2 is predicate id
    selected_descs = []
    selected_sentences = []
    selected_object_ids = []  # e.g., [0, ...]; 0 is object id
    for idx in selected_relation_indices:
        s, p, o = desc["obj_relations"][idx]
        s, p, o = int(s), int(p), int(o)
        if object_descs is None:
            s_name = object_types[obj_class_ids[s]].replace("_", " ")
            o_name = object_types[obj_class_ids[o]].replace("_", " ")
        else:
            if np.random.rand() < 0.75:
                s_name = object_descs[s]
            else:  # 25% of the time to use the object type as the description
                s_name = object_types[obj_class_ids[s]].replace("_", " ")
                s_name = f"{get_article(s_name)} {s_name}"  # "a" or "an" is added
            if np.random.rand() < 0.75:
                o_name = object_descs[o]
            else:
                o_name = object_types[obj_class_ids[o]].replace("_", " ")
                o_name = f"{get_article(o_name)} {o_name}"

        p_str = predicate_types[p]
        rev_p_str = reverse_rel(p_str)

        if p_str in ["left of", "right of"]:
            if np.random.rand() < 0.5:
                p_str = "to the " + p_str
                rev_p_str = "to the " + rev_p_str
        elif p_str in ["closely left of", "closely right of"]:
            if np.random.rand() < 0.25:
                p_str = "closely to the " + p_str.split(" ")[-2] + " of"
                rev_p_str = "closely to the " + rev_p_str.split(" ")[-2] + " of"
            elif np.random.rand() < 0.5:
                p_str = "to the close " + p_str.split(" ")[-2] + " of"
                rev_p_str = "to the close " + rev_p_str.split(" ")[-2] + " of"
            elif np.random.rand() < 0.75:
                p_str = "to the near " + p_str.split(" ")[-2] + " of"
                rev_p_str = "to the near " + rev_p_str.split(" ")[-2] + " of"

        # Choose whether to describe s relative to o or o relative to s
        if np.random.rand() < 0.5:
            verbs = ["Position", "Set up", "Arrange", "Add", "Place", "Install"]
            if "lamp" in s_name:
                verbs += ["Hang"]
            verb = verbs[np.random.choice(len(verbs))]
            verb_past = convert_to_past_participle(verb)
            sentence = f"{s_name} is {verb_past} {p_str} {o_name}."
            selected_descs.append((s_name, o_name))
            selected_object_ids.append(s)
        else:  # 50% of the time to reverse the order
            verbs = ["Position", "Set up", "Arrange", "Add", "Place", "Install"]
            if "lamp" in o_name:
                verbs += ["Hang"]
            verb = verbs[np.random.choice(len(verbs))]
            verb_past = convert_to_past_participle(verb)
            sentence = f"{o_name} is {verb_past} {rev_p_str} {s_name}."
            selected_descs.append((o_name, s_name))
            selected_object_ids.append(o)

        selected_sentences.append(sentence)

    # 시작 서두 추가
    text = f"This image shows top view of a {room_type} where "

    conjunctions = [". Additionally, ", ". Next, ", ". Furthermore, ", ". Finally, ", " "]
    for i, sentence in enumerate(selected_sentences):
        if i == 0:
            # Remove the period at the end and connect with 'where'
            sentence = sentence[:-1]  # Remove the period
            text += sentence
        else:
            conjunction = conjunctions[np.random.choice(len(conjunctions))]
            while conjunction.strip() == "Finally," and i != len(selected_sentences) - 1:
                # "Finally" should be used only in the last sentence
                conjunction = conjunctions[np.random.choice(len(conjunctions))]
            if conjunction != " ":
                sentence = sentence[0].lower() + sentence[1:]
            text += conjunction + sentence

    return_obj_ids = return_obj_ids
    if return_obj_ids:
        return text, selected_relations, selected_descs, selected_object_ids
    else:
        return text, selected_relations, selected_descs  # return `selected_relations`, `selected_descs` for evaluation


def predicate_types():
    return [
        "above", "left of", "in front of",
        "closely left of", "closely in front of",
        "below", "right of", "behind",
        "closely right of", "closely behind"
    ]

def create_polygons(translations, sizes, angles):
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


def get_combined_bounding_box(polygons):
    """
    여러 Shapely Polygon 객체를 감싸는 단일 바운딩 박스를 구하는 함수.

    Args:
        polygons (list[Polygon]): Shapely Polygon 객체 리스트.

    Returns:
        Polygon: 단일 바운딩 박스(Shapely Polygon 객체).
    """
    # 초기 바운딩 박스 값을 None으로 설정
    min_x, min_y, max_x, max_y = None, None, None, None

    for polygon in polygons:
        # 각 폴리곤의 바운딩 박스를 얻음
        bounds = polygon.bounds  # (min_x, min_y, max_x, max_y)

        # 현재 최소/최대값과 비교하여 업데이트
        if min_x is None or bounds[0] < min_x:
            min_x = bounds[0]
        if min_y is None or bounds[1] < min_y:
            min_y = bounds[1]
        if max_x is None or bounds[2] > max_x:
            max_x = bounds[2]
        if max_y is None or bounds[3] > max_y:
            max_y = bounds[3]

    # 최종 바운딩 박스 생성
    bounding_box = box(min_x, min_y, max_x, max_y)
    return bounding_box

def center_and_translate(furnitures, floor):
    """
    floor의 중심을 계산하고, 이를 기준으로 모든 furniture와 floor를 새로운 중심으로 옮기는 함수.

    Args:
        furnitures (list[Polygon]): Shapely Polygon 객체 리스트.
        floor (Polygon): 바운딩 박스를 나타내는 Shapely Polygon 객체.

    Returns:
        tuple: (translated_furnitures, translated_floor)
    """
    # floor의 중심 계산
    floor_bounds = floor.bounds  # (minx, miny, maxx, maxy)
    floor_center_x = (floor_bounds[0] + floor_bounds[2]) / 2
    floor_center_y = (floor_bounds[1] + floor_bounds[3]) / 2

    # 중심 이동 값 계산
    dx = -floor_center_x
    dy = -floor_center_y

    # 모든 furniture와 floor를 중심으로 옮기기
    translated_furnitures = [translate(furniture, xoff=dx, yoff=dy) for furniture in furnitures]
    translated_floor = translate(floor, xoff=dx, yoff=dy)

    return translated_furnitures, translated_floor

def compute_text_offset(current_polygon, other_polygons, offset_distance=0.5):
    """
    현재 폴리곤이 다른 폴리곤들과 겹치는 경우,
    단, 겹치는 폴리곤이 현재 폴리곤보다 작을 경우에만,
    겹치는 모든 (더 작은) 폴리곤의 중심으로부터 멀어지는 방향의 벡터 합산
    (정규화 후 offset_distance 만큼의 이동)을 반환한다.
    """
    current_center = np.array([current_polygon.centroid.x, current_polygon.centroid.y])
    displacement = np.array([0.0, 0.0])
    count = 0

    for other in other_polygons:
        # 자기 자신은 제외
        if current_polygon == other:
            continue
        # 현재 폴리곤보다 작은(면적이 작은) 폴리곤만 고려
        if current_polygon.area <= other.area:
            continue

        # 폴리곤 간 겹침이 발생하는지 확인 (경계 포함)
        if current_polygon.intersects(other):
            other_center = np.array([other.centroid.x, other.centroid.y])
            diff = current_center - other_center  # 다른 폴리곤 중심으로부터 현재 중심의 방향
            norm = np.linalg.norm(diff)
            # 두 중심이 너무 가까워 0이 되는 경우 회피
            if norm > 1e-6:
                displacement += diff / norm
                count += 1

    if count > 0:
        # 평균 displacement 방향 구하기
        avg_disp = displacement / count
        norm = np.linalg.norm(avg_disp)
        if norm > 1e-6:
            # offset_distance 만큼 이동시키기
            offset = (avg_disp / norm) * offset_distance
            return offset
    # 겹치는 (더 작은) 폴리곤이 없으면, 기본적으로 0 offset
    return np.array([0.0, 0.0])


def visualize_polygons(polygons, floor, class_label_texts, class_label_indices, view_bound):
    """
    Shapely Polygon 리스트를 시각화하는 함수.

    Args:
        polygons (list[Polygon]): Shapely Polygon 객체 리스트.
        floor (Polygon): 바닥면을 나타내는 Shapely Polygon.
        class_label_texts (list[str]): 각 폴리곤에 대응하는 클래스 레이블 텍스트.
        class_label_indices (list[int]): 각 폴리곤에 대응하는 클래스 레이블 인덱스.
        view_bound (tuple): 시각화 범위를 (minx, maxx, miny, maxy) 형태로 지정.
    """
    n_classes = 30
    cmap = plt.cm.get_cmap('nipy_spectral', n_classes)
    color_palette = [cmap(i) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # 바닥면 그리기
    x_floor, y_floor = floor.exterior.xy
    ax.plot(x_floor, y_floor, label="Floor", color="black")

    # 텍스트 위치 겹침 해결을 위해 각 폴리곤에 대해 offset 적용
    for polygon, label_text, label_idx in zip(polygons, class_label_texts, class_label_indices):
        x_poly, y_poly = polygon.exterior.xy  # 폴리곤 외곽선의 x, y 좌표
        color = color_palette[label_idx]  # 색상 선택

        ax.plot(x_poly, y_poly, color=color)  # 외곽선 그리기
        ax.fill(x_poly, y_poly, color=color, alpha=0.4)  # 채우기

        # 폴리곤 중심 계산
        centroid_x, centroid_y = polygon.centroid.x, polygon.centroid.y

        # 다른 폴리곤들과 겹치는 경우 오프셋 적용
        offset = compute_text_offset(polygon, polygons)
        new_text_pos = (centroid_x + offset[0], centroid_y + offset[1])

        # 텍스트 내의 "_"를 줄바꿈 문자로 대체
        label_text_modified = label_text.replace("_", "\n")
        ax.text(new_text_pos[0], new_text_pos[1], label_text_modified,
                color="black", ha="center", va="center", fontsize=7, fontweight="bold")

    # 시각화 범위 설정
    minx, maxx, miny, maxy = view_bound
    ax.set_xlim(minx - 0.1, maxx + 0.1)
    ax.set_ylim(miny - 0.1, maxy + 0.1)

    ax.set_aspect('equal')
    ax.invert_yaxis()  # y축 뒤집기
    ax.axis('off')  # 축 숨기기

    return fig

dataset_folder = "E:\\Resources\\IndoorSceneSynthesis\\InstructScene"
room_types = ["threed_front_bedroom"]

rendered_image_file_name = "blender_rendered_scene_256\\topdown.png"
layout_npz_file_name = "boxes.npz"
description_pkl_file_name = "descriptions.pkl"
dataset_stats_file_name = "dataset_stats.txt"
models_info_pkl_file_name = "models_info.pkl"

output_folder = "outputs\\indoor_layouts"
real_output_path = "real_images"
layout_output_path = "target_images"
description_output_path = "text_description"

if not os.path.exists(os.path.join(output_folder, real_output_path)):
    os.makedirs(os.path.join(output_folder, real_output_path))
if not os.path.exists(os.path.join(output_folder, layout_output_path)):
    os.makedirs(os.path.join(output_folder, layout_output_path))
if not os.path.exists(os.path.join(output_folder, description_output_path)):
    os.makedirs(os.path.join(output_folder, description_output_path))

for room_type in room_types:
    dataset_path = os.path.join(dataset_folder, room_type)
    dataset_stats_file_path = os.path.join(dataset_path, dataset_stats_file_name)

    with open(dataset_stats_file_path, "r") as file:
        dataset_stats = json.load(file)
        class_labels_book = dataset_stats["class_labels"]
        minx, maxx, minz, maxz = (dataset_stats["bounds_sizes"][0], dataset_stats["bounds_sizes"][3],
                                  dataset_stats["bounds_sizes"][2], dataset_stats["bounds_sizes"][5])
        # 각 방향의 크기 계산
        x_range = maxx - minx
        z_range = maxz - minz

        # 더 큰 값을 선택
        max_range = max(x_range, z_range)
        view_bound = (-max_range, max_range, -max_range, max_range)

    # 해당 경로에서 폴더 리스트 가져오기
    if os.path.exists(dataset_path):
        subfolders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    else:
        print(f"Path does not exist: {dataset_path}")
        continue

    for subfolder in tqdm(subfolders):
        data_folder = os.path.join(dataset_path, subfolder)
        rendered_image_file_path = os.path.join(data_folder, rendered_image_file_name)
        layout_npz_file_path = os.path.join(data_folder, layout_npz_file_name)
        description_pkl_file_path = os.path.join(data_folder, description_pkl_file_name)
        models_info_pkl_file_path = os.path.join(data_folder, models_info_pkl_file_name)

        rendered_image_file = Image.open(rendered_image_file_path)
        rendered_image_output_path = os.path.join(output_folder, real_output_path, f"{subfolder}_rendered.png")
        rendered_image_file.save(rendered_image_output_path)

        layout_npz = np.load(layout_npz_file_path)
        class_label_indices = np.argmax(layout_npz["class_labels"], axis=1)
        class_label_texts = [class_labels_book[label] for label in class_label_indices]
        translations = layout_npz["translations"]
        sizes = layout_npz["sizes"]
        angles = layout_npz["angles"]
        model_ids = layout_npz["jids"]
        gpt_captions = []

        with open(description_pkl_file_path, 'rb') as file:
            description_pkl = pickle.load(file)
        description = {"obj_class_ids": description_pkl["obj_class_ids"],
                       "obj_relations": description_pkl["obj_relations"]}

        with open(models_info_pkl_file_path, 'rb') as file:
            models_info_pkl = pickle.load(file)

        for model_id in model_ids:
            for model_info in models_info_pkl:
                if model_info["model_id"] == model_id:
                    gpt_captions.append(model_info["chatgpt_caption"])
                    break
        scene_text_description, _, _ = fill_templates(
            room_type=room_type,
            desc=description,
            object_types=class_labels_book,
            predicate_types=predicate_types(),
            object_descs=gpt_captions,
            seed=None
        )

        furnitures = create_polygons(translations, sizes, angles)
        floor = get_combined_bounding_box(furnitures)
        furnitures, floor = center_and_translate(furnitures, floor)

        layout_fig = visualize_polygons(furnitures, floor, class_label_texts, class_label_indices, view_bound)
        layout_image_output_path = os.path.join(output_folder, layout_output_path, f"{subfolder}_layout.png")
        layout_fig.savefig(layout_image_output_path, bbox_inches=None, dpi=100, pad_inches=0)
        plt.close(layout_fig)  # 해당 figure 닫기

        scene_text_description_output_path = os.path.join(output_folder, description_output_path, f"{subfolder}_description.txt")
        with open(scene_text_description_output_path, "w", encoding="utf-8") as file:
            file.write(scene_text_description)

        desired_size = (256, 256)
        img = Image.open(layout_image_output_path)
        img_size = img.size  # (width, height)
        if img_size != desired_size:
            print(f"경고: 이미지 크기가 예상과 다릅니다! 예상: {desired_size}, 실제: {img_size}")
            exit()