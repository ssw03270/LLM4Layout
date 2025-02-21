import os

import matplotlib.pyplot as plt
from PIL import Image

from shapely.geometry import Polygon, box
from shapely.affinity import translate

from math import cos, sin

import pickle
import numpy as np
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from data import get_raw_dataset, filter_function
import json

from tqdm import tqdm

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
        if label_text in ["ceiling_lamp", "pendant_lamp"]:
            continue

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

def save_messages(messages, output_file):
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(messages, json_file, ensure_ascii=False, indent=4)

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config

def generate_dataset(raw_data, room_type, task_type="remaining values", output_path="./dataset", split="train"):
    rendered_image_path = "rendered_image.png"
    layout_image_path = "layout_image.png"
    instruction_path = "instruction.txt"
    message_path = "message.jsonl"

    user_prompt = ("I want to generate layout in {Domain} style. "
                   "Please generate the layout according to the following text condition."
                   "\"{TextInstruction}\". ")
    code_template = """```html
<html>
    <body>
{code}
    </body>
</html>
```"""
    assistant_prompt = """```html
<html>
    <body>
{code}
    </body>
</html>
```"""
    messages = []

    models_info_paths = raw_data._path_to_models_info

    text_class_labels = raw_data._class_labels
    min_bounds, max_bounds = raw_data._sizes
    minx, maxx, minz, maxz = (min_bounds[0], max_bounds[0],
                              min_bounds[2], max_bounds[2])
    max_range = max(maxx - minx, maxz - minz)
    view_bound = (-max_range, max_range, -max_range, max_range)

    min_translation = raw_data._centroids[0]
    for data_idx, data in enumerate(tqdm(raw_data)):
        data_path = data.image_path.replace("InstructScene", "topview_instruction/save_dir").replace("rendered_scene_256.png", "")
        tag = data_path.replace("dataset/topview_instruction/save_dir/", "")

        base_save_path = os.path.join(output_path, split, tag)
        os.makedirs(base_save_path, exist_ok=True)

        class_labels = data.class_labels
        translations = data.translations
        sizes = data.sizes
        angles = data.angles
        captions = data.captions

        ###### model infos
        models_info_path = models_info_paths[data_idx]
        with open(models_info_path, "rb") as f:
            data_models_info = pickle.load(f)

        ###### rendered figure generation
        rendered_image_file_path = os.path.join(data_path, "topdown.png")
        rendered_image = Image.open(rendered_image_file_path)

        rendered_image.save(os.path.join(base_save_path, rendered_image_path))

        ###### text instruction generation
        text_instruction_file_path = os.path.join(data_path, "instruction.txt")
        with open(text_instruction_file_path, "r") as f:
            text_instruction = f.read()

        with open(os.path.join(base_save_path, instruction_path), "w", encoding="utf-8") as file:
            file.write(text_instruction)

        ###### layout figure generation
        furnitures = create_polygons(translations, sizes, angles)
        floor = get_combined_bounding_box(furnitures)
        furnitures, floor = center_and_translate(furnitures, floor)

        class_label_indices = np.argmax(class_labels, axis=1)
        class_label_texts = [text_class_labels[label] for label in class_label_indices]
        layout_image = visualize_polygons(furnitures, floor, class_label_texts, class_label_indices, view_bound)

        layout_image.savefig(os.path.join(base_save_path, layout_image_path), bbox_inches=None, dpi=100, pad_inches=0)
        plt.close(layout_image)  # 해당 figure 닫기

        ###### text generation
        element_count = len(class_labels)
        gt_layout_text = ""
        masked_layout_text = ""
        for element_idx in range(element_count):
            class_label = class_labels[element_idx]
            text_class_label = text_class_labels[np.argmax(class_label)]
            trans = translations[element_idx] - min_translation
            size = sizes[element_idx]
            angle = np.rad2deg(angles[element_idx][0])
            caption = captions[element_idx]

            data_model_info = data_models_info[element_idx]
            objfeat_vq_indices = data_model_info["objfeat_vq_indices"]
            text_objfeat = ""
            for vq_index in objfeat_vq_indices:
                text_objfeat += f"[img:{vq_index}]"

            gt_element_text = (f"<rect data-category=\"{text_class_label}\" "
                               f"data-image={text_objfeat} "
                               f"transform=\"translate3d({trans[0]:.2f}, {trans[1]:.2f}, {trans[2]:.2f}) "
                               f"scale3d({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f}) "
                               f"rotateY({angle:.2f})\"/>")
            masked_element_text = (f"<rect data-category={text_class_label} "
                                   f"data-image=[img:FILL_idx][img:FILL_idx][img:FILL_idx][img:FILL_idx] "
                                   f"transform=\"translate3d(<FILL_x>, <FILL_y>, <FILL_z>) "
                                   f"scale3d(<FILL_x>, <FILL_y>, <FILL_z>) "
                                   f"rotateY(<FILL_deg>)\"/>")

            gt_layout_text += f"        {gt_element_text}"
            masked_layout_text += f"        {masked_element_text}"

            if element_idx < element_count - 1:
                gt_layout_text += "\n"
                masked_layout_text += "\n"

        _user_prompt = user_prompt.format(Domain=room_type, TextInstruction=text_instruction)
        _code_template = code_template.format(code=masked_layout_text)
        _user_prompt += "\n" + _code_template

        _assistant_prompt = assistant_prompt.format(code=gt_layout_text)

        message = {"instruction": _user_prompt, "input": "", "output": _assistant_prompt, "tag": tag}
        messages.append(message)

        with open(os.path.join(base_save_path, message_path), 'w', encoding='utf-8') as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")

    return messages

def main():
    room_types = ["bedroom", "diningroom", "livingroom"]
    for room_type in room_types:
        config_file = f"./configs/{room_type}_sg2sc_diffusion_objfeat.yaml"
        config = load_config(config_file)

        train_raw = get_raw_dataset(
            config["data"],
            filter_function(
                config["data"],
                split=config["training"].get("splits", ["train", "val"])
            ),
            path_to_bounds=None,
            split=config["training"].get("splits", ["train", "val"]),
        )
        train_messages = generate_dataset(train_raw, room_type, output_path="./dataset", split="train")

        val_raw = get_raw_dataset(
            config["data"],
            filter_function(
                config["data"],
                split=config["validation"].get("splits", ["test"])
            ),
            path_to_bounds=None,
            split=config["validation"].get("splits", ["test"])
        )
        val_messages = generate_dataset(val_raw, room_type, output_path="./dataset", split="test")

        save_messages(train_messages, f"{room_type}_train_dataset.json")
        save_messages(val_messages, f"{room_type}_val_dataset.json")

if __name__ == "__main__":
    main()


