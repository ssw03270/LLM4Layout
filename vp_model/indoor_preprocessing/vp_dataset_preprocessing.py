import os

import json
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from shapely.geometry import Polygon, box
from shapely.affinity import translate

from math import cos, sin

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

def visualize_polygons(polygons, floor, view_bound):
    """
    Shapely Polygon 리스트를 시각화하는 함수.

    Args:
        polygons (list[Polygon]): Shapely Polygon 객체 리스트.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    x, y = floor.exterior.xy
    ax.plot(x, y, label="Floor", color="black")

    for polygon in polygons:
        x, y = polygon.exterior.xy  # 폴리곤 외곽선의 x, y 좌표
        ax.plot(x, y, label="Polygon")  # 외곽선 그리기
        ax.fill(x, y, alpha=0.4)  # 투명한 색으로 채우기

    # 시각화 범위 설정
    minx, maxx, miny, maxy = view_bound
    print(view_bound)
    ax.set_xlim(minx - 0.5, maxx + 0.5)
    ax.set_ylim(miny - 0.5, maxy + 0.5)

    ax.set_aspect('equal')
    ax.invert_yaxis()  # y축 뒤집기
    ax.axis('off')  # 축 숨기기

    return fig

dataset_folder = "F:\\Indoor\\InstructScene"
room_types = ["threed_front_bedroom"]

rendered_image_file_name = "blender_rendered_scene_256\\topdown.png"
layout_npz_file_name = "boxes.npz"
dataset_stats_file_name = "dataset_stats.txt"

output_folder = "outputs\\layouts"
real_output_path = "real_images"
layout_output_path = "target_images"

if not os.path.exists(os.path.join(output_folder, real_output_path)):
    os.makedirs(os.path.join(output_folder, real_output_path))
if not os.path.exists(os.path.join(output_folder, layout_output_path)):
    os.makedirs(os.path.join(output_folder, layout_output_path))

for room_type in room_types:
    dataset_path = os.path.join(dataset_folder, room_type)
    dataset_stats_file_path = os.path.join(dataset_path, dataset_stats_file_name)
    with open(dataset_stats_file_path, "r") as file:
        dataset_stats = json.load(file)
        class_labels_text = dataset_stats["class_labels"]
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

    for subfolder in subfolders:
        data_folder = os.path.join(dataset_path, subfolder)
        rendered_image_file_path = os.path.join(data_folder, rendered_image_file_name)
        layout_npz_file_path = os.path.join(data_folder, layout_npz_file_name)

        rendered_image_file = Image.open(rendered_image_file_path)
        rendered_image_output_path = os.path.join(output_folder, real_output_path, f"{subfolder}_rendered.png")
        rendered_image_file.save(rendered_image_output_path)

        layout_npz = np.load(layout_npz_file_path)
        class_labels = [class_labels_text[label] for label in np.argmax(layout_npz["class_labels"], axis=1)]
        translations = layout_npz["translations"]
        sizes = layout_npz["sizes"]
        angles = layout_npz["angles"]

        furnitures = create_polygons(translations, sizes, angles)
        floor = get_combined_bounding_box(furnitures)
        furnitures, floor = center_and_translate(furnitures, floor)

        layout_fig = visualize_polygons(furnitures, floor, view_bound)
        layout_output_path = os.path.join(output_folder, layout_output_path, f"{subfolder}_layout.png")
        layout_fig.savefig(layout_output_path, bbox_inches="tight")

        exit()