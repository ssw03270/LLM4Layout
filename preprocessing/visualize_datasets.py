import os
import numpy as np
import random
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from shapely.affinity import rotate, translate


def load_stats(stats_path):
    """
    dataset_stats.txt (JSON 포맷)에서 오브젝트 타입(object_types) 정보를 불러오는 예시 함수.
    """
    import json
    with open(stats_path, "r", encoding='utf-8') as f:
        stats = json.load(f)
    return stats


def visualize_scene(scene_data, stats):
    """
    하나의 씬(scene)에 대한 오브젝트들을 2D Top-View로 시각화합니다.

    Parameters:
    -----------
    scene_data: dict
        - "class_labels": 오브젝트 클래스의 원-핫(One-hot) 인코딩 리스트
        - "sizes": 오브젝트 크기 [width, height, depth]
        - "angles": 오브젝트 회전 각도 [angle]
        - "translations": 오브젝트 위치 [x, y, z]
        - "room_mask": (선택) 방 구조를 담은 mask (원본 코드에서는 사용하지 않고 있음)
    stats: dict
        - "object_types": 라벨 인덱스별 오브젝트 카테고리 이름 리스트
    """
    object_types = stats['object_types']

    # 시각화 준비
    fig, ax = plt.subplots()

    # 씬 내부의 오브젝트들을 순회하며 2D 폴리곤을 생성
    for class_label, size, angle, translation in zip(
            scene_data['class_labels'],
            scene_data['sizes'],
            scene_data['angles'],
            scene_data['translations']
    ):
        # class_label은 원-핫 인코딩이므로, 어떤 인덱스가 1인지 찾기
        label_idx = np.where(class_label)[0][0]
        category_name = object_types[label_idx]

        # size: [width, height, depth], 여기서는 (x, y, z) 형태를 가정
        width, height, depth = size[0] * 2, size[1] * 2, size[2] * 2  # 원본 코드처럼 *2
        dx, dy, dz = translation

        # Shapely를 사용하여 2D Polygon(Top-View) 구성
        # (width, depth) 기준으로 XY plane 대신 XZ plane을 사용하므로,
        # Polygon의 y좌표 자리에 depth를 넣어서 "바닥에서 본 형태"를 그립니다.
        base_rect = Polygon([
            (-width / 2, -depth / 2),
            (width / 2, -depth / 2),
            (width / 2, depth / 2),
            (-width / 2, depth / 2)
        ])

        # 회전은 라디안 단위로, base_rect를 중심에서 angle[0]만큼 회전 (Y축 기준 회전 가정)
        # Shapely rotate 함수는 기본적으로 origin='center'일 때,
        # polygon의 centroid를 기준으로 회전합니다. (origin 파라미터로 조절 가능)
        rect_rotated = rotate(base_rect, angle[0], use_radians=True)

        # X방향으로 dx, Y방향(실제로는 Z축)을 위해 dz를 사용해 평행이동
        rect_translated = translate(rect_rotated, xoff=dx, yoff=dz)

        # 폴리곤 시각화
        x_coords, y_coords = rect_translated.exterior.xy
        ax.fill(x_coords, y_coords, alpha=0.4, label=category_name)

        # 개별 오브젝트 테두리를 그리거나, 텍스트로 카테고리 표시도 가능
        ax.plot(x_coords, y_coords, color='black', linewidth=0.8)

    # 중복 레이블(legend) 제거를 위해 set() 사용
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', fontsize=8)

    # 시각화 설정
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Z-axis (top view)')
    plt.title("Scene Visualization (Top-View)")
    plt.tight_layout()
    plt.show()


def visualize_saved_dataset(npz_folder_path, stats_path, split='train'):
    """
    저장된 npy 데이터(train.npy, val.npy, test.npy 등)를 하나 불러와서 시각화합니다.

    Parameters:
    -----------
    npz_folder_path: str
        - train.npy, val.npy, test.npy 등 생성된 파일이 저장된 폴더 경로
    stats_path: str
        - dataset_stats.txt (JSON 형식) 파일 경로
    split: str
        - 'train', 'val', 'test' 중 하나
    """
    # stats (오브젝트 카테고리) 불러오기
    stats = load_stats(stats_path)

    # ex) 'train.npy' 불러오기
    dataset_file = os.path.join(npz_folder_path, f"{split}.npy")
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"{dataset_file} 파일이 존재하지 않습니다.")

    # npy 파일 로드 (list of dict 형태)
    dataset = np.load(dataset_file, allow_pickle=True)

    # 랜덤하게 한 씬(scene)을 골라 시각화
    scene_idx = random.randint(0, len(dataset) - 1)
    scene_data = dataset[scene_idx]

    print(f"[{split}] 데이터셋에서 index={scene_idx} 씬을 시각화합니다.")
    visualize_scene(scene_data, stats)


if __name__ == "__main__":
    """
    예시 실행:
        1) `npz_folder_path`에 train.npy, val.npy, test.npy가 있는 폴더 경로를 지정
        2) `stats_path`에 dataset_stats.txt (JSON 포맷) 경로를 지정
        3) split 인자를 'train', 'val', 'test' 중 하나로 설정
        4) python visualize_dataset.py 처럼 실행
    """
    # 예시 경로 (사용 환경에 맞춰 수정)
    npz_folder_path = "../datasets/threed_front_livingroom"
    stats_path = "E:/Resources/IndoorSceneSynthesis/InstructScene/threed_front_livingroom/dataset_stats.txt"

    visualize_saved_dataset(npz_folder_path, stats_path, split='train')
