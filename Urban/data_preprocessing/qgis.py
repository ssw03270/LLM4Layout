import os
import glob
import pickle
from tqdm import tqdm

from qgis.core import (
    QgsProject,
    QgsCoordinateReferenceSystem,
    QgsPrintLayout,
    QgsLayoutItemMap,
    QgsLayoutSize,
    QgsLayoutExporter,
    QgsRectangle,
    QgsUnitTypes,
)
from qgis.utils import iface
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QColor

# ---------------------------
# 1) 먼저, Layout과 Map Item을 초기화하는 함수
# ---------------------------
def setup_layout_and_map():
    """QGIS 프로젝트에서 재사용할 Layout과 Map Item을 하나 만들고 반환"""
    project = QgsProject.instance()
    layout_manager = project.layoutManager()
    layout_name = "Satellite_Export_Layout_3857"

    # 기존에 동일 이름의 레이아웃이 있으면 제거
    existing_layout = layout_manager.layoutByName(layout_name)
    if existing_layout:
        layout_manager.removeLayout(existing_layout)

    # 새 레이아웃 생성
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName(layout_name)
    layout_manager.addLayout(layout)

    # 페이지(용지) 설정
    page = layout.pageCollection().pages()[0]
    page.setPageSize(QgsLayoutSize(200, 200, QgsUnitTypes.LayoutMillimeters))

    # 맵 아이템 추가
    map_item = QgsLayoutItemMap(layout)
    map_item.setRect(0, 0, 200, 200)  # mm 단위
    layout.addLayoutItem(map_item)

    # 배경 흰색 설정
    map_item.setBackgroundColor(QColor(255, 255, 255))

    # 맵 아이템의 CRS 설정
    target_crs = QgsCoordinateReferenceSystem("EPSG:3857")
    map_item.setCrs(target_crs)

    return layout, map_item

# ---------------------------
# 2) bbox 하나를 받아서 이미지를 내보내는 함수
# ---------------------------
def export_one_bbox_3857(map_item, bbox_3857, output_path, google_satellite_layer):
    """
    map_item: 이미 생성해둔 QgsLayoutItemMap
    bbox_3857: ((x_min, y_min), (x_max, y_max)) (EPSG:3857 좌표)
    output_path: 출력 이미지 경로
    google_satellite_layer: 'Google Satellite' 레이어 객체
    """
    (x_min, y_min), (x_max, y_max) = bbox_3857

    # 맵 범위 설정
    rect = QgsRectangle(x_min, y_min, x_max, y_max)
    map_item.zoomToExtent(rect)
    map_item.setExtent(rect)

    # 표시할 레이어(여기서는 Google Satellite만)
    map_item.setLayers([google_satellite_layer])

    # Layout Exporter
    # (Layout은 map_item.parentLayout()로 얻을 수 있음)
    layout = map_item.layout()
    exporter = QgsLayoutExporter(layout)
    export_settings = QgsLayoutExporter.ImageExportSettings()
    export_settings.imageSize = QSize(256, 256)  # 원하는 해상도
    export_settings.pngCompressionLevel = 0        # 화질 우선

    # 이미지 출력
    result = exporter.exportToImage(output_path, export_settings)
    # 필요 시 결과 확인 가능
    # if result == QgsLayoutExporter.Success:
    #     print(f"Export success: {output_path}")
    # else:
    #     print(f"Export failed: {output_path}")


# ---------------------------
# 3) 메인 루프
# ---------------------------
dataset_dir_path = "F:\\Urban\\Preprocessed_Dataset\\pkl_dataset"
output_dir_path = "F:\\Urban\\Preprocessed_Dataset\\image_dataset\\satellite_image"

# QGIS 프로젝트 인스턴스
project = QgsProject.instance()

# "Google Satellite" 레이어 찾기
google_satellite_layers = project.mapLayersByName("Google Satellite")
if not google_satellite_layers:
    print("에러: 'Google Satellite' 레이어를 찾을 수 없습니다.")
    raise SystemExit
google_satellite_layer = google_satellite_layers[0]

# 출력 폴더 생성
os.makedirs(output_dir_path, exist_ok=True)

# (A) 레이아웃 & 맵아이템 미리 셋업 (1회)
layout, map_item = setup_layout_and_map()

# (B) pkl 파일 목록
pkl_file_paths = glob.glob(os.path.join(dataset_dir_path, '*.pkl'), recursive=True)

# (C) pkl 파일을 하나씩 읽어서 처리
for file_path in tqdm(pkl_file_paths, desc="Processing PKL files"):
    try:
        output_path = file_path.replace("pkl_dataset", "image_dataset\\satellite_image")
        output_path = output_path.replace("_blk_information_dict.pkl", "_google_satellite_image.png")

        # 이미 존재한다면(이미 처리했다면) 스킵할 수도 있음
        if os.path.exists(output_path):
            continue

        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            epsg3857_cropping_region = data["epsg3857_cropping_region"]
            urban_name = data["urban_name"]
            data_idx = data["data_idx"]

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        continue

    # 출력 파일명 생성
    file_name = f"{urban_name}_{data_idx}_google_satellite_image.png"
    output_path = os.path.join(output_dir_path, file_name)

    # (D) 단일 bbox에 대해 export 수행
    export_one_bbox_3857(
        map_item=map_item,
        bbox_3857=epsg3857_cropping_region,
        output_path=output_path,
        google_satellite_layer=google_satellite_layer
    )

# (E) 모든 작업이 끝난 후 레이아웃 제거(선택)
layout_manager = project.layoutManager()
layout_manager.removeLayout(layout)
print("=== 모든 크롭 이미지 내보내기를 완료했습니다. ===")
