import os
import pickle

from tqdm import tqdm

from shapely.geometry import Polygon
import pyproj
from pyproj import CRS

from Urban.data_preprocessing.utils import get_bbox_polygon, move_polygon_to_center

def get_utm_crs(lon, lat):
    utm_zone = int((lon + 180) / 6) + 1
    hemisphere = 'north' if lat >= 0 else 'south'
    return CRS.from_dict({
        'proj': 'utm',
        'zone': utm_zone,
        'south': hemisphere == 'south'
    })

def convert_polygon_to_utm(polygon):
    centroid = polygon.centroid
    utm_crs = get_utm_crs(centroid.x, centroid.y)
    wgs84_crs = CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(wgs84_crs, utm_crs, always_xy=True)

    try:
        utm_coords = [transformer.transform(lon, lat) for lon, lat in polygon.exterior.coords]
    except Exception as e:
        print("Error during transformation:", e)
        return None

    utm_polygon = Polygon(utm_coords)
    return utm_polygon

dataset_dir_path = "F:\\City_Team\\COHO\\Our_Dataset"
output_dir_path = "F:\\City_Team\\COHO\\Preprocessed_Dataset"
urban_name_list = [f for f in os.listdir(dataset_dir_path) if os.path.isdir(os.path.join(dataset_dir_path, f))]

# min_x = -216.84764564596117
# min_y = -221.97276302985847
# max_x = 216.84764564596117
# max_y = 221.97276302985847
# state = (min_x, min_y, max_x, max_y)

bbox_min_x_list, bbox_min_y_list, bbox_max_x_list, bbox_max_y_list = [], [], [], []

for urban_idx, urban_name in enumerate(urban_name_list):
    current_processed_urban_path = os.path.join(dataset_dir_path, urban_name, f"raw_geo/{urban_name}_blk_bldg_3d.pkl")

    # 파일 존재 여부 확인
    if not os.path.exists(current_processed_urban_path):
        print(f"Warning: File not found - {current_processed_urban_path}")
        continue  # 파일이 없으면 다음 urban_name으로 넘어감

    with open(current_processed_urban_path, "rb") as f:
        current_processed_urban_data_list = pickle.load(f)   # blk_TIGER_geometry, blk_TIGER_properties, bldg_3D_geometry, blk_id, bldg_total_num

    for data_idx in tqdm(range(len(current_processed_urban_data_list)),
                         desc=f"Processing {urban_name} ({urban_idx+1}/{len(urban_name_list)})"):
        wgs84_blk_geometry = current_processed_urban_data_list[data_idx]["blk_TIGER_geometry"]
        wgs84_bldg_geometry_list = current_processed_urban_data_list[data_idx]["bldg_3D_geometry"]

        if len(wgs84_bldg_geometry_list) == 0 or len(wgs84_bldg_geometry_list) > 120:
            continue

        utm_blk_polygon = convert_polygon_to_utm(wgs84_blk_geometry)
        if utm_blk_polygon is None:
            continue

        bbox_polygon, _, _, _, _ = get_bbox_polygon(utm_blk_polygon)
        moved_bbox_polygon, move_vec = move_polygon_to_center(bbox_polygon, (0, 0))

        _, bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = get_bbox_polygon(moved_bbox_polygon)

        bbox_min_x_list.append(bbox_min_x)
        bbox_min_y_list.append(bbox_min_y)
        bbox_max_x_list.append(bbox_max_x)
        bbox_max_y_list.append(bbox_max_y)

bbox_data = {
    "bbox_min_x_list": bbox_min_x_list,
    "bbox_min_y_list": bbox_min_y_list,
    "bbox_max_x_list": bbox_max_x_list,
    "bbox_max_y_list": bbox_max_y_list,
}

with open(os.path.join(output_dir_path, "bounding_box_statistics.pkl"), "wb") as f:
    pickle.dump(bbox_data, f)
