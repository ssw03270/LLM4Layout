import os
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# utils 안에 아래 함수들이 정의되어 있다고 가정
from utils import (
    convert_epsg4326_to_utm,
    get_bbox_polygon,
    move_polygon_to_center,
    convert_utm_to_epsg3857,
    get_clopping_region,
    get_bounding_corners,
    get_bbox_parameters,
    create_rotated_bbox_polygon,
    generate_polygons_image,
    save_image,
    check_file_exists
)


def process_block_data(
        args
):
    """
    단일 블록 데이터에 대한 전처리를 수행하고,
    결과물을 저장(pkl, 이미지 등)하는 함수입니다.

    Parameters
    ----------
    args : tuple
        (data_idx,
         data,
         urban_name,
         dataset_dir_path,
         output_dir_path,
         bbox_quantiles,
         stats,
         utm_crs)

    Returns
    -------
    None
        (필요하다면 다른 값을 반환할 수도 있습니다.)
    """
    (data_idx,
     current_block_data,
     urban_name,
     dataset_dir_path,
     output_dir_path,
     bbox_quantiles,
     stats) = args

    if check_file_exists(os.path.join(output_dir_path, "image_dataset", "bldg_polygon_image", f"{urban_name}_{data_idx}_bldg_polygon_image.png")):
        return None

    wgs84_blk_geometry = current_block_data["blk_TIGER_geometry"]
    wgs84_bldg_geometry_list = current_block_data["bldg_3D_geometry"]

    # 건물 수 필터링 (개수가 너무 많거나 없는 경우 스킵)
    if len(wgs84_bldg_geometry_list) == 0 or len(wgs84_bldg_geometry_list) > 120:
        return None

    # 1) 블록 & 건물 EPSG:4326 -> UTM 변환
    utm_blk_polygon, utm_crs = convert_epsg4326_to_utm(wgs84_blk_geometry)
    if utm_blk_polygon is None:
        return None

    # 각 건물만 따로 변환
    utm_bldg_polygon_list = [convert_epsg4326_to_utm(bldg[1])[0] for bldg in wgs84_bldg_geometry_list]
    bldg_id_list = [bldg[0] for bldg in wgs84_bldg_geometry_list]
    bldg_height_list = [bldg[2] for bldg in wgs84_bldg_geometry_list]

    # 2) UTM 블록에 대한 bbox 구하기
    utm_blk_bbox_polygon, _, _, _, _ = get_bbox_polygon(utm_blk_polygon)
    moved_bbox_polygon, move_vec = move_polygon_to_center(utm_blk_bbox_polygon, (0, 0))
    _, bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = get_bbox_polygon(moved_bbox_polygon)

    # 3) bbox가 사전에 구해둔 사분위 범위(bbox_quantiles) 내에 있는지 확인
    #    (Q1~Q3 범위 밖이면 스킵)
    if (bbox_min_x < bbox_quantiles['bbox_min_x_list']['25%'] or
            bbox_min_x > bbox_quantiles['bbox_min_x_list']['75%'] or
            bbox_min_y < bbox_quantiles['bbox_min_y_list']['25%'] or
            bbox_min_y > bbox_quantiles['bbox_min_y_list']['75%'] or
            bbox_max_x < bbox_quantiles['bbox_max_x_list']['25%'] or
            bbox_max_x > bbox_quantiles['bbox_max_x_list']['75%'] or
            bbox_max_y < bbox_quantiles['bbox_max_y_list']['25%'] or
            bbox_max_y > bbox_quantiles['bbox_max_y_list']['75%']
    ):
        return None

    # 4) 블록 & 건물 폴리곤을 bbox 중심으로 이동
    moved_blk_polygon, _ = move_polygon_to_center(utm_blk_polygon, (0, 0), move_vec)
    moved_bldg_polygon_list = [
        move_polygon_to_center(bldg, (0, 0), move_vec)[0]
        for bldg in utm_bldg_polygon_list
    ]

    # 5) 건물 bbox 계산 및 회전 사각형 bbox 폴리곤 생성
    moved_bldg_bbox_list = [get_bbox_parameters(bldg) for bldg in moved_bldg_polygon_list]
    moved_bldg_bbox_polygon_list = [
        create_rotated_bbox_polygon(x, y, w, h, angle)
        for (x, y, w, h, angle) in moved_bldg_bbox_list
    ]

    # 6) EPSG:3857 크롭핑 영역 계산
    epsg3857_cropping_region_polygon = convert_utm_to_epsg3857(
        get_clopping_region(utm_blk_bbox_polygon.centroid, stats),
        utm_crs
    )
    epsg3857_cropping_region = get_bounding_corners(epsg3857_cropping_region_polygon)

    # 7) 블록 정보 dict를 저장
    blk_information_dict = {
        "urban_name": urban_name,
        "data_idx": data_idx,
        "bldg_id_list": bldg_id_list,
        "bldg_height_list": bldg_height_list,
        "moved_blk_polygon": moved_blk_polygon,
        "moved_bldg_polygon_list": moved_bldg_polygon_list,
        "moved_bldg_bbox_list": moved_bldg_bbox_list,
        "moved_bldg_bbox_polygon_list": moved_bldg_bbox_polygon_list,
        "wgs84_blk_geometry": wgs84_blk_geometry,
        "wgs84_bldg_geometry_list": wgs84_bldg_geometry_list,
        "epsg3857_cropping_region": epsg3857_cropping_region
    }

    # pkl 저장
    pkl_save_path = os.path.join(
        output_dir_path,
        "pkl_dataset",
        f"{urban_name}_{data_idx}_blk_information_dict.pkl"
    )
    os.makedirs(os.path.dirname(pkl_save_path), exist_ok=True)
    with open(pkl_save_path, 'wb') as file:
        pickle.dump(blk_information_dict, file)

    # 8) 이미지를 생성 후 저장
    bldg_polygon_image = generate_polygons_image(moved_blk_polygon, moved_bldg_polygon_list, stats)
    bldg_bbox_image = generate_polygons_image(moved_blk_polygon, moved_bldg_bbox_polygon_list, stats)
    blk_image = generate_polygons_image(moved_blk_polygon, moved_bldg_bbox_polygon_list, stats, draw_type='boundary')

    # 이미지 저장 경로가 없으면 생성
    bldg_polygon_image_path = os.path.join(output_dir_path, "image_dataset", "bldg_polygon_image")
    bldg_bbox_image_path = os.path.join(output_dir_path, "image_dataset", "bldg_bbox_image")
    blk_image_path = os.path.join(output_dir_path, "image_dataset", "blk_image")

    os.makedirs(bldg_polygon_image_path, exist_ok=True)
    os.makedirs(bldg_bbox_image_path, exist_ok=True)
    os.makedirs(blk_image_path, exist_ok=True)

    save_image(
        bldg_polygon_image,
        file_name=f"{urban_name}_{data_idx}_bldg_polygon_image.png",
        save_path=bldg_polygon_image_path
    )
    save_image(
        bldg_bbox_image,
        file_name=f"{urban_name}_{data_idx}_bldg_bbox_image.png",
        save_path=bldg_bbox_image_path
    )
    save_image(
        blk_image,
        file_name=f"{urban_name}_{data_idx}_blk_image.png",
        save_path=blk_image_path
    )

    return None


if __name__ == "__main__":

    dataset_dir_path = "F:\\Urban\\Our_Dataset"
    output_dir_path = "F:\\Urban\\Preprocessed_Dataset"

    # 도시(어반) 리스트
    urban_name_list = [
        f for f in os.listdir(dataset_dir_path)
        if os.path.isdir(os.path.join(dataset_dir_path, f))
    ]

    # 저장된 통계 데이터 불러오기 (bbox quantiles)
    quantiles_path = os.path.join(output_dir_path, "bounding_box_quantiles.pkl")
    with open(quantiles_path, "rb") as file:
        bbox_quantiles = pickle.load(file)

    min_x = bbox_quantiles['bbox_min_x_list']['25%']
    max_x = bbox_quantiles['bbox_max_x_list']['75%']
    min_y = bbox_quantiles['bbox_min_y_list']['25%']
    max_y = bbox_quantiles['bbox_max_y_list']['75%']

    min_abs_value = min(min_x, max_x, min_y, max_y, key=abs)

    stats = (-1 * abs(min_abs_value), -1 * abs(min_abs_value), abs(min_abs_value), abs(min_abs_value))

    # 도시별로 순회
    for urban_idx, urban_name in enumerate(urban_name_list):
        current_processed_urban_path = os.path.join(
            dataset_dir_path,
            urban_name,
            f"raw_geo/{urban_name}_blk_bldg_3d.pkl"
        )

        # 파일 존재 여부 확인
        if not os.path.exists(current_processed_urban_path):
            print(f"Warning: File not found - {current_processed_urban_path}")
            continue  # 파일이 없으면 다음 urban_name으로 넘어감

        # print(f"\n[INFO] Loading data for {urban_name}...")
        with open(current_processed_urban_path, "rb") as f:
            current_processed_urban_data_list = pickle.load(f)

        # 병렬 처리를 위한 준비 (data_idx와 함께 튜플 형태로 묶는다)
        data_args_list = []
        for data_idx, data in enumerate(current_processed_urban_data_list):
            # 하나의 블록 데이터에 대한 정보를 process_block_data()에 전달
            data_args_list.append(
                (
                    data_idx,
                    data,
                    urban_name,
                    dataset_dir_path,
                    output_dir_path,
                    bbox_quantiles,
                    stats
                )
            )

        # 멀티프로세싱으로 병렬 처리
        # max_workers는 시스템 사양에 따라 적절히 조정
        with ProcessPoolExecutor(max_workers=8) as executor:
            # tqdm으로 진행 상황 표시
            list(
                tqdm(
                    executor.map(process_block_data, data_args_list),
                    total=len(data_args_list),
                    desc=f"Processing {urban_name} ({urban_idx + 1}/{len(urban_name_list)})"
                )
            )

        # print(f"[INFO] Finished processing {urban_name}.\n")
