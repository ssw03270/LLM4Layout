import numpy as np
import math
import os

import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.affinity import translate

import pyproj
from pyproj import CRS

from PIL import Image
import io

def get_utm_crs(lon, lat):
    utm_zone = int((lon + 180) / 6) + 1
    hemisphere = 'north' if lat >= 0 else 'south'
    return CRS.from_dict({
        'proj': 'utm',
        'zone': utm_zone,
        'south': hemisphere == 'south'
    })

def convert_epsg4326_to_utm(polygon):
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
    return utm_polygon, utm_crs


def convert_utm_to_epsg3857(polygon, utm_crs):
    """
    UTM 좌표계를 EPSG:3857로 변환하는 함수.

    :param polygon: UTM 좌표계로 표현된 Shapely Polygon 객체
    :param utm_crs: UTM 좌표계의 CRS 객체
    :return: EPSG:3857로 변환된 Polygon 객체
    """
    epsg3857_crs = CRS.from_epsg(3857)
    transformer = pyproj.Transformer.from_crs(utm_crs, epsg3857_crs, always_xy=True)

    try:
        # 변환 수행
        epsg3857_coords = [transformer.transform(x, y) for x, y in polygon.exterior.coords]
    except Exception as e:
        print("Error during transformation:", e)
        return None

    # EPSG:3857로 변환된 Polygon 생성
    epsg3857_polygon = Polygon(epsg3857_coords)
    return epsg3857_polygon

def check_file_exists(file_path):
    """
    특정 파일이 존재하는지 확인하는 함수.

    Args:
        file_path (str): 확인하려는 파일의 경로.

    Returns:
        bool: 파일이 존재하면 True, 그렇지 않으면 False.
    """
    return os.path.exists(file_path)

def plot_polygons(single_polygon, polygon_list):
    # 단일 폴리곤과 리스트에 있는 폴리곤 시각화
    fig, ax = plt.subplots(figsize=(8, 8))

    # 단일 폴리곤 그리기 (빨간색)
    x, y = single_polygon.exterior.xy
    ax.fill(x, y, alpha=0.5, fc='red', ec='black', label='Single Polygon')

    # 폴리곤 리스트 그리기 (파란색)
    for i, poly in enumerate(polygon_list):
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.3, fc='blue', ec='black', label=f'Polygon {i + 1}' if i == 0 else None)

    # 축 설정 및 범례 추가
    ax.set_aspect('equal')
    ax.legend()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Shapely Polygons Visualization')
    plt.grid(True)
    plt.show()

def get_bbox_polygon(polygon):
    """
    Shapely Polygon 객체를 감싸는 bounding box 반환 함수.

    Parameters:
        polygon (Polygon): Shapely Polygon 객체.

    Returns:
        tuple: (min_x, min_y, max_x, max_y)
    """
    if not isinstance(polygon, Polygon):
        raise TypeError("Input must be a Shapely Polygon object.")

    min_x, min_y, max_x, max_y = polygon.bounds

    # Bounding box를 Polygon 객체로 생성
    bbox_polygon = Polygon([
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
        (min_x, min_y)  # 닫힌 폴리곤이 되도록 시작점과 동일한 좌표로 끝냄
    ])

    return bbox_polygon, min_x, min_y, max_x, max_y

def move_polygon_to_center(polygon, new_center, move_vec=None):
    """
    Shapely Polygon 객체를 지정한 중심으로 이동.

    Parameters:
        polygon (Polygon): 이동할 Shapely Polygon 객체.
        new_center (tuple): 새로운 중심 좌표 (new_x, new_y).

    Returns:
        Polygon: 지정한 중심으로 이동된 Shapely Polygon 객체.
    """
    if not isinstance(polygon, Polygon):
        raise TypeError("Input must be a Shapely Polygon object.")

    if not isinstance(new_center, tuple) or len(new_center) != 2:
        raise ValueError("new_center must be a tuple of (x, y) coordinates.")

    # 폴리곤의 현재 중심 계산
    current_center = polygon.centroid
    current_x, current_y = current_center.x, current_center.y

    # 이동해야 할 거리 계산
    if move_vec is None:
        dx = new_center[0] - current_x
        dy = new_center[1] - current_y
    else:
        dx, dy = move_vec

    # 폴리곤 이동
    moved_polygon = translate(polygon, xoff=dx, yoff=dy)

    return moved_polygon, (dx, dy)

def get_minimum_rotated_bbox(polygon: Polygon):
    """
    Given a Shapely Polygon, returns its minimum rotated bounding box as a new Polygon.

    Args:
        polygon (Polygon): Input Shapely Polygon.

    Returns:
        Polygon: Minimum rotated bounding box of the input Polygon.
    """
    # Compute the minimum rotated bounding box
    min_rotated_bbox = polygon.minimum_rotated_rectangle

    return min_rotated_bbox

# 통계 계산 함수
def calculate_statistics(data_list):
    stats = {
        "min": np.min(data_list),
        "max": np.max(data_list),
        "mean": np.mean(data_list),
        "median": np.median(data_list),
        "std_dev": np.std(data_list),
        "center": (np.min(data_list) + np.max(data_list)) / 2
    }
    return stats

def generate_polygons_image(blk_polygon, bldg_polygons, stats, draw_type="all"):
    """
    Displays the floor plan polygon and object polygons for visual inspection.
    """

    minx, miny, maxx, maxy = stats

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')  # Set the figure background color to black
    ax.set_facecolor('black')        # Set the axes background color to black

    x_blk, y_blk = blk_polygon.exterior.xy
    ax.fill(x_blk, y_blk, color='white', alpha=1, label='Room Polygon')

    if draw_type != "boundary":
        for idx, poly in enumerate(bldg_polygons):

            x_obj, y_obj = poly.exterior.xy
            ax.plot(x_obj, y_obj, color='black')
            ax.fill(x_obj, y_obj, color='black', alpha=0.5)

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

    buf.close()
    return img

def create_rotated_bbox_polygon(x, y, w, h, angle):
    """
    Create a rotated bounding box polygon given its center, dimensions, and rotation angle.

    Args:
        x (float): Center x-coordinate of the bounding box.
        y (float): Center y-coordinate of the bounding box.
        w (float): Width of the bounding box.
        h (float): Height of the bounding box.
        angle (float): Rotation angle in degrees (measured counter-clockwise from x-axis).

    Returns:
        Polygon: A Shapely Polygon representing the rotated bounding box.
    """
    # Convert angle to radians
    angle_rad = np.radians(angle)

    # Calculate the corner points of the unrotated rectangle
    dx = w / 2
    dy = h / 2
    corners = np.array([
        [-dx, -dy],
        [dx, -dy],
        [dx, dy],
        [-dx, dy]
    ])

    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    # Rotate corners and translate to the center (x, y)
    rotated_corners = corners @ rotation_matrix.T + np.array([x, y])

    # Create and return a Shapely Polygon
    return Polygon(rotated_corners)

def get_bbox_parameters(polygon: Polygon):
    """
    Given a Shapely Polygon, calculate its minimum rotated bounding box parameters.

    Args:
        polygon (Polygon): Input Shapely Polygon.

    Returns:
        tuple: (x, y, w, h, angle) where:
            - x (float): Center x-coordinate of the bounding box.
            - y (float): Center y-coordinate of the bounding box.
            - w (float): Width of the bounding box.
            - h (float): Height of the bounding box.
            - angle (float): Rotation angle of the bounding box in degrees.
    """
    # Get the minimum rotated bounding box
    min_rotated_bbox = polygon.minimum_rotated_rectangle

    # Extract the coordinates of the bounding box corners
    bbox_coords = np.array(min_rotated_bbox.exterior.coords)[:-1]  # Exclude the last duplicate point

    # Calculate the width and height
    edge1 = bbox_coords[1] - bbox_coords[0]
    edge2 = bbox_coords[2] - bbox_coords[1]

    w = np.linalg.norm(edge1)  # Length of the first edge
    h = np.linalg.norm(edge2)  # Length of the second edge

    if w >= h:
        longer_edge = edge1
        w = np.linalg.norm(edge1)
        h = np.linalg.norm(edge2)
    else:
        longer_edge = edge2
        h = np.linalg.norm(edge1)
        w = np.linalg.norm(edge2)

    # Calculate the center of the bounding box
    x = bbox_coords[:, 0].mean()
    y = bbox_coords[:, 1].mean()

    # Calculate the rotation angle (angle of the first edge with respect to the x-axis)
    angle = np.degrees(np.arctan2(longer_edge[1], longer_edge[0])) % 360

    return (x, y, w, h, angle)

def save_image(image_array, file_name, save_path):
    save_path = f"{save_path}/{file_name}"
    image_array.save(save_path)

def get_clopping_region(centroid, stats):
    minx, miny, maxx, maxy = stats
    x, y = centroid.xy

    polygon = Polygon([
        (x + minx, y + miny),  # Bottom-left corner
        (x + maxx, y + miny),  # Bottom-right corner
        (x + maxx, y + maxy),  # Top-right corner
        (x + minx, y + maxy),  # Top-left corner
        (x + minx, y + miny)   # Close the polygon (back to the start)
    ])

    return polygon

def get_bounding_corners(polygon):
    # Get the bounds of the polygon
    minx, miny, maxx, maxy = polygon.bounds

    # Define bottom-left and top-right corners
    bottom_left = (minx, miny)
    top_right = (maxx, maxy)

    return bottom_left, top_right