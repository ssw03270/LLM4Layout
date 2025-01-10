import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import pyproj

# 변환된 UTM 좌표
utm_polygon = Polygon([
    (382905.5120965361, 4324325.245474416),
    (383098.00552427775, 4324325.245474416),
    (383098.00552427775, 4324518.8757108),
    (382905.5120965361, 4324518.8757108),
    (382905.5120965361, 4324325.245474416)
])

# UTM Zone 33N 설정 (필요에 따라 조정)
utm_crs = "EPSG:32633"
wgs84_crs = "EPSG:4326"

transformer = pyproj.Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)
wgs84_coords = [transformer.transform(x, y) for x, y in utm_polygon.exterior.coords]

# WGS84 좌표로 폴리곤 생성
wgs84_polygon = Polygon(wgs84_coords)

# 지도에 시각화
x, y = wgs84_polygon.exterior.xy
plt.figure(figsize=(8, 8))
plt.plot(x, y, marker='o')
plt.title("Transformed Polygon in WGS84")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()
