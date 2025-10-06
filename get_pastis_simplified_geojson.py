import geopandas as gpd
import pyproj
from shapely.ops import transform
import json

# 加载原始 GeoJSON 文件
input_file = "/scratch/zf281/pangaea-bench/data/PASTIS-HD/metadata.geojson"
gdf = gpd.read_file(input_file)

# 投影转换设置：从UTM转换到WGS84经纬度
project = pyproj.Transformer.from_crs(gdf.crs, "EPSG:4326", always_xy=True).transform

# 转换每个几何体到 WGS84 坐标系，保留原始形状
features = []
for idx, row in gdf.iterrows():
    # 转换几何体到 WGS84
    transformed_geometry = transform(project, row.geometry)

    # 创建 feature
    feature = {
        "type": "Feature",
        "properties": dict(row.drop('geometry')),  # 保留所有属性
        "geometry": transformed_geometry.__geo_interface__
    }
    features.append(feature)

# 创建新的 GeoJSON 数据结构
output_geojson = {
    "type": "FeatureCollection",
    "name": "pastis-hd",
    "crs": {
        "type": "name",
        "properties": {
            "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
        }
    },
    "features": features
}

# 输出新的 GeoJSON 文件
output_file = "/scratch/zf281/pangaea-bench/data/PASTIS-HD/transformed_metadata.geojson"
with open(output_file, "w") as f:
    json.dump(output_geojson, f, indent=4)

print(f"转换后的GeoJSON已保存至 {output_file}")
