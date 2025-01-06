import pandas as pd
import geopandas as gpd
from shapely.wkt import loads
from shapely.ops import nearest_points
import matplotlib.pyplot as plt
from tqdm import tqdm  # 添加进度条以跟踪进度
from rtree import index  # 使用R树索引加速最近邻查找

# 加载道路网络数据
df_road = pd.read_csv('./DM_2024_Dataset/road.csv')  # 确保文件路径正确

# 将WKT字符串转换为Shapely的LineString对象
df_road['geometry'] = df_road['geometry'].apply(loads)

# 创建GeoDataFrame并设置坐标参考系统
gdf_road = gpd.GeoDataFrame(df_road, geometry='geometry')
gdf_road.set_crs(epsg=4326, inplace=True)  # 假设数据是WGS84坐标系

# 加载轨迹数据（新的格式）
df_gps = pd.read_csv('gcj_jump.csv', dtype={'lon': float, 'lat': float, 'id': int})  # 确保文件路径正确

# 将轨迹数据转换为GeoDataFrame
gdf_gps = gpd.GeoDataFrame(df_gps, geometry=gpd.points_from_xy(df_gps['lon'], df_gps['lat']))

# 构建R树索引以加速最近邻查找
road_index = index.Index()
for idx, road in gdf_road.iterrows():
    road_index.insert(idx, road['geometry'].bounds)

# 地图匹配：找到每个GPS点最近的道路
def match_point_to_nearest_road(point, roads, road_idx):
    nearest_road_id = list(road_idx.nearest(point.bounds, 1))[0]
    nearest_road = roads.iloc[nearest_road_id]
    return nearest_road

matched_points = []
for idx, row in tqdm(gdf_gps.iterrows(), total=len(gdf_gps), desc="Matching points"):  # 匹配所有点
    matched_road = match_point_to_nearest_road(row['geometry'], gdf_road, road_index)
    point_on_road = nearest_points(row['geometry'], matched_road['geometry'])[1]
    matched_points.append({
        'original_lon': row['lon'],
        'original_lat': row['lat'],
        'matched_road_id': matched_road['id'],
        'point_on_road_lon': point_on_road.x,
        'point_on_road_lat': point_on_road.y,
        'gps_point_id': row['id']
    })

# 创建包含匹配信息的新DataFrame
df_matched = pd.DataFrame(matched_points)

# 打印结果到CSV文件
output_csv = 'matched_points_jump.csv'
df_matched.to_csv(output_csv, index=False)
print(f"Matched points saved to {output_csv}")