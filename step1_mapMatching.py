import argparse
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from shapely.wkt import loads
import geopandas as gpd
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
from tqdm import tqdm
from rtree import index
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--mode', choices=['traj', 'eta_task'], help='Mode of operation')
parser.add_argument('--denoise', action='store_true', help='Enable denoising')
args = parser.parse_args()

# 根据模式选择文件名
denoise = True if args.mode else false
file_suffix = 'eta_task' if args.mode == 'eta_task' else 'traj'


# 加载道路网络数据
df_road = pd.read_csv('./DM_2024_Dataset/road.csv')

# 将WKT字符串转换为Shapely的LineString对象
if 'geometry' in df_road.columns:
    df_road['geometry'] = df_road['geometry'].apply(lambda x: loads(x) if isinstance(x, str) else x)
else:
    print("Warning: No column named 'geometry' found.")
    exit(1)

gdf_road = gpd.GeoDataFrame(df_road, geometry='geometry')
gdf_road.set_crs(epsg=4326, inplace=True)

# 加载轨迹数据（新的格式）
df_gps = pd.read_csv(f'./runs/gcj_{file_suffix}.csv', dtype={'lon': float, 'lat': float, 'id': int, 'time': str})  # 确保文件路径正确

# 确认时间戳列名为 'time' 并转换为 datetime 类型
time_column_name = 'time'
if time_column_name in df_gps.columns:
    df_gps[time_column_name] = pd.to_datetime(df_gps[time_column_name], utc=True)  # 解析 ISO 8601 时间戳，并考虑时区
else:
    print(f"Warning: No column named '{time_column_name}' found.")
    exit(1)

# 按时间排序
df_gps.sort_values(time_column_name, inplace=True)

# 检查并处理重复的时间戳
if df_gps[time_column_name].duplicated().any():
    print("Warning: Duplicate timestamps found. Removing duplicates.")
    df_gps = df_gps.drop_duplicates(subset=[time_column_name])

# 设置时间列为索引
df_gps.set_index(time_column_name, inplace=True)

# 插入缺失的时间点，使用时间插值避免重复标签问题
df_gps_resampled = df_gps.resample('s').asfreq()  # 按秒重采样
df_gps_interpolated = df_gps_resampled.interpolate(method='time')  # 时间插值

if denoise:
    # 卡尔曼滤波器初始化
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([df_gps_interpolated['lon'].iloc[0],
                    df_gps_interpolated['lat'].iloc[0],
                    0, 0])  # 初始位置和速度
    kf.F = np.array([[1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])  # 状态转移矩阵
    kf.H = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]])  # 测量函数
    kf.P *= 1000  # 协方差矩阵
    kf.R = np.eye(2) * 5  # 测量噪声矩阵
    kf.Q = Q_discrete_white_noise(dim=4, dt=1.0, var=0.1)  # 过程噪声矩阵

filtered_positions = []
for idx, row in tqdm(df_gps_interpolated.iterrows(), total=len(df_gps_interpolated), desc="Kalman Filtering"):
    z = np.array([row['lon'], row['lat']])
    kf.predict()
    kf.update(z)
    filtered_positions.append(kf.x)

filtered_positions = np.array(filtered_positions)
df_filtered = pd.DataFrame(filtered_positions, columns=['lon', 'lat', 'vx', 'vy'])
df_filtered['time'] = df_gps_interpolated.index  # 添加时间索引作为新列

# 为df_filtered添加'id'列
df_filtered['point_id'] = range(len(df_filtered))

# 将过滤后的数据转换为GeoDataFrame
gdf_filtered = gpd.GeoDataFrame(df_filtered, geometry=gpd.points_from_xy(df_filtered['lon'], df_filtered['lat']))

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
for idx, row in tqdm(gdf_filtered.iterrows(), total=len(gdf_filtered), desc="Matching points"):  # 匹配所有点
    matched_road = match_point_to_nearest_road(row['geometry'], gdf_road, road_index)
    point_on_road = nearest_points(row['geometry'], matched_road['geometry'])[1]
    matched_points.append({
        'original_lon': row['lon'],
        'original_lat': row['lat'],
        'matched_road_id': matched_road['id'],
        'point_on_road_lon': point_on_road.x,
        'point_on_road_lat': point_on_road.y,
        'gps_point_id': idx
    })

# 创建包含匹配信息的新DataFrame
df_matched = pd.DataFrame(matched_points)

# 打印结果到CSV文件
file_suffix="den_"+file_suffix if denoise else file_suffix
output_csv = f'./runs/matched_points_{file_suffix}.csv'
df_matched.to_csv(output_csv, index=False)
print(f"Matched points saved to {output_csv}")

# 如果需要保存过滤后的GPS点数据到CSV文件
filtered_gps_output_csv ='./runs/filtered_gps_points.csv'
if not(os.path.exists(filtered_gps_output_csv)):
    df_filtered.to_csv(filtered_gps_output_csv, index=False)
    print(f"Filtered GPS points saved to {filtered_gps_output_csv}")