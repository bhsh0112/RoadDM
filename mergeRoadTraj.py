import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json

# 读取数据
traj_data = pd.read_csv('./DM_2024_Dataset/traj.csv')
road_data = pd.read_csv('./runs/road_filled.csv')
with open('road_neighbor_cd.json', 'r') as f:
    road_neighbors = json.load(f)

# 定义haversine函数
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r

def convert_list_to_floats(coord_list):
    # 将列表中的元素组合成一个字符串
    coord_str = ''.join(coord_list)
    
    # 移除非数字字符（逗号、空格和点）
    coord_str = coord_str.replace(',', '')
    coord_str = coord_str.replace(' ', '')
    
    # 分割字符串以提取经度和纬度
    # 假设坐标格式为 "104.10242,30.662"，我们可以通过逗号分割
    coords = coord_str.split('.')
    
    # 将分割后的字符串重新组合，确保每个坐标部分正确
    lon_str = coords[0] + '.' + coords[1] if len(coords) > 1 else coords[0]
    lat_str = coords[2] + '.' + coords[3] if len(coords) > 3 else coords[2]
    
    # 转换为浮点数
    lon = float(lon_str)
    lat = float(lat_str)
    
    return lon, lat

# 定义匹配函数
def find_nearest_segment(coord, road_data, road_neighbors, start_id=None):
    # 如果未提供起始路段ID，则随机选择一个路段作为起始点
    if start_id is None:
        start_id = list(road_data['id'])[0]
    
    # 使用邻接关系快速找到最近的路段
    min_distance = float('inf')
    nearest_segment_id = None
    visited = set()  # 记录已访问的路段，避免重复访问
    
    def dfs(current_id):
        nonlocal min_distance, nearest_segment_id
        if current_id in visited or current_id not in road_data['id'].values:
            return
        visited.add(current_id)
        
        segment_lon = road_data.loc[road_data['id'] == current_id, 'm_lon'].iloc[0]
        segment_lat = road_data.loc[road_data['id'] == current_id, 'm_lat'].iloc[0]
        distance = haversine(lon, lat, segment_lon, segment_lat)
        
        if distance < min_distance:
            min_distance = distance
            nearest_segment_id = current_id
        
        # 遍历邻接路段
        for neighbor in road_neighbors.get(current_id, []):
            dfs(neighbor)
    
    lon, lat = convert_list_to_floats(coord)
    dfs(start_id)
    return nearest_segment_id

matched_file='./DM_2024_Dataset/matched_traj.csv'
# 遍历traj.csv中的每一条数据
for index, row in traj_data.iterrows():
    coord = tuple(row['coordinates'])  # 确保coordinates是元组格式
    segment_id = find_nearest_segment(coord, road_data)
    if segment_id is not None:
        segment_attributes = road_data.loc[road_data['id'] == segment_id].iloc[0]
        matched_row=pd.concat([row, segment_attributes], axis=1)
        matched_row.to_csv(matched_file, index=False, mode='a', header=False)
