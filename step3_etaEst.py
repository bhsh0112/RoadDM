import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 加载数据
road_data = pd.read_csv('./runs/road_filled.csv')
traj_data = pd.read_csv('./DM_2024_Dataset/traj.csv')
match_data = pd.read_csv('matched_points_all.csv')

# 将时间戳转换为实际时间
traj_data['time'] = pd.to_datetime(traj_data['time'])

# 计算每个轨迹点和后一个点之间的时间差（单位：分钟）
traj_data['time_diff'] = traj_data.groupby('trajectory_id')['time'].diff().dt.total_seconds() / 60

# 填充每个轨迹的最后一个一个轨迹点的时间差为0
traj_data['time_diff'] = traj_data.groupby('trajectory_id')['time_diff'].transform(lambda x: x.fillna(0))



# 计算相同轨迹上相邻轨迹点的距离差
def calculate_distance(coord1, coord2):
    coord_list = eval(coord1)
    coord_floats1 = [float(num) for num in coord_list]
    coord_list = eval(coord2)
    coord_floats2 = [float(num) for num in coord_list]
    print(((coord_floats1[0] - coord_floats2[0])**2 + (coord_floats1[1] - coord_floats2[1])**2)**0.5)
    return ((coord_floats1[0] - coord_floats2[0])**2 + (coord_floats1[1] - coord_floats2[1])**2)**0.5
def distance_diff(series):
    return calculate_distance(series.iloc[0], series.iloc[1])

traj_data['distance_diff'] = traj_data.groupby('trajectory_id')['coordinates'].apply(distance_diff).bfill()
print(traj_data['distance_diff'])
# 填充每个轨迹的最后一个一个轨迹点的距离差为0
traj_data['distance_diff'] = traj_data.groupby('trajectory_id')['distance_diff'].transform(lambda x: x.fillna(0))

#TODO:希望以speed为输出，而非时间差
traj_data['speed'] = traj_data.apply(lambda row: row['distance_diff'] / row['time_diff'] if row['time_diff'] != 0 else 0, axis=1)
print(traj_data['speed'])

# 合并road_data和traj_data，添加road.csv中的特征
merged_data = pd.merge(match_data, road_data, left_on='matched_road_id', right_on='id', how='left')
merged_data = pd.merge(merged_data, traj_data, left_on='gps_point_id', right_on='point_id', how='left')
# print(merged_data.size())


# 准备特征和标签
features = ['highway','lanes','tunnel','bridge','roundabout','oneway']

labels = ['speed']

# 选择特征和标签
X = merged_data[features]
print(X)
y = merged_data[labels]
print(y)

# 去除含有NaN值的行
X = X.dropna()
y = y.dropna()

# 训练测试集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测和评估
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# 预测新轨迹的到达时间
# 假设我们有一个新轨迹的数据
# new_trajectory = traj_data[traj_data['trajectory_id'] == 999999]  # 假设999999是新轨迹的ID
# new_trajectory_features = new_trajectory[features + ['distance_diff']]

# # 预测新轨迹每个点的时间差
# time_diffs = model.predict(new_trajectory_features)

# # 计算到达时间
# new_trajectory['predicted_time'] = new_trajectory['time'] + pd.to_timedelta(time_diffs, unit='minutes')

# print(new_trajectory[['coordinates', 'predicted_time']])