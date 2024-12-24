import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 加载数据
road_data = pd.read_csv('./runs/road_filled.csv')
traj_data = pd.read_csv('./DM_2024_Dataset/traj.csv')
match_traj_data = pd.read_csv('./runs/matched_points_traj.csv')

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
    return ((coord_floats1[0] - coord_floats2[0])**2 + (coord_floats1[1] - coord_floats2[1])**2)**0.5
def distance_diff(series):
    return calculate_distance(series.iloc[0], series.iloc[1])

traj_data['distance_diff'] = traj_data.groupby('trajectory_id')['coordinates'].apply(distance_diff).bfill()
# 填充每个轨迹的最后一个一个轨迹点的距离差为0
traj_data['distance_diff'] = traj_data.groupby('trajectory_id')['distance_diff'].transform(lambda x: x.fillna(0))

#计算速度()
traj_data['speed'] = traj_data.apply(lambda row: row['distance_diff'] / row['time_diff'] if row['time_diff'] != 0 else 0, axis=1)

# 合并road_data和traj_data，添加road.csv中的特征
merged_data = pd.merge(match_traj_data, road_data, left_on='matched_road_id', right_on='id', how='left')
merged_data = pd.merge(merged_data, traj_data, left_on='gps_point_id', right_on='point_id', how='left')


# 准备特征和标签
features = ['highway','lanes','tunnel','bridge','roundabout','oneway']

labels = ['speed']

# 选择特征和标签
X = merged_data[features]
y = merged_data[labels]

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

# 预测
eta_data=pd.read_csv('./DM_2024_Dataset/eta_task.csv')
match_eta_data=pd.read_csv('./runs/matched_points_eta_task.csv')



eta_data['distance_diff'] = traj_data.groupby('trajectory_id')['coordinates'].apply(distance_diff).bfill()
# 填充每个轨迹的最后一个一个轨迹点的距离差为0
eta_data['distance_diff'] = traj_data.groupby('trajectory_id')['distance_diff'].transform(lambda x: x.fillna(0))

# 合并road_data和eta_data，添加road.csv中的特征
merged_data = pd.merge(match_eta_data, road_data, left_on='matched_road_id', right_on='id', how='left')
merged_data = pd.merge(merged_data, traj_data, left_on='gps_point_id', right_on='point_id', how='left')

#投入预测
# missing_indices = eta_data['time'].isnull()
features_to_predict = ['highway','lanes','tunnel','bridge','roundabout','oneway']
X_all = merged_data[features_to_predict]
eta_data['predicted_speed'] = model.predict(X_all)

# 计算时间，基于速度和距离差
eta_data['time_to_travel'] = eta_data.apply(lambda row: row['distance_diff'] / row['predicted_speed']/60 if row['predicted_speed'] != 0 else 0, axis=1)
print(eta_data['time_to_travel'])
print(eta_data['distance_diff'])
print(eta_data['predicted_speed'])

# 基于第一个点的时间，计算所有点的时间
eta_data['current_time'] = pd.NaT
eta_data['time'] = pd.to_datetime(eta_data['time'])
# eta_data['current_time'] = eta_data.groupby('trajectory_id').apply(
#     lambda group: group['time'].shift(1) + pd.to_timedelta(group['time_to_travel'].shift(1), unit='minutes')
# )
for trajectory_id, group in eta_data.groupby('trajectory_id'):
    # 第一个点的时间设置为原始的'time'
    # eta_data.loc[eta_data['trajectory_id'] == trajectory_id, 'current_time'].iloc[0] = eta_data.loc[eta_data['trajectory_id'] == trajectory_id, 'time'].iloc[0]
    # 获取每个轨迹的第一个点的索引
    first_index = eta_data[eta_data['trajectory_id'] == trajectory_id].index[0]
    # 第一个点的时间设置为原始的'time'
    eta_data.at[first_index, 'current_time'] = eta_data.at[first_index, 'time']
    # 从第二个点开始，时间是前一点的时间加上旅行时间
    for i in range(1, len(group)):
        prev_index = eta_data[eta_data['trajectory_id'] == trajectory_id].index[i-1]
        current_index = eta_data[eta_data['trajectory_id'] == trajectory_id].index[i]
        prev_time = eta_data.at[prev_index, 'current_time']
        time_to_travel = eta_data.at[current_index, 'time_to_travel']
        # 确保time_to_travel是数值类型，然后将其转换为timedelta
        time_to_travel_td = pd.to_timedelta(time_to_travel, unit='minutes') if pd.notnull(time_to_travel) else pd.Timedelta(0)
        eta_data.at[current_index, 'current_time'] = prev_time + time_to_travel_td

eta_data['time'] = eta_data['current_time']

# 将'current_time'列转换为datetime类型
eta_data['time'] = pd.to_datetime(eta_data['current_time'])

#输出
first_column_name = eta_data.columns[0]
selected_columns = [first_column_name, 'type', 'time', 'entity_id','coordinates','trajectory_id']
eta_data_selected = eta_data[selected_columns]
eta_data_selected.to_csv('./runs/eta_task_filled.csv', index=False)
