import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# 假设'traj.csv'文件在同一目录下
traj_data = pd.read_csv('./DM_2024_Dataset/traj.csv')

# 将时间字符串转换为时间戳（秒）
traj_data['time'] = traj_data['time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ').timestamp())

# 将坐标字符串转换为浮点数列表
traj_data['coordinates'] = traj_data['coordinates'].apply(lambda x: np.fromstring(x.strip('[]').replace(' ', ''), sep=','))

# 计算速度
def calculate_speed(row, data):
    if row.name == 0:
        return np.nan  # 第一个点没有前一个点，无法计算速度
    elif row.name >= 1: 
        prev_row = data.iloc[row.name - 1]
        
    if prev_row['trajectory_id'] != row['trajectory_id']:
        return np.nan  # 不同轨迹的速度无法比较
    distance = np.sqrt((row['coordinates'][0] - prev_row['coordinates'][0])**2 + (row['coordinates'][1] - prev_row['coordinates'][1])**2)
    time_diff = row['time'] - prev_row['time']
    speed = distance / time_diff if time_diff > 0 else 0
    return speed

# 应用计算速度函数
traj_data['speed'] = traj_data.apply(calculate_speed, data=traj_data, axis=1)

# 去除速度为NaN的行
traj_data = traj_data.dropna(subset=['speed'])

# 准备数据
X = traj_data[['speed']]
y = traj_data['time']  # 这里我们用时间戳作为目标变量，实际应用中可能需要转换为预计到达时间

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 模型训练
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 模型预测
y_pred = model.predict(X_test_scaled)

# 模型评估
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 假设我们有一个新的轨迹数据集'eta_task.csv'需要预测到达时间
# 这里我们用traj_data作为示例，实际应用中应该使用eta_task.csv
eta_task_data = traj_data.copy()

# 计算新轨迹数据集的速度特征
eta_task_data['speed'] = eta_task_data.apply(calculate_speed, data=eta_task_data, axis=1)
eta_task_data = eta_task_data.dropna(subset=['speed'])

# 准备新轨迹数据集的特征
eta_task_features = eta_task_data[['speed']]
eta_task_features_scaled = scaler.transform(eta_task_features)

# 预测新轨迹数据集的到达时间
eta_predictions = model.predict(eta_task_features_scaled)

# 将预测结果添加到eta_task_data中
eta_task_data['predicted_arrival_time'] = eta_predictions
eta_task_data[['trajectory_id', 'predicted_arrival_time']].to_csv('eta_task_predicted.csv', index=False)