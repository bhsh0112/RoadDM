import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def merge_traj_and_match(traj_file, match_file, output_file):
    traj_data = pd.read_csv(traj_file)
    match_data = pd.read_csv(match_file)

    if len(traj_data) != len(match_data):
        raise ValueError("不匹配")

    traj_data['matched_road_id'] = match_data['matched_road_id']
    traj_data.to_csv(output_file, index=False)
    # print("合并")

def add_match_road_id(jump_task_file, matched_points_jump_file, output_file):
    """
    将 jump_task.csv 和 matched_points_jump.csv 合并，并生成新的文件
    """
    jump_task = pd.read_csv(jump_task_file)
    matched_points_jump = pd.read_csv(matched_points_jump_file)

    result_data = []
    matched_points_index = 0  # 用于遍历 matched_points_jump 中的行

    for i, row in jump_task.iterrows():
        # 如果是该轨道的终点，match_road_id 留空
        if i == len(jump_task) - 1 or row['trajectory_id'] != jump_task.iloc[i + 1]['trajectory_id']:
            continue
        else:
            # 对应 matched_points_jump 中的 road_id
            if matched_points_index < len(matched_points_jump):
                result_data.append({
                    **row.to_dict(),
                    'matched_road_id': matched_points_jump.iloc[matched_points_index]['matched_road_id']
                })
                matched_points_index += 1

    result_df = pd.DataFrame(result_data)
    result_df.to_csv(output_file, index=False)
    # print("合并")

traj_file = './DM_2024_Dataset/traj.csv'
match_file = './runs/matched_points_traj.csv'
traj_output_file = 'cleaned_new_traj.csv'

jump_task_file = './DM_2024_Dataset/jump_task.csv'
matched_points_jump_file = './runs/matched_points_jump.csv'
jump_output_file = './runs/cleaned_new_jump.csv'

# 合并训练集
merge_traj_and_match(traj_file, match_file, traj_output_file)

# 合并测试集
add_match_road_id(jump_task_file, matched_points_jump_file, jump_output_file)


# 4. 加载训练集和测试集数据
traj_task_data = pd.read_csv(traj_output_file)
jump_task_data = pd.read_csv(jump_output_file)

# 1. 加载数据
#jump_task_path = 'cleaned_new_jump.csv'  # 替换为清理后文件的实际路径
#jump_task_data = pd.read_csv(jump_task_path)
#traj_task_path = 'cleaned_new_traj.csv'
#traj_task_data = pd.read_csv(traj_task_path)

# 2. 重新推导终点信息
def add_terminal_points(data):
    """
    推导终点信息，并添加到数据中。
    假设每条轨迹的最后一个点为终点，将其 matched_road_id 作为分类目标。
    """
    grouped = data.groupby('trajectory_id')  # 按 trajectory_id 分组
    terminal_rows = []

    for trajectory_id, group in grouped:
        if group.shape[0] < 2:
            continue  # 如果轨迹点不足 2 个，跳过
        terminal_row = group.iloc[-1].copy()  # 最后一行作为终点
        terminal_row['matched_road_id'] = terminal_row['matched_road_id']  # 保留原始终点信息
        terminal_rows.append(terminal_row)

    # 合并终点行到数据中
    terminal_df = pd.DataFrame(terminal_rows)
    return terminal_df

# 添加终点信息
terminal_data = add_terminal_points(traj_task_data)

# 3. 构建特征和标签
def build_features_and_labels(data, terminal_data):
    """
    根据每条轨迹的所有非终点点的 road_id 构建特征和分类目标
    """
    grouped = data.groupby('trajectory_id')
    features = []
    labels = []

    for trajectory_id, group in grouped:
        road_ids = group['matched_road_id'].tolist()
        if len(road_ids) < 2:  # 如果轨迹点不足 2 个，跳过
            continue

        # 选择所有非终点点的 road_id 作为特征
        non_terminal_points = road_ids[:-1]  # 所有非终点点
        terminal_row = terminal_data[terminal_data['trajectory_id'] == trajectory_id]
        if terminal_row.empty:
            continue  # 如果终点信息缺失，跳过
        terminal_road_id = terminal_row['matched_road_id'].values[0]  # 获取终点 road_id
        features.append(non_terminal_points)
        labels.append(terminal_road_id)

    print(f"特征数量: {len(features)}, 标签数量: {len(labels)}")
    return features, labels

# 提取训练数据的特征和分类目标
features, labels = build_features_and_labels(traj_task_data, terminal_data)

# 4. 特征填充或截断为固定长度
def encode_features(features, max_length):
    """
    将路段ID序列填充或截断为固定长度
    """
    encoded_features = []
    for feature in features:
        if len(feature) < max_length:
            encoded = feature + [0] * (max_length - len(feature))  # 填充0到固定长度
        else:
            encoded = feature[:max_length]  # 截断到固定长度
        encoded_features.append(encoded)
    return np.array(encoded_features)

# 设置最大序列长度（轨迹的最长非终点点数量）
max_length = max(len(f) for f in features)

# 对训练数据的特征进行填充或截断
X = encode_features(features, max_length)

# 将分类目标（终点的 road_id）转换为整数编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# 5. 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 训练分类模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 验证模型性能
y_pred = model.predict(X_val)  # 对验证集进行预测
accuracy = accuracy_score(y_val, y_pred)  # 计算准确率
print(f"分类模型的准确率: {accuracy:.2f}")

# 7. 对测试集进行预测
def predict_jump_task(data, terminal_data, model, label_encoder, max_length):
    """
    对 jump_task.csv 进行预测，输出每条轨迹的终点 road_id
    """
    results = []
    grouped = data.groupby('trajectory_id')

    for trajectory_id, group in grouped:
        road_ids = group['matched_road_id'].tolist()
        if len(road_ids) < 2:  # 如果轨迹点不足 2 个，跳过
            continue

        # 构建特征矩阵（所有非终点点）
        non_terminal_points = road_ids[:-1]
        test_feature = encode_features([non_terminal_points], max_length)

        # 使用模型预测终点的分类标签
        predicted_class = model.predict(test_feature)
        predicted_road_id = label_encoder.inverse_transform(predicted_class)[0]

        # 保存预测结果
        results.append({'trajectory_id': trajectory_id, 'predicted_road_id': predicted_road_id})

    return pd.DataFrame(results)

# 对测试集进行预测
test_results = predict_jump_task(jump_task_data, terminal_data, model, label_encoder, max_length)

# 8. 保存预测结果到 CSV 文件
test_results.to_csv('jump_task_predictions.csv', index=False)
print("预测完成，结果已保存至 jump_task_predictions.csv")
