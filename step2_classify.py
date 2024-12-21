import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer

# 加载数据
road_data = pd.read_csv('/Users/shanhao/Desktop/grate3-1/数据挖掘/大作业/DM_2024_Dataset/road.csv')

# 数据预处理
# 将类别型特征进行One-Hot编码，数值型特征进行标准化
categorical_features = ['tunnel', 'bridge', 'roundabout', 'oneway']
numerical_features = ['lanes', 'length', 'maxspeed', 's_lon', 's_lat', 'e_lon', 'e_lat', 'm_lon', 'm_lat']

# 创建预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 定义模型
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 准备数据
X = road_data.drop(['highway', 'geometry'], axis=1)  # 删除非特征列和非数值列
y = road_data['highway']  # 目标变量

# 去除目标变量中包含NaN的行
X = X[~y.isnull()]
y = y[~y.isnull()]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 评估模型
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 预测缺失的highway类别
missing_indices = road_data['highway'].isnull()
predicted_highway = model.predict(road_data[missing_indices].drop(['highway', 'geometry'], axis=1))
road_data.loc[missing_indices, 'highway'] = predicted_highway

# 保存结果
road_data.to_csv('road_filled.csv', index=False)