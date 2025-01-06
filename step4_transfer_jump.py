import ast
import pandas as pd
import transbigdata as tbd

# 定义一个函数来安全地解析坐标，并返回经度和纬度
def parse_coordinates(coord):
    try:
        if pd.notna(coord):  # 检查是否不是NaN
            parsed_coord = ast.literal_eval(coord)
            if isinstance(parsed_coord, (list, tuple)) and len(parsed_coord) == 2:
                return parsed_coord
    except (ValueError, SyntaxError):  # 解析失败的情况
        pass
    return [None, None]  # 如果解析失败或者数据不符合预期格式，则返回None

# 读取原始CSV文件到DataFrame
traj_df = pd.read_csv('./DM_2024_Dataset/jump_task.csv')

# 应用函数并创建新的列
traj_df[['lon', 'lat']] = traj_df['coordinates'].apply(
    lambda coord: pd.Series(parse_coordinates(coord))
)

# 过滤掉无效的（即包含None的）经纬度行
traj_df = traj_df.dropna(subset=['lon', 'lat'])

# 保存原始数据到CSV（不含无效的坐标）
traj_df.to_csv('ori_jump.csv', index=False)

# 定义转换函数，并假设输入坐标为GCJ-02
def convert_coordinates(row):
    lon, lat = row['lon'], row['lat']
    return pd.Series(tbd.gcj02towgs84(lon, lat))

# 应用转换函数到每一行数据
gcj_df = traj_df.apply(convert_coordinates, axis=1)
gcj_df.columns = ['lon', 'lat']

# 添加id列
gcj_df['id'] = range(len(gcj_df))

# 保存转换后的数据到CSV（不含无效的坐标）
gcj_df.to_csv('./runs/gcj_jump.csv', index=False)