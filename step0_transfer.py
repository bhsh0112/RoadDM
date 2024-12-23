import ast
import pandas as pd
import transbigdata as tbd
import argparse
import os

# 设置命令行参数解析
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('mode', choices=['traj', 'eta_task'], help='Mode of operation')
args = parser.parse_args()

# 根据模式选择文件名
file_suffix = 'eta_task' if args.mode == 'eta_task' else 'traj'

traj_df = pd.read_csv(f'./DM_2024_Dataset/{file_suffix}.csv')
traj_df['lon'] = traj_df['coordinates'].apply(lambda coord: ast.literal_eval(coord)[0])
traj_df['lat'] = traj_df['coordinates'].apply(lambda coord: ast.literal_eval(coord)[1])
traj_df.to_csv(f'./runs/ori_{file_suffix}.csv', index=False)

gcj_df = traj_df.apply(lambda row: tbd.gcj02towgs84(row['lon'], row['lat']), axis=1, result_type='expand')
gcj_df.columns = ['lon', 'lat']
gcj_df['id'] = range(len(gcj_df))
gcj_df.to_csv(f'./runs/gcj_{file_suffix}.csv', index=False)