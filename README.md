

# 1 GetStart

## 1.1 环境配置

```
pip install -r requirements.txt
```

## 1.2 代码运行

！！！！TODO！！！！

需要整理运行代码，构建一个main.py利用参数调控执行到第几个步骤

# 2 代码笔记

## 2.1 文件说明

- DM_2024_Dataset：课程组提供数据（具体说明见文件夹内的说明文档）
- runs：为保证整洁性，将运行全过程生成的中间代码存储在该文件夹中
  - ori_xx.csv：坐标转换前的xx.csv数据
  - gcj_xx.csv：坐标转换后的xx.csv数据
  - matched_points_xx.csv：完成STEP1后，对xx.csv和road.csv进行匹配后的结果
  - road_filled.csv：完成STEP2后，补全路段分类的数据
  - eta_task_filled.csv：完成STEP3后，不全到达时间的数据
- step0_transfer.py：坐标转换
- step1_mapMatching.py：任务一，路网匹配代码
- step2_roadClassify.py：任务二，路段分类代码
- step3_etaEst.py：任务三，ETA估计代码

## 2.2 代码说明

### STEP0 坐标转换

​		利用课程组给出的transfer.py进行修改后实现

​		主要修改为：为代码添加了一个参数，该参数可以为"traj"和"eta_task"之一，为"traj"时转换`traj.csv`中的数据，反之转换`eta_task.csv`中的数据

### STEP1 路网匹配

### STEP2 路段分类

分类器选择：随机森林RandomForestClassifier

### STEP3 ETA估计

#### 基本思路

​		任务目的为对每一个点估计到达时间，但对速度的预测才是本任务的核心预测点，到达时间根据速度计算。

​		因此，实现本任务的核心思路为，选用随机森林模型作为核心模型，以`['highway','lanes','tunnel','bridge','roundabout','oneway']`作为特征，以轨迹点的速度为输出。训练上述模型，以实现对速度的估计，进而达到能够预测到达时间的功能





## TODO

- [ ] 路网匹配
  - [x] 基本功能实现
  - [ ] 路网匹配的去噪处理
- [ ] 路段分类
  - [x] 基本功能实现
  - [ ] 路段分类更新数据
  - [ ] 路段分类分类器调研优化
- [ ] ETA估计
  - [x] 基本功能实现
  - [x] ETA估计预测部分
  - [ ] ETA估计模型优化
- [ ] 下一跳预测
  - [ ] 基本功能实现