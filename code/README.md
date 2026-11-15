# 工作帽佩戴状态检测（任务二）— 代码与实验说明

本目录实现「佩戴工作帽 / 未佩戴工作帽」两状态目标检测的完整流程：数据清洗与独立划分 → 训练 → 评估 → 图片/视频推理 → 实验报告。

## 目录

```
code/
  scripts/
    prepare_dataset.py     # 数据清洗、副本合并、独立划分、生成 data.yaml / qc.json
    train.py               # YOLO11 训练封装（保存 best.pt 与 config.json）
    evaluate.py            # val/test 指标、混淆矩阵、PR/F1 曲线、置信度阈值扫描
    predict.py             # 图片/目录/视频推理，输出类别+置信度并保存结果
    make_demo_video.py     # 用 test 图片合成演示视频
    make_report_tables.py  # 生成两组实验的对比表
  runs/                    # 训练与评估输出（含旧版训练记录，仅作历史）
  weights/                 # 最终 best.pt
  outputs/                 # 推理结果图、结果视频、检测记录 CSV
  report/                  # 指标表、对比表、图表与实验报告
dataset/helmet/            # 重建后的独立数据集（ASCII 路径）
```

## 环境

- Python 3.9.23，torch 2.8.0+cu129，ultralytics 8.4.174，GPU：RTX 4070 Laptop 8GB
- 解释器：`d:\miniconda3\envs\pytorch\python.exe`

## 一键流程

```powershell
$py = 'd:\miniconda3\envs\pytorch\python.exe'
$env:PYTHONIOENCODING = 'utf-8'

# 1. 数据清洗与划分（可重复执行，会重建 dataset\helmet）
& $py code\scripts\prepare_dataset.py

# 2. 基线训练（默认增强）
& $py code\scripts\train.py --name baseline_yolo11n --epochs 100 --imgsz 640 --batch 16

# 3. 对比实验（强增强）
& $py code\scripts\train.py --name strongaug_yolo11n --epochs 100 --imgsz 640 --batch 16 --aug strong

# 4. 评估（val + 独立 test）
& $py code\scripts\evaluate.py `
    --run baseline=d:/hat/code/weights/baseline_yolo11n_best.pt `
    --run strongaug=d:/hat/code/weights/strongaug_yolo11n_best.pt

# 5. 对比表
& $py code\scripts\make_report_tables.py --runs baseline_yolo11n strongaug_yolo11n

# 6. 图片推理
& $py code\scripts\predict.py `
    --weights d:/hat/code/weights/baseline_yolo11n_best.pt `
    --source d:/hat/dataset/helmet/images/test --conf 0.5

# 7. 演示视频：先用 test 图片合成，再做推理
& $py code\scripts\make_demo_video.py
& $py code\scripts\predict.py `
    --weights d:/hat/code/weights/baseline_yolo11n_best.pt `
    --source d:/hat/code/outputs/demo/test_clip.mp4 --conf 0.5
```

## 数据说明

- 原始数据：`d:\hat\data\考核任务数据\安全帽（无test）数据`（只读，含 Roboflow 生成的离线增强副本，且无 test 划分）。
- 类别：`0 with_work_cap`（佩戴工作帽）、`1 without_work_cap`（未佩戴工作帽）；检测框口径统一为「头部」。
- 清洗：同一源帧的离线增强副本按「同类 + 框坐标容差 0.02 + 灰度 NCC ≥ 0.90」合并为 1 张，2384 张 → 1848 张唯一帧。
- 划分：同一「视频族 + 帧序号」的帧只会进入同一个子集，按时间顺序 70/15/15 切块；随后清除跨子集的近重复帧（NCC ≥ 0.99 且框差 ≤ 0.02），最终 train 1323 / val 191 / test 209。
- test 集仅在最终评估与推理演示中使用，不参与训练和调参。

详见 `dataset/helmet/qc.json` 与 `dataset/helmet/split_manifest.csv`。

## 产物

| 产物 | 位置 |
|---|---|
| 数据集配置 | `d:\hat\dataset\helmet\data.yaml` |
| 训练权重 | `code\weights\<run>_best.pt` |
| 训练配置 | `code\runs\<run>\config.json` |
| 指标表 | `code\report\metrics.csv`、`metrics_table.md` |
| 阈值扫描 | `code\report\threshold_sweep.csv` |
| 对比表 | `code\report\compare_table.md` |
| 混淆矩阵/PR/F1 | `code\report\assets\<run>\<split>\` |
| 推理结果 | `code\outputs\<source>\images|video`、`detections.csv`、`summary.csv` |
| 实验报告 | `code\report\实验报告.md` |
