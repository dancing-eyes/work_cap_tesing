考核内容：工作帽佩戴状态检测

一、任务描述
使用Roboflow helmet v3数据集（无数张工人有无带帽的图片），对一个模型做监督训练，让这个模型具备识别工人是否有带工帽的能力。
                     
二、模型的选择
  我们选择了yolo11n.pt，即 YOLO11 的 nano 版本，使用 COCO 预训练权重作为初始化。
三、任务设计（包括设计思路，训练/测试过程、实现代码及实验结果）
1.观察数据决定数据清洗的方式
安全帽数据（无test）
总共2384张数据，图像为640*360，类别两种hat和head。
数据里有2384 张照片，但并不是所有的照片都相同的。里面有很多同一张图片的增强副本，也就是一张图片经过一些修改形成新的图片目的是增强模型的鲁棒性。

所以，清洗数据尤其必要！
2.数据清晰的步骤
防止同一帧的增广副本同时出现在train/valid/test里，导致数据泄露。让模型有机会背答案趁机拿虚高的分数。
第一步，将数据存储为嵌套字典

recoreds为存储数据的嵌套字典，记录了每一张图片的文件名，帧来源，来自train还是valid，文件路径，对应的是否戴帽子的标签以及框的坐标。

第二步，检查标签是否有差错

检查是否有除了0，1之外的标签，也就是那张图片不存在人和帽子这些东西。

第三步，将同一帧的增广副本聚在一起，以免后续同一帧的不同副本泄露至不同集合中，导致训练泄露。
对于stem名字相同的图片通过脚本判断是不是同一张图片的增广副本
判断条件：1. 框偏差在0.2以内。2. 把图片变成灰度缩略图，看两图能否达到90%的相似度。

关键在于这个same_source

第四步，按帧的时间顺序划分数据集，70%train，15%valid，15%test。


第五步，导出数据文件。
dataset/helmet/
├── images/
│   ├── train/   val/   test/
└── labels/
    ├── train/   val/   test/

dataset/helmet/
├── images/train/c00000_gate_cam1_00123.jpg
├── images/train/c00000_gate_cam1_00123.rf.hflip.jpg   ← 实际带原后缀
├── labels/train/c00000_gate_cam1_00123.txt
├── data.yaml
├── split_manifest.csv
└── qc.json
3.模型的训练 
第一步，参数的定义

epochs为100
batch size=16
image size=640
数据增强设置

参数	基线（默认增强）	强增强（--aug strong）
hsv_h	0.015	0.03
hsv_s	0.7	0.9
hsv_v	0.4	0.5
degrees（旋转）	0.0	10.0
translate	0.1	0.2
scale	0.5	0.7
shear（剪切）	0.0	2.0
perspective	0.0	0.0005
mixup	0.0	0.15
erasing（随机擦除）	0.4	0.5
fliplr / mosaic	0.5 / 1.0	0.5 / 1.0

此部分用于增强对比实验
强增强就是保持语义不变的随机变换，对训练集进行光照、色温、曝光变化，左右对称等变化，让模型见多识广，增强模型的泛化能力。










学习率相关配置
项	值
optimizer	auto（由 ultralytics 自动选择优化器并按 batch/nbs 自动推算 lr）
lr0（实际生效）	0.001667
lrf（终止学习率系数）	0.01，即末端 lr ≈ 1.67e-5，线性衰减
momentum	0.937
weight_decay	0.0005
warmup_epochs / momentum / bias_lr	3.0 / 0.8 / 0.1
cos_lr	false（线性衰减，非余弦）
amp	true（混合精度）
seed / deterministic	0 / true

第二步，开始训练
选择是否强增强，然后开始训练


四、分类结果分析 
混淆矩阵
baseline在test上的混淆矩阵

增强型在test上的混淆矩阵

baseline在val上的混淆矩阵

增强型在val上的混淆矩阵

很明显，baseline实际表现效果更好，特别是without_work_cap这一列。
对于最后的实验结果
模型	增强	split	P	R	mAP@0.5	mAP@0.5:0.95
YOLO11n	默认	val	0.9988	0.9931	0.9949	0.9266
YOLO11n	强	val	0.9902	1.0000	0.9949	0.8945
YOLO11n	默认	test	0.9346	0.9788	0.9729	0.8034
YOLO11n	强	test	0.9542	0.9694	0.9713	0.7889

1.val比test实验结果要好
2.Recall里baseline要强于增强型，说明baseline正确判断的更多，mAP@0.5和mAP@0.5:0.95都比增强型要高，说明框框画得更准。
