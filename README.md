# 项目结构

整体项目结构如下：

```bash
tree

.
├── README.md                                       # 说明文档
├── requertment.txt                                 # python环境依赖
├── 1_itlubber_submit.py                            # 初始方案
├── 2_itlubber_submit_pseudo.py                     # 伪标签方案
├── 3_inference.py                                  # 模型推理脚本
├── data                                            # 数据文件夹
│   ├── Test_A.zip                                  # 比赛提供的测试集数据
│   ├── Train.zip                                   # 比赛提供的训练集数据
│   └── 数据说明.txt
├── result.csv                                      # 初始方案生成的提交文件 0.8620
├── pseudo_result.csv                               # 最终提交的伪标签方案结果 0.8636
├── inference_result.csv                            # 使用伪标签方案保存的模型推理的到的结果 0.8636
├── vloong_anomaly_detection_model.torch            # 初始方案保存的模型
└── vloong_anomaly_detection_model_pseudo.torch     # 伪标签方案保存的模型

1 directory, 13 files
```

# 运行步骤

0. 环境配置

python版本为 3.8，pytorch 版本为 1.8.1，torch-geometric 版本为 2.1.0（最新的也行）

```bash
python

Python 3.8.5 (default, Sep  4 2020, 07:30:14)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

1. 解压数据到 `data` 目录下方

```bash
unzip -d data/ data/Train.zip
unzip -d data/ data/Test_A.zip
```

解压后的文件结构如下:

```bash
tree -L 2

.
├── README.md
├── requertment.txt
├── 1_itlubber_submit.py
├── 2_itlubber_submit_pseudo.py
├── 3_inference.py
├── data
│   ├── Test_A                                      # 测试集解压的文件夹
│   ├── Test_A.zip                                  # 比赛提供的测试集数据
│   ├── Train                                       # 训练集解压的文件夹
│   ├── Train.zip                                   # 比赛提供的训练集数据
│   └── 数据说明.txt
├── result.csv
├── pseudo_result.csv
├── inference_result.csv
├── vloong_anomaly_detection_model.torch
└── vloong_anomaly_detection_model_pseudo.torch

3 directory, 13 files
```

2. 运行 `1_itlubber_submit.py` ，得到初始方案的预测结果 `result.csv` 和模型文件 `vloong_anomaly_detection_model.torch`

```bash
python 1_itlubber_submit.py
```

3. 运行 `2_itlubber_submit_pseudo.py` ，得到最终方案的预测结果 `pseudo_result.csv` 和模型文件 `vloong_anomaly_detection_model_pseudo.torch`

```bash
python 2_itlubber_submit_pseudo.py
```

4. 运行 `3_itlubber_inference.py` ，加载步骤 `2` 保存的模型文件 `vloong_anomaly_detection_model_pseudo.torch` 进行推理得到 `inference_result.csv` 结果文件

```bash
python 3_itlubber_inference.py --test_dir data/Test_A/ --model vloong_anomaly_detection_model_pseudo.torch --batch_size 64
```

5. 提交 `inference_result.csv` 为最终预测结果
