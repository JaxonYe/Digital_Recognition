# 2025年人工智能导论课的课设 数字识别 · 模型复现实验

> **项目概述**  
> 复现并对比 LeNet-5、AlexNet、VGG-16、GoogLeNet（Inception v1）、ResNet-18、DenseNet-121 在 **MNIST** 与 **SVHN** 上的域内与跨域表现；统一训练/评测管线；并提供 **PyQt5 推理界面**（支持画板/图片推理、中间层特征图与卷积核权重可视化）。

---

## 目录
- [环境与依赖](#环境与依赖)
- [数据集与预处理](#数据集与预处理)
- [项目结构](#项目结构)
- [使用方法](#使用方法)
  - [训练](#训练)
  - [评测（含跨域评测）](#评测含跨域评测)
  - [推理 GUI（PyQt5）](#推理-gui-pyqt5)
- [结果汇总](#结果汇总)
  - [域内测试](#域内测试)
  - [跨域测试](#跨域测试)
- [结果分析（精要）](#结果分析精要)
- [可复现实验设置](#可复现实验设置)
- [参考](#参考)

---

## 环境与依赖

- OS：Windows / Linux 均可（CUDA 可用时默认用 GPU）  
- Python：3.10  
- 深度学习框架：PyTorch（与 `torchvision`）  
- IDE（可选）：PyCharm 2025  
- GPU：RTX 3060 12GB（或任意可用显卡）

安装依赖（示例）：
```bash
pip install torch torchvision matplotlib numpy tqdm pyqt5
```

---

## 数据集与预处理

- **MNIST**：灰度 1×28×28。  
  - LeNet：保留 1×28×28，`ToTensor + Normalize(MNIST_MEAN, MNIST_STD)`  
  - 其他网络：`Resize(224) + Grayscale(3)`，再 `ToTensor + Normalize`（3 通道统计）
- **SVHN**（cropped digits 单标签版）：RGB 3×32×32。  
  - 标签 10 → 数字 0（读入时映射）。  
  - LeNet：`Resize(28) + Grayscale(1)`，`Normalize(0.5, 0.5)`  
  - 其他网络：保留 RGB，`Resize(224)`，`Normalize(0.5,0.5,0.5)`

数据由 `torchvision.datasets` 自动下载到 `data/`（SVHN 下载到 `data/SVHN/`）。

---

## 项目结构

```
.
├── data_transforms.py   # 预处理/增强（随模型与数据集自适应）
├── model.py             # LeNet / AlexNet / VGG16 / GoogLeNet / ResNet / DenseNet（从零实现）
├── models_factory.py    # 工厂方法：按字符串创建模型
├── train.py             # 训练入口（训练/验证/测试曲线、best.pt、日志）
├── evaluate.py          # 评测入口（主域 & 跨域；混淆矩阵/样例网格/metrics）
├── predict.py           # PyQt5 推理界面：画板/图片 + 特征图/卷积核可视化
├── parameters.py        # 统一参数（训练/评测/路径）
├── utils.py             # 常用工具（绘图、日志、固定随机种子等）
└── outputs/             # 训练与评测结果（自动创建并按数据集/模型分层）
```

---

## 使用方法

### 训练
本项目**不依赖命令行超参**；在 `parameters.py` 内配置后直接运行：
```bash
python train.py
```
产物保存在：`outputs/<dataset>/<arch>/train/`（包含 `best.pt`、曲线图、日志与 `args_train.json`）。

### 评测（含跨域评测）
同样通过 `parameters.py` 控制（如是否开启 `do_cross_eval`）：
```bash
python evaluate.py
```
主域输出在：`outputs/<dataset>/<arch>/evaluate/`。  
跨域输出在：`outputs/<dataset>/<arch>/evaluate/cross_to_<other>/`。

### 推理 GUI（PyQt5）
```bash
python predict.py
```
- 界面支持：选择模型权重、黑底画板/图片导入、一键推理、概率分布、**中间层特征图**与**卷积核权重**可视化。  
- 与离线训练同口径预处理与权重加载，保证数值一致性。

---

## 结果汇总

### 域内测试

> 单位：分类精度 **%**。MNIST 与 SVHN 的**训练/验证/测试**三列分别对应 `train/val/test` 的准确率。

| 模型 | MNIST 训练 | MNIST 验证 | MNIST 测试 | SVHN 训练 | SVHN 验证 | SVHN 测试 |
|:--|--:|--:|--:|--:|--:|--:|
| LeNet | 100.0 | 98.98 | 99.00 | 93.73 | 88.17 | 86.45 |
| AlexNet | 99.80 | 99.36 | 99.33 | 98.77 | 93.37 | 93.12 |
| VGG-16 | 11.25 | 11.10 | 11.35 | 18.90 | 19.03 | 19.58 |
| VGG-16(+BN) | 99.92 | 99.45 | 99.44 | 18.90 | 19.03 | 19.58 |
| VGG-16(+BN) (Adam→SGD) | 99.78 | 99.53 | 99.36 | 99.64 | 94.56 | 95.14 |
| GoogLeNet | 99.80 | 99.23 | 99.23 | 18.90 | 19.03 | 19.58 |
| GoogLeNet(+BN) | 99.97 | 99.63 | 99.48 | 99.95 | 95.15 | 95.24 |
| ResNet-18 | 99.97 | 99.63 | 99.39 | 99.94 | 95.05 | 95.17 |
| DenseNet-121 | 99.91 | 99.61 | 99.50 | 99.74 | 94.38 | 94.49 |

### 跨域测试

> 单位：分类精度 **%**。左：在 MNIST 训练 → SVHN 测试；右：在 SVHN 训练 → MNIST 测试。  
> 注：表中 `GoogLeNet(+BN) MNIST→SVHN = 21.97%`（原始表格“2197”为格式化小数点遗漏）。

| 模型 | MNIST→SVHN 测试 | SVHN→MNIST 测试 |
|:--|--:|--:|
| LeNet | 12.74 | 52.66 |
| AlexNet | 25.48 | 65.20 |
| VGG-16 | 19.58 | 11.35 |
| VGG-16(+BN) | 26.26 | 11.35 |
| VGG-16(+BN) (Adam→SGD) | 14.67 | **81.77** |
| GoogLeNet | 24.27 | 11.35 |
| GoogLeNet(+BN) | 21.97 | 66.43 |
| ResNet-18 | 14.44 | 62.44 |
| DenseNet-121 | 9.60 | 12.42 |

---

## 结果分析（精要）

- **MNIST（简单分布）**：除“未加 BN 的 VGG-16 / GoogLeNet”外，几乎全部模型均达到 **99% 左右**。浅层 LeNet 已足以胜任；更深网络在该任务上受益有限。  
- **SVHN（复杂分布）**：需要更强的表征与稳定的优化。**带 BN 的 GoogLeNet / ResNet-18** 达到 **95%+**；而**未加 BN** 的 VGG-16/GoogLeNet 仅 ~19%，表现为“训练正常但泛化/稳定性差”的**优化失稳**现象。对 VGG-16 **改用 SGD** 后，测试集达 **95.14%**，显示出 **优化器与正则** 对性能的关键作用。  
- **跨域（MNIST→SVHN vs. SVHN→MNIST）**：从**简单→复杂**（MNIST→SVHN）普遍**很差**（< 30%），从**复杂→简单**（SVHN→MNIST）明显更好（50%~80%）。这是典型的**分布偏移**与**表示不对称**：在简单数据上学到的“粗粒度形状特征”很难覆盖 SVHN 的背景、颜色与尺度变化；而在复杂数据上学到的多尺度/颜色不变性迁移到 MNIST 更容易。  
- **小结**：BN、优化器选择、输入预处理一致性与足够的容量/深度是 SVHN 上收敛与泛化的关键；跨域结果表明若要进一步提升，需要显式引入 **域对齐/风格迁移/自训练** 等机制。

---

## 可复现实验设置

- 随机种子：**1**（Python / NumPy / PyTorch / CUDA 全部固定），`cudnn.deterministic=True, cudnn.benchmark=False`。  
- 训练与评测目录：按 `outputs/<dataset>/<arch>/{train|evaluate}` 分层组织；评测阶段会自动保存 `confusion_matrix.png / metrics_eval.json / samples_correct.png / samples_miscls.png / args_evaluate.json`。  
- 其余细节（批大小、学习率、优化器、是否做跨域等）在 `parameters.py` 中集中管理。

---

## 参考

1) LeCun Y., Bottou L., Bengio Y., et al. *Gradient-based Learning Applied to Document Recognition*, 1998.  
2) Krizhevsky A., Sutskever I., Hinton G.E. *ImageNet Classification with Deep Convolutional Neural Networks*, 2012.  
3) Simonyan K., Zisserman A. *Very Deep Convolutional Networks for Large-Scale Image Recognition*, 2014.  
4) Szegedy C., Liu W., Jia Y., et al. *Going Deeper with Convolutions*, 2015.  
5) He K., Zhang X., Ren S., Sun J. *Deep Residual Learning for Image Recognition*, 2016.  
6) Huang G., Liu Z., Van Der Maaten L., Weinberger K.Q. *Densely Connected Convolutional Networks*, 2017.
