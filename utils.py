import os, json, random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List

def set_seed(seed: int = 42):
    """# 设定所有可控随机源的种子，保证实验可复现
    - Python 原生 random
    - NumPy
    - PyTorch（CPU/GPU）
    - cuDNN 的确定性与禁用动态优化（benchmark=False）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # 确定性：相同输入/顺序→相同输出（可能略降速）
    torch.backends.cudnn.benchmark = False      # 关闭动态算法搜索，避免不同批次导致的不一致

def ensure_dir(path: str):
    """# 若目录不存在则递归创建；存在则忽略（不报错）
    - path 可为相对/绝对路径
    - 与 os.makedirs(..., exist_ok=True) 等价封装
    """
    os.makedirs(path, exist_ok=True)

def save_jsonl(path: str, record: Dict):
    """# 以 JSON Lines 追加写入一条记录
    - 每条记录占一行，便于流式追加与后续解析
    - ensure_ascii=False 保留中文
    """
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def plot_curves(history: List[Dict], save_path: str):
    """# 根据 history 绘制损失/准确率曲线并各自保存为 PNG
    - history: [{'epoch': int, 'train_loss': float, 'val_loss': float, 'train_acc': float, 'val_acc': float}, ...]
    - save_path: 仅用于生成文件名前缀；会导出 *_loss.png 与 *_acc.png 两张图
    """
    epochs = [h['epoch'] for h in history]              # x 轴：epoch 序列
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]

    # --- 损失曲线 ---
    plt.figure()
    plt.plot(epochs, train_loss, label='train_loss')    # 训练损失
    plt.plot(epochs, val_loss, label='val_loss')        # 验证损失
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.title('Loss Curves')
    plt.savefig(save_path.replace('.png', '_loss.png'), bbox_inches='tight', dpi=150)  # 导出 *_loss.png
    plt.close()

    # --- 准确率曲线 ---
    plt.figure()
    plt.plot(epochs, train_acc, label='train_acc')      # 训练准确率
    plt.plot(epochs, val_acc, label='val_acc')          # 验证准确率
    plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend(); plt.title('Accuracy Curves')
    plt.savefig(save_path.replace('.png', '_acc.png'), bbox_inches='tight', dpi=150)   # 导出 *_acc.png
    plt.close()

def plot_confusion_matrix(cm: np.ndarray, classes, save_path: str):
    """# 绘制并保存混淆矩阵图片
    - cm: 形状为 [num_classes, num_classes] 的整数矩阵（行：真实标签；列：预测标签）
    - classes: 类别名列表（长度需与 cm 维度一致），用于坐标刻度标签
    - save_path: 输出 PNG 路径
    备注：
    - 此实现直接显示计数（未归一化）；如需百分比可外部先对 cm 做归一化后再传入
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest')   # 直接绘制热力图（默认颜色映射）
    plt.title('Confusion Matrix')
    plt.colorbar()                            # 颜色条
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)          # x 轴刻度：预测标签
    plt.yticks(tick_marks, classes)          # y 轴刻度：真实标签
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # 在格子中心叠加具体计数，便于阅读
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)  # 导出 PNG
    plt.close()
