# -*- coding: utf-8 -*-
"""
evaluate.py
- 读取 parameters.py 的统一参数（单入口 get_args）
- 主域评测：保存到 outputs/<dataset>/<arch>/evaluate/
  * confusion_matrix.png
  * metrics_eval.json （加入 train/val/test 三套指标）
  * samples_correct.png （4x4 成功样例，仅测试集）
  * samples_miscls.png  （4x4 失败样例，仅测试集）
  * args_evaluate.json （本次评测用到的参数）
- 跨域评测：在 outputs/<dataset>/<arch>/evaluate/<cross_to_xxx>/ 下仍只保存测试集指标与可视化
"""
import os                         # 路径处理
import random                     # Python 随机数
import json                       # 保存指标/参数到 JSON
import argparse  # 保留不影响      # 这里保留 argparse，不影响实际逻辑
import numpy as np                # 数值运算
import matplotlib.pyplot as plt   # 可视化样例网格
import torch                      # PyTorch 主库
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets   # 官方数据集封装（MNIST/SVHN）

from utils import plot_confusion_matrix, ensure_dir   # 自己的工具函数：画混淆矩阵；确保目录存在
from models_factory import create_model               # 工厂方法：根据字符串创建模型实例
from data_transforms import get_transforms            # 统一的数据预处理（和模型/数据集适配）
from parameters import get_args                       # 统一入口，读取所有参数（不走命令行）

# 统一读取参数（不走命令行，默认值见 parameters.py）
args = get_args()  # 注意：parameters.py 已自动推断 eval_dir/cross_dir/weights/save_* 等路径


"""==================================================================================================="""
# 为保证结果可复现：设置 Python/NumPy/PyTorch 的随机种子；并固定 cuDNN 的策略
print('Using random seed : ', args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)  # set random seed for all gpus
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)            # 设置 CUDA 随机数种子（当前 GPU）
    torch.cuda.manual_seed_all(args.seed)        # 设置所有 GPU 的随机数种子（多卡时）
    torch.backends.cudnn.deterministic = True    # 确保每次运行的计算图选择一致（牺牲少量速度）
    torch.backends.cudnn.benchmark = False       # 关闭动态寻优，进一步保证复现
"""==================================================================================================="""

# ---- Windows 多进程安全：顶层函数可被 pickle ----
def svhn_label_to_digit(y):
    """SVHN 标签 10 代表数字 0，这里做一次映射。"""
    y = int(y)
    return 0 if y == 10 else y


@torch.no_grad()
def evaluate(model, loader, device):
    """
    返回：
    - avg_loss, avg_acc, cm
    - (ok_imgs, ok_lbls, ok_preds)
    - (no_imgs, no_lbls, no_preds)

    说明：
    * 遍历 DataLoader，前向得到 logits，计算交叉熵与准确率；
    * 同时收集最多 16 张“分类正确/错误”的样例，后面用于可视化；
    * 计算 10×10 的混淆矩阵（行是真实标签，列是预测标签）。
    """
    model.eval()                                # 评估模式（关闭 Dropout/BN 的学习）
    total_loss, total_acc, n = 0.0, 0.0, 0
    criterion = nn.CrossEntropyLoss()           # 多分类的标准损失
    all_preds, all_labels = [], []              # 汇总全量预测与标签

    # 额外收集：前 16 个正确/错误样例（用于 4x4 网格可视化）
    ok_imgs, ok_lbls, ok_preds = [], [], []
    no_imgs, no_lbls, no_preds = [], [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)       # 张量搬到 GPU/CPU
        logits = model(x)                       # 前向计算，得到 [B, num_classes]
        loss = criterion(logits, y)             # 单批次平均损失

        bs = x.size(0)
        total_loss += loss.item() * bs          # 按样本数累加，最后再除以总样本
        total_acc += (torch.argmax(logits, dim=1) == y).float().sum().item()
        n += bs

        pred = torch.argmax(logits, dim=1)      # 离散预测标签 [B]
        all_preds.append(pred.cpu().numpy())    # 放到 CPU 以便 numpy 处理
        all_labels.append(y.cpu().numpy())

        # 收集样例（各取最多 16 张，优先从前面的 batch 中挑）
        if len(ok_imgs) < 16 or len(no_imgs) < 16:
            for i in range(bs):
                if pred[i].item() == y[i].item():
                    if len(ok_imgs) < 16:
                        ok_imgs.append(x[i].detach().cpu())
                        ok_lbls.append(int(y[i].item()))
                        ok_preds.append(int(pred[i].item()))
                else:
                    if len(no_imgs) < 16:
                        no_imgs.append(x[i].detach().cpu())
                        no_lbls.append(int(y[i].item()))
                        no_preds.append(int(pred[i].item()))
                if len(ok_imgs) >= 16 and len(no_imgs) >= 16:
                    break

    avg_loss = total_loss / n                   # 全集平均损失
    avg_acc = total_acc / n                     # 全集平均准确率

    preds = np.concatenate(all_preds)           # [N]
    labels = np.concatenate(all_labels)         # [N]
    cm = np.zeros((10, 10), dtype=int)          # 10 类的混淆矩阵
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    return avg_loss, avg_acc, cm, (ok_imgs, ok_lbls, ok_preds), (no_imgs, no_lbls, no_preds)


def _save_grid(img_tensors, labels, preds, path, title, grid=4):
    """
    将若干张 CxHxW 的张量拼成 grid×grid 的图后保存。
    显示时对多通道取均值成灰度；像素做 0-1 归一化方便可视化。
    """
    if len(img_tensors) == 0:
        print('No samples to save for', title)
        return

    n = min(len(img_tensors), grid * grid)      # 最多显示 grid^2 张
    cols, rows = grid, grid
    plt.figure(figsize=(cols * 2.2, rows * 2.2))
    for i in range(n):
        t = img_tensors[i]  # Tensor [C,H,W]
        if t.dim() == 3:
            if t.size(0) == 1:
                img = t[0].numpy()              # 单通道：直接取第 0 通道
            else:
                img = t.mean(0).numpy()         # 多通道：取均值当灰度展示
        else:
            img = t.numpy()
        # 简单归一化到 [0,1]，避免显示过暗/过亮
        img = (img - img.min()) / (img.max() - img.min() + 1e-9)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        if labels is not None and preds is not None and i < len(labels) and i < len(preds):
            plt.title(f'{labels[i]}→{preds[i]}', fontsize=20)  # 标题“真→预测”
        plt.axis('off')

    plt.suptitle(title)                          # 整体标题
    ensure_dir(os.path.dirname(path) or '.')     # 确保保存目录存在
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved grid to', path)


def _build_test_loader(dataset_name, data_dir, arch, pretrained, batch_size, num_workers):
    """按数据集名字构造测试集 DataLoader。"""
    _, test_t = get_transforms(arch, pretrained, dataset_name)  # 与模型/数据集/预训练范式一致的 transforms
    if dataset_name == 'mnist':
        test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_t)
    elif dataset_name == 'svhn':
        svhn_root = os.path.join(data_dir, 'SVHN')   # 让 SVHN 放在 data_dir/SVHN/ 下
        ensure_dir(svhn_root)
        test_set = datasets.SVHN(root=svhn_root, split='test', download=True, transform=test_t,
                                 target_transform=svhn_label_to_digit)  # 将标签 10 → 0
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')
    return DataLoader(test_set, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)  # 评估不打乱顺序；启用 pin_memory 提升拷贝效率


def _build_train_val_loaders_for_eval(dataset_name, data_dir, arch, pretrained, batch_size, num_workers, val_ratio):
    """
    为了在评估脚本里统计 train/val 的指标，这里复刻训练时的划分方式：
    - 使用相同的训练 transforms（t_train）；
    - 使用 random_split 与相同的 generator 种子（训练脚本里固定为 42）；
    - MNIST: train=True；SVHN: split='train' 且做 10→0 标签映射。
    """
    t_train, _ = get_transforms(arch, pretrained, dataset_name)

    if dataset_name == 'mnist':
        full = datasets.MNIST(root=data_dir, train=True, download=True, transform=t_train)
    elif dataset_name == 'svhn':
        svhn_root = os.path.join(data_dir, 'SVHN')
        ensure_dir(svhn_root)
        full = datasets.SVHN(root=svhn_root, split='train', download=True, transform=t_train,
                             target_transform=svhn_label_to_digit)
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')

    # 按训练脚本的口径：random_split(..., generator=manual_seed(42))
    val_size = int(len(full) * val_ratio)
    train_size = len(full) - val_size
    gen = torch.Generator().manual_seed(42)   # 与 train.py 完全一致
    train_set, val_set = random_split(full, [train_size, val_size], generator=gen)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False,  # 评估不需要打乱
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def _run_and_save(model, device, loader, save_dir, arch, trained_on, eval_on):
    """运行评测并在 save_dir 下保存四类产物（用于跨域测试保持原有四件套）。"""
    ensure_dir(save_dir)
    loss, acc, cm, ok_pack, no_pack = evaluate(model, loader, device)
    ok_imgs, ok_lbls, ok_preds = ok_pack
    no_imgs, no_lbls, no_preds = no_pack

    # 1) 混淆矩阵
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, classes=[str(i) for i in range(10)], save_path=cm_path)
    print('Saved confusion matrix to', cm_path)

    # 2) 指标 JSON（包含训练域/评测域信息，便于课程报告汇总）
    metrics = {
        'arch': arch,
        'trained_on': trained_on,
        'eval_on': eval_on,
        'test_loss': float(loss),
        'test_acc': float(acc)
    }
    metrics_path = os.path.join(save_dir, 'metrics_eval.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print('Saved metrics to', metrics_path)

    # 3) 成功/失败样例网格（各 4×4）
    _save_grid(ok_imgs, ok_lbls, ok_preds, os.path.join(save_dir, 'samples_correct.png'),
               'Correct Samples (4x4)', grid=4)
    _save_grid(no_imgs, no_lbls, no_preds, os.path.join(save_dir, 'samples_miscls.png'),
               'Misclassified Samples (4x4)', grid=4)

    # 控制台提示
    print(f"[{arch}] Trained on {trained_on} | Evaluated on {eval_on} -> loss={loss:.4f} acc={acc:.4f}")


def main():
    # 目录准备（parameters.py 已推断出 eval_dir / cross_dir / weights / save_*）
    ensure_dir(args.eval_dir)    # 主域评测目录：outputs/<dataset>/<arch>/evaluate/
    ensure_dir(args.cross_dir)   # 跨域评测目录：evaluate/cross_to_<other>/

    print('Using weights :', args.weights)
    print('Eval outputs  :', args.eval_dir)
    print('Cross outputs :', args.cross_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, '| Arch:', args.arch, '| Train Dataset:', args.dataset, '| Pretrained:', args.pretrained)

    # -------- 构造模型并加载权重（权重来自训练在 args.dataset 上的 best.pt）--------
    ckpt = torch.load(args.weights, map_location='cpu')      # 统一先映射到 CPU，再 .to(device)
    model = create_model(arch=args.arch, pretrained=args.pretrained, use_relu=args.use_relu)
    model.load_state_dict(ckpt['model_state_dict'])          # 仅加载权重字典
    model.to(device)

    # -------- 主域：构造 train/val/test 三套 DataLoader（评估用）--------
    # 说明：train/val 使用训练时的 t_train 和相同的 random_split( seed=42 )；test 使用 t_test
    train_loader_eval, val_loader_eval = _build_train_val_loaders_for_eval(
        args.dataset, args.data_dir, args.arch, args.pretrained,
        args.eval_batch_size, args.num_workers, args.val_ratio
    )
    main_loader = _build_test_loader(
        args.dataset, args.data_dir, args.arch, args.pretrained,
        args.eval_batch_size, args.num_workers
    )

    # -------- 分别在 train / val / test 上评估 --------
    tr_loss, tr_acc, _, _, _ = evaluate(model, train_loader_eval, device)
    va_loss, va_acc, _, _, _ = evaluate(model, val_loader_eval, device)
    te_loss, te_acc, cm, ok_pack, no_pack = evaluate(model, main_loader, device)
    ok_imgs, ok_lbls, ok_preds = ok_pack
    no_imgs, no_lbls, no_preds = no_pack

    # （仅测试集）保存混淆矩阵与样例网格
    plot_confusion_matrix(cm, classes=[str(i) for i in range(10)], save_path=args.save_cm)  # 混淆矩阵
    _save_grid(ok_imgs, ok_lbls, ok_preds, args.save_samples_correct, 'Correct Samples (4x4)', grid=4)  # 正确样例
    _save_grid(no_imgs, no_lbls, no_preds, args.save_samples_miscls, 'Misclassified Samples (4x4)', grid=4)  # 错误样例

    # 将三套指标写入同一个 metrics_eval.json，便于表格化展示
    metrics = {
        'arch': args.arch,
        'trained_on': args.dataset,
        'eval_on': args.dataset,
        'train_loss': float(tr_loss),
        'train_acc': float(tr_acc),
        'val_loss': float(va_loss),
        'val_acc': float(va_acc),
        'test_loss': float(te_loss),
        'test_acc': float(te_acc)
    }
    with open(args.save_metrics, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[{args.arch}] TRAIN  | loss={tr_loss:.4f} acc={tr_acc:.4f}")
    print(f"[{args.arch}] VAL    | loss={va_loss:.4f} acc={va_acc:.4f}")
    print(f"[{args.arch}] TEST   | loss={te_loss:.4f} acc={te_acc:.4f}")
    print('Saved (main):', args.save_cm, args.save_metrics, args.save_samples_correct, args.save_samples_miscls)

    # -------- 跨域测试（eval_on = other_dataset）--------
    if args.do_cross_eval:  # 开关：是否做跨域评测
        other_dataset = 'svhn' if args.dataset == 'mnist' else 'mnist'
        cross_loader = _build_test_loader(other_dataset, args.data_dir, args.arch, args.pretrained,
                                          args.eval_batch_size, args.num_workers)
        _run_and_save(model, device, cross_loader, args.cross_dir,
                      arch=args.arch, trained_on=args.dataset, eval_on=other_dataset)

    # 保存本次评估用到的参数（便于复现实验/写报告）
    with open(os.path.join(args.eval_dir, 'args_evaluate.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print('Saved args to', os.path.join(args.eval_dir, 'args_evaluate.json'))


if __name__ == '__main__':
    main()  # 程序入口
