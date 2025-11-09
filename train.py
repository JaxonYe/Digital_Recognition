import os
import json
import random
import numpy as np
import argparse
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from utils import set_seed, ensure_dir, save_jsonl, plot_curves
from models_factory import create_model
from data_transforms import get_transforms

from parameters import get_args
args = get_args()  # 直接拿；用 args.epochs/batch_size/val_ratio/lr/optimizer/...

"""==================================================================================================="""
# 为保证实验可复现：统一设置 Python/NumPy/PyTorch 的随机数种子，以及 cuDNN 的确定性选项
print('Using random seed : ', args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)  # set random seed for all gpus
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)  # 设置 CUDA 随机数种子（当前 GPU）
    torch.cuda.manual_seed_all(args.seed)  # 设置所有 GPU 的随机数种子
    torch.backends.cudnn.deterministic = True  # 确保每次运行的计算结果一致
    torch.backends.cudnn.benchmark = False  # 禁用动态优化
"""==================================================================================================="""


def train_one_epoch(model, loader, criterion, optimizer, device):
    """单个 epoch 的训练循环：前向 -> 计算损失 -> 反向传播 -> 参数更新，并累计平均 loss/acc。"""
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in tqdm(loader, desc='train', leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += (torch.argmax(logits, dim=1) == y).float().sum().item()
        n += bs
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估循环：仅前向与统计指标，不做反向；返回平均 loss/acc。"""
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in tqdm(loader, desc='eval', leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += (torch.argmax(logits, dim=1) == y).float().sum().item()
        n += bs
    return total_loss / n, total_acc / n


def svhn_label_to_digit(y):
    """SVHN 标签映射：SVHN 中 10 表示数字 0，这里统一转成 0。"""
    y = int(y)
    return 0 if y == 10 else y


def get_loaders(data_dir: str, batch_size: int, arch: str, pretrained: bool,
                val_ratio: float = 0.1, dataset: str = 'mnist'):
    """
    构建 train/val/test 三个 DataLoader。
    - 根据 (arch, pretrained, dataset) 获取与模型/数据集匹配的 transforms；
    - MNIST 与 SVHN 分别按官方接口读取；SVHN 额外做标签 10->0 的映射；
    - 使用 random_split 按 val_ratio 划分验证集。
    """
    # 1) 先拿到 transforms（让它知道是 mnist 还是 svhn，下一步会改 data_transforms.py）
    t_train, t_test = get_transforms(arch, pretrained, dataset)

    if dataset == 'mnist':
        # MNIST: 官方 train/test 划分
        full = datasets.MNIST(root=data_dir, train=True, download=True, transform=t_train)
        test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=t_test)

    elif dataset == 'svhn':
        # SVHN 用 split='train' / 'test'，且可能有 label 10 表示数字 0 —— 稳妥起见做一次映射
        # to_digit = lambda y: 0 if int(y) == 10 else int(y)
        svhn_root = os.path.join(data_dir, 'SVHN')  # 放在 data_dir/SVHN/ 子目录下
        ensure_dir(svhn_root)
        full = datasets.SVHN(root=svhn_root, split='train', download=True, transform=t_train,
                             target_transform=svhn_label_to_digit)
        test_set = datasets.SVHN(root=svhn_root, split='test', download=True, transform=t_test,
                             target_transform=svhn_label_to_digit)
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')

    # 验证集划分（保持你的原逻辑不变）
    val_size = int(len(full) * val_ratio)
    train_size = len(full) - val_size
    gen = torch.Generator().manual_seed(42)  # 固定划分
    train_set, val_set = random_split(full, [train_size, val_size], generator=gen)

    # DataLoader：训练集打乱；验证/测试不打乱；pin_memory 提升主机到 GPU 的拷贝效率
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


def main():
    """训练入口：构建数据/模型/优化器，执行训练-验证循环，保存最好模型与曲线等产物。"""

    set_seed(args.seed)          # 再次显式设置随机种子（工具函数中也会做环境相关设置）
    ensure_dir(args.outputs)     # 确保输出根目录存在

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, '| Arch:', args.arch, '| Pretrained:', args.pretrained)

    # 每个模型一个子文件夹（parameters.py 中已将其设为 outputs/<dataset>/<arch>/train/）
    ensure_dir(args.train_dir)

    # 按当前数据集构建 DataLoader
    train_loader, val_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.arch, args.pretrained, dataset=args.dataset)

    # 创建模型与优化器/损失函数
    model = create_model(arch=args.arch, num_classes=10, pretrained=args.pretrained, use_relu=args.use_relu).to(device)
    criterion = nn.CrossEntropyLoss()
    # 优化器：VGG 用 SGD+momentum 更稳，带权重衰减；其余沿用你的设置 改了后就变正常了！
    if args.arch == 'vgg16':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr) if args.optimizer=='adam' else optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    best_acc = 0.0              # 记录最佳验证集准确率
    history = []                # 保存每个 epoch 的训练/验证指标（用于画曲线）
    log_path = os.path.join(args.train_dir, 'logs.jsonl')   # 训练日志（逐行 JSON）
    best_path = os.path.join(args.train_dir, 'best.pt')     # 最优权重

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        record = {'epoch': epoch, 'train_loss': tr_loss, 'train_acc': tr_acc, 'val_loss': va_loss, 'val_acc': va_acc}
        history.append(record)
        save_jsonl(log_path, record)  # 追加写入一行 JSON
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f}")
        if va_acc > best_acc:
            # 若验证准确率更优，则更新最佳模型
            best_acc = va_acc
            torch.save({'epoch': epoch, 'arch': args.arch, 'model_state_dict': model.state_dict(),
                        'val_acc': va_acc, 'seed': args.seed, 'pretrained': args.pretrained, 'use_relu': args.use_relu}, best_path)

    # 最终在测试集上评估一次（报告参考）
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test | loss={te_loss:.4f} acc={te_acc:.4f}")

    # 画损失/准确率曲线（工具函数内部会根据 history 生成并保存）
    plot_curves(history, os.path.join(args.train_dir, 'curves.png'))
    print('Done. Best val_acc: {:.4f}'.format(best_acc))

    # 额外保存本次训练所用的所有参数，便于复现/写报告
    with open(os.path.join(args.train_dir, 'args_train.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
