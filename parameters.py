# -*- coding: utf-8 -*-
"""
parameters.py
- 统一参数（训练/评估/跨域），一个入口：get_args()
- 全部以下划线命名：batch_size / eval_batch_size / val_ratio 等
- 默认不走命令行：parse_args([])；只改本文件默认值即可
- 路径规则：
    out_dir   = outputs/<dataset>/<arch>/
    train_dir = out_dir/train/
    eval_dir  = out_dir/evaluate/
  评估默认读取 train_dir/best.pt，评估结果写到 eval_dir/ 下
"""
import argparse
import os
import platform

_DEFAULT_WORKERS = 0 if platform.system().lower().startswith("win") else 2


# ----------------------- 分组增参 -----------------------
def basic_parameters(parser: argparse.ArgumentParser):
    # Dataset / IO
    parser.add_argument('--dataset',   default='mnist', type=str,
                        choices=['mnist', 'svhn'], help='数据集')
    parser.add_argument('--data_dir',  default='data',  type=str, help='数据集根目录')
    parser.add_argument('--outputs',   default='outputs', type=str, help='输出根目录（outputs/<dataset>/<arch>/）')

    # Model
    parser.add_argument('--arch',       default='googlenet', type=str,
                        choices=['lenet','alexnet','vgg16','googlenet','resnet18','densenet121'], help='模型结构')
    parser.add_argument('--pretrained', default=False,   type=bool, help='非 LeNet 是否使用 ImageNet 预训练')
    parser.add_argument('--use_relu',   default=False,   type=bool, help='仅 LeNet 生效')

    # Env
    parser.add_argument('--seed',        default=1, type=int, help='随机种子')
    parser.add_argument('--num_workers', default=_DEFAULT_WORKERS, type=int,
                        help='DataLoader workers（Windows 建议 0）')
    return parser


def training_parameters(parser: argparse.ArgumentParser):
    parser.add_argument('--epochs',      default=50,     type=int,   help='训练轮数')
    parser.add_argument('--batch_size',  default=32,   type=int,   help='训练/验证 batch 大小')
    parser.add_argument('--val_ratio',   default=0.1,   type=float, help='验证集比例')
    parser.add_argument('--lr',          default=1e-3,  type=float, help='学习率')
    parser.add_argument('--optimizer',   default='adam', type=str,  choices=['adam','sgd'], help='优化器')
    return parser


def evaluation_parameters(parser: argparse.ArgumentParser):
    # 评估 batch
    parser.add_argument('--eval_batch_size', default=256, type=int, help='评估 batch 大小')

    # 权重 & 保存路径（不传则自动按 train_dir / eval_dir 推断）
    parser.add_argument('--weights',             default=None, type=str,
                        help='权重路径（默认 out_dir/train/best.pt）')
    parser.add_argument('--save_cm',             default=None, type=str,
                        help='confusion_matrix.png（默认 out_dir/evaluate/）')
    parser.add_argument('--save_metrics',        default=None, type=str,
                        help='metrics_eval.json（默认 out_dir/evaluate/）')
    parser.add_argument('--save_samples_correct', default=None, type=str,
                        help='分类成功样例 4x4（默认 out_dir/evaluate/）')
    parser.add_argument('--save_samples_miscls',  default=None, type=str,
                        help='分类失败样例 4x4（默认 out_dir/evaluate/）')

    # 跨域评测
    parser.add_argument('--do_cross_eval',  default=True,  type=bool,
                        help='是否同时做一次跨域评测（MNIST↔SVHN）')
    parser.add_argument('--cross_dir_name', default=None,  type=str,
                        help='跨域结果子目录名，默认 cross_to_<other>（位于 out_dir/evaluate/ 下）')
    return parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Unified parameters for train/evaluate')
    basic_parameters(p)
    training_parameters(p)
    evaluation_parameters(p)
    return p


# ----------------------- 路径推断 -----------------------
def _postprocess(args: argparse.Namespace) -> argparse.Namespace:
    # 统一根目录
    args.out_dir   = os.path.join(args.outputs, args.dataset, args.arch)
    args.train_dir = os.path.join(args.out_dir, 'train')
    args.eval_dir  = os.path.join(args.out_dir, 'evaluate')

    # 权重默认来自训练目录
    if args.weights is None:
        args.weights = os.path.join(args.train_dir, 'best.pt')

    # 评估默认保存到 eval_dir
    if args.save_cm is None:
        args.save_cm = os.path.join(args.eval_dir, 'confusion_matrix.png')
    if args.save_metrics is None:
        args.save_metrics = os.path.join(args.eval_dir, 'metrics_eval.json')
    if args.save_samples_correct is None:
        args.save_samples_correct = os.path.join(args.eval_dir, 'samples_correct.png')
    if args.save_samples_miscls is None:
        args.save_samples_miscls = os.path.join(args.eval_dir, 'samples_miscls.png')

    # 跨域子目录名 & 路径（位于 eval_dir 下）
    if args.cross_dir_name is None:
        other = 'svhn' if args.dataset == 'mnist' else 'mnist'
        args.cross_dir_name = f'cross_to_{other}'
    args.cross_dir = os.path.join(args.eval_dir, args.cross_dir_name)

    return args


# ----------------------- 唯一入口 -----------------------
def get_args(cli: bool = False) -> argparse.Namespace:
    """
    返回包含训练+评估全部字段的 args（全部用下划线命名）。
      - cli=False（默认）：不走命令行，使用本文件默认值
      - cli=True：允许命令行覆盖
    """
    parser = build_parser()
    args = parser.parse_args([] if not cli else None)
    return _postprocess(args)
