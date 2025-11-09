# -*- coding: utf-8 -*-
"""
models_factory.py
- 不再从 torchvision 调包，全部使用 model.py 中的手写结构
- 保持 create_model(...) 签名不变：arch, num_classes, pretrained, use_relu
  * 对于非 LeNet 的模型，pretrained 参数仅用于数据预处理决策；模型本身不会加载预训练权重
"""
from typing import Optional
import warnings
import torch.nn as nn

from model import LeNet5, AlexNet, VGG16, GoogLeNet, ResNet18, DenseNet121


def create_model(arch: str = 'lenet',
                 num_classes: int = 10,
                 pretrained: bool = False,
                 use_relu: bool = False) -> nn.Module:
    """
    根据字符串创建模型实例。
    :param arch: 'lenet' | 'alexnet' | 'vgg16' | 'googlenet' | 'resnet18' | 'densenet121'
    :param num_classes: 分类数
    :param pretrained: 手写模型不加载任何预训练；该参数仅用于你在 data_transforms 中决定是否用 ImageNet 统计
    :param use_relu: 仅对 LeNet 生效（切换 Tanh/RELU）
    """
    arch = arch.lower()
    if arch == 'lenet':
        model = LeNet5(num_classes=num_classes, use_relu=use_relu)
        if pretrained:
            warnings.warn("手写 LeNet5 不支持预训练权重；`pretrained=True` 将被忽略。", UserWarning)
        return model

    # 非 LeNet：全部是 3 通道起步的经典结构；为了兼容 MNIST，在 forward 内已自动把 1 通道 repeat 成 3 通道
    if arch == 'alexnet':
        model = AlexNet(num_classes=num_classes)
    elif arch == 'vgg16':
        model = VGG16(num_classes=num_classes, batch_norm=True)   # 如需 BN 可改 True
    elif arch == 'googlenet':
        model = GoogLeNet(num_classes=num_classes, batch_norm=True)    # 如需 BN 可改 True
    elif arch == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    elif arch == 'densenet121':
        model = DenseNet121(num_classes=num_classes)
    else:
        raise ValueError(f'Unsupported arch: {arch}')

    if pretrained:
        warnings.warn(f"手写 {arch} 不支持预训练权重；`pretrained=True` 将被忽略（仅影响你的 transforms 选择）。", UserWarning)
    return model
