# -*- coding: utf-8 -*-
"""
model.py
- 纯手写的经典 CNN 结构：LeNet5 / AlexNet / VGG16 / GoogLeNet (Inception v1) / ResNet18 / DenseNet121
- 适配 1 通道（MNIST）与 3 通道（SVHN）：大模型在 forward 中遇到 C=1 会自动 repeat 成 3 通道
- 广义输入尺寸：各模型使用自适应池化避免 fc 尺寸绑定到某个固定输入大小
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------- 小工具 ---------------------------

def _to_3ch_if_needed(x: torch.Tensor) -> torch.Tensor:
    """若输入为 1 通道，自动复制为 3 通道；否则原样返回。"""
    if x.dim() == 4 and x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
    return x


# --------------------------- LeNet-5 ---------------------------

class LeNet5(nn.Module):
    """
    经典 LeNet-5：支持 use_relu 切换 Tanh/RELU；使用自适应池化到 4x4 以消除输入分辨率依赖。
    默认按单通道设计；若输入为 3 通道，会在 forward 内自动均值到 1 通道以贴近原论文风格。
    """
    def __init__(self, num_classes: int = 10, use_relu: bool = False):
        super().__init__()
        self.act = nn.ReLU(inplace=True) if use_relu else nn.Tanh()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # 原版为 avg pool
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        # 自适应到 4x4，保证后续 FC 维度稳定
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 如果是 3 通道，转成灰度（贴近 LeNet 设定）
        if x.size(1) == 3:
            x = x.mean(dim=1, keepdim=True)

        x = self.act(self.conv1(x))
        x = self.pool(x)
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = self.adapt(x)               # [B,16,4,4]
        x = torch.flatten(x, 1)         # [B,256]
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x


# --------------------------- AlexNet ---------------------------

class AlexNet(nn.Module):
    """
    纯手写 AlexNet（简化：不含 LRN，使用自适应池化到 6x6，与经典实现一致）
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_3ch_if_needed(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# --------------------------- VGG16 ---------------------------

def _make_vgg_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

_VGG16_CFG = [64, 64, 'M',
              128, 128, 'M',
              256, 256, 256, 'M',
              512, 512, 512, 'M',
              512, 512, 512, 'M']

class VGG16(nn.Module):
    """
    纯手写 VGG16（可选 BN：这里默认不加 BN；自适应池化到 7x7）
    """
    def __init__(self, num_classes: int = 10, batch_norm: bool = False):
        super().__init__()
        self.features = _make_vgg_layers(_VGG16_CFG, batch_norm=batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),  # 0.5改为0.3
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_3ch_if_needed(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# --------------------------- GoogLeNet (Inception v1) ---------------------------

class Inception(nn.Module):
    """
    Inception v1 基本模块（4 分支）
    输入 C -> concat 输出 C_out = b1 + b3 + b5 + bp
    """
    def __init__(self, in_ch, b1, b3_reduce, b3, b5_reduce, b5, pool_proj, batch_norm):
        super().__init__()
        if batch_norm:
            # 1x1 分支  加入BN
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_ch, b1, kernel_size=1, bias=False),
                nn.BatchNorm2d(b1),
                nn.ReLU(inplace=True)
            )
            # 1x1 -> 3x3 分支
            self.branch3 = nn.Sequential(
                nn.Conv2d(in_ch, b3_reduce, kernel_size=1, bias=False),
                nn.BatchNorm2d(b3_reduce),
                nn.ReLU(inplace=True),
                nn.Conv2d(b3_reduce, b3, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(b3),
                nn.ReLU(inplace=True)
            )
            # 1x1 -> 5x5 分支（5x5 计算量较大，但在本任务仍可用）
            self.branch5 = nn.Sequential(
                nn.Conv2d(in_ch, b5_reduce, kernel_size=1, bias=False),
                nn.BatchNorm2d(b5_reduce),
                nn.ReLU(inplace=True),
                nn.Conv2d(b5_reduce, b5, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm2d(b5),
                nn.ReLU(inplace=True)
            )
            # 池化 -> 1x1 分支
            self.branch_pool = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_ch, pool_proj, kernel_size=1, bias=False),
                nn.BatchNorm2d(pool_proj),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_ch, b1, kernel_size=1), nn.ReLU(inplace=True)
            )
            self.branch3 = nn.Sequential(
                nn.Conv2d(in_ch, b3_reduce, kernel_size=1), nn.ReLU(inplace=True),
                nn.Conv2d(b3_reduce, b3, kernel_size=3, padding=1), nn.ReLU(inplace=True)
            )
            self.branch5 = nn.Sequential(
                nn.Conv2d(in_ch, b5_reduce, kernel_size=1), nn.ReLU(inplace=True),
                nn.Conv2d(b5_reduce, b5, kernel_size=5, padding=2), nn.ReLU(inplace=True)
            )
            self.branch_pool = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_ch, pool_proj, kernel_size=1), nn.ReLU(inplace=True)
            )

    def forward(self, x):
        y1 = self.branch1(x)
        y3 = self.branch3(x)
        y5 = self.branch5(x)
        yp = self.branch_pool(x)
        return torch.cat([y1, y3, y5, yp], dim=1)


class GoogLeNet(nn.Module):
    """
    纯手写 GoogLeNet（不带 AuxLogits），自适应全局平均池化后接全连接。
    结构参照 Inception v1（简化版本，足够 MNIST/SVHN 分类实验）。
    """
    def __init__(self, num_classes: int = 10, batch_norm: bool = False):
        super().__init__()
        if batch_norm:
            # Stem：Conv -> BN -> ReLU，去掉 LRN，用 BN 提升稳定性
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),

                nn.Conv2d(64, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),

                nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1)
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1)
            )
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32, batch_norm)   # out 256
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64, batch_norm) # out 480
        self.pool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64, batch_norm)  # out 512
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64, batch_norm) # out 512
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64, batch_norm) # out 512
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64, batch_norm) # out 528
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128, batch_norm) # out 832
        self.pool4 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128, batch_norm) # out 832
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128, batch_norm) # out 1024

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(0.4)
        self.dropout = nn.Dropout(0.2)  # 从 0.4 降到 0.2，利于从零训练小数据
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_3ch_if_needed(x)
        x = self.stem(x)
        x = self.inception3a(x); x = self.inception3b(x); x = self.pool3(x)
        x = self.inception4a(x); x = self.inception4b(x); x = self.inception4c(x); x = self.inception4d(x); x = self.inception4e(x); x = self.pool4(x)
        x = self.inception5a(x); x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# --------------------------- ResNet-18 ---------------------------

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """
    纯手写 ResNet-18
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64,  64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * BasicBlock.expansion)
            )
        layers = [BasicBlock(in_ch, out_ch, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch * BasicBlock.expansion, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_3ch_if_needed(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# --------------------------- DenseNet-121 ---------------------------

class _DenseLayer(nn.Module):
    def __init__(self, in_ch, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        inter = bn_size * growth_rate
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, inter, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(inter)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        if self.drop_rate > 0.0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, in_ch, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(num_layers):
            layers.append(_DenseLayer(ch, growth_rate, bn_size, drop_rate))
            ch += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_ch = ch

    def forward(self, x):
        return self.block(x)


class _Transition(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x


class DenseNet121(nn.Module):
    """
    纯手写 DenseNet-121：growth_rate=32，block_config=(6,12,24,16)
    """
    def __init__(self, num_classes: int = 10, growth_rate=32, block_config=(6,12,24,16), bn_size=4, drop_rate=0.0):
        super().__init__()
        num_init_features = 64

        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Dense Blocks
        num_features = num_init_features
        self.block1 = _DenseBlock(block_config[0], num_features, growth_rate, bn_size, drop_rate)
        num_features = self.block1.out_ch
        self.trans1 = _Transition(num_features, num_features // 2)
        num_features = num_features // 2

        self.block2 = _DenseBlock(block_config[1], num_features, growth_rate, bn_size, drop_rate)
        num_features = self.block2.out_ch
        self.trans2 = _Transition(num_features, num_features // 2)
        num_features = num_features // 2

        self.block3 = _DenseBlock(block_config[2], num_features, growth_rate, bn_size, drop_rate)
        num_features = self.block3.out_ch
        self.trans3 = _Transition(num_features, num_features // 2)
        num_features = num_features // 2

        self.block4 = _DenseBlock(block_config[3], num_features, growth_rate, bn_size, drop_rate)
        num_features = self.block4.out_ch

        self.norm_final = nn.BatchNorm2d(num_features)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_3ch_if_needed(x)
        x = self.features(x)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.trans3(self.block3(x))
        x = self.relu_final(self.norm_final(self.block4(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
