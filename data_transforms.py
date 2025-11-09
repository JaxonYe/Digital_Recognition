from typing import Tuple
from torchvision import transforms

# MNIST 数据集的均值与方差（单通道）——常用的标准化参数
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
# ImageNet 的均值与方差（RGB 三通道）——用于“按 ImageNet 预训练范式”的归一化
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def get_transforms(arch: str, pretrained: bool = False, dataset: str = 'mnist') -> Tuple[transforms.Compose, transforms.Compose]:
    """
    根据模型与数据集生成训练/测试用的 torchvision 预处理（仅返回张量空间的变换，不做数据增强）。
    参数：
        arch:        模型字符串标识（lenet/alexnet/vgg16/googlenet/resnet18/densenet121）
        pretrained:  非 LeNet 是否遵循 ImageNet 预训练统计来做归一化（仅影响归一化均值/方差选择）
        dataset:     数据集名称（mnist/svhn）
    返回：
        (train_transforms, test_transforms)  统一返回两个 transforms.Compose
    说明：
        - LeNet：默认按“灰度 1×28×28”处理；
        - 其他大模型：按“RGB 3×224×224”处理（MNIST 会先转 3 通道），和你的手写大模型结构对齐；
        - 若 pretrained=True：使用 ImageNet 的均值方差；否则使用更简单的统计（如 MNIST 三倍通道或 0.5/0.5）。
    """
    arch = arch.lower()      # 防止大小写带来的不一致
    dataset = dataset.lower()

    # ---------- MNIST：保持你的原逻辑 ----------
    if dataset == 'mnist':
        if arch in ['lenet','lenet5','lenet-5']:
            # LeNet：保持 MNIST 原生尺寸与单通道；使用 MNIST 统计做 Normalize
            t = transforms.Compose([
                transforms.ToTensor(),                      # [H,W] -> [1,H,W], 像素值缩放到[0,1]
                transforms.Normalize(MNIST_MEAN, MNIST_STD) # 标准化（单通道）
            ])
            return t, t
        else:
            # 其他大模型：统一转为 3 通道并上采样到 224，以适配手写 AlexNet/VGG/GoogLeNet/ResNet/DenseNet 的输入
            # 若 pretrained=True 用 ImageNet 统计；否则用 "MNIST 统计 × 3 通道"（简单可用）
            mean, std = (IMAGENET_MEAN, IMAGENET_STD) if pretrained else (MNIST_MEAN*3, MNIST_STD*3)
            train_t = transforms.Compose([
                transforms.Resize(224),                      # 统一输入分辨率
                transforms.Grayscale(num_output_channels=3), # 灰度复制到 3 通道
                transforms.ToTensor(),
                transforms.Normalize(mean, std),             # 三通道标准化
            ])
            test_t = train_t
            return train_t, test_t

    # ---------- SVHN：单数字 RGB 32×32，建议保留颜色 ----------
    if dataset == 'svhn':
        if arch in ['lenet','lenet5','lenet-5']:
            # LeNet 仍按 1×28×28（会损失颜色，但结构最简单；也便于与 MNIST 对齐）
            t = transforms.Compose([
                transforms.Resize((28,28)),                  # 缩放到 LeNet 习惯尺寸
                transforms.Grayscale(num_output_channels=1), # 转为单通道
                transforms.ToTensor(),
                # 这里用 (0.5,0.5) 做简单归一化，避免依赖额外统计；当然也可改为自算的 SVHN 灰度统计
                transforms.Normalize((0.5,), (0.5,)),
            ])
            return t, t
        else:
            # 大模型：保留 RGB，并上采样到 224 输入；是否用 ImageNet 统计由 pretrained 决定
            mean, std = (IMAGENET_MEAN, IMAGENET_STD) if pretrained else ((0.5,0.5,0.5), (0.5,0.5,0.5))
            train_t = transforms.Compose([
                transforms.Resize(224),       # SVHN 原图 32×32，放大到 224×224
                # 不做 Grayscale，保留颜色信息以提升对真实街景数字的判别
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            test_t = train_t
            return train_t, test_t

    # 兜底：传入了未支持的数据集名
    raise ValueError(f'Unsupported dataset: {dataset}')
