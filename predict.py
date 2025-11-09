# -*- coding: utf-8 -*-
"""
predict.py （PyQt5 图形界面 · 加入卷积核权重可视化）
- 画板黑底白字
- 中间层激活可视化：切换层会自动对“上一次输入”复跑前向，稳定拿到新层激活
- 新增：卷积核权重可视化（同一层选择器，最多显示16个核，seismic零中心色图）
- 保留：中文字体自适应、概率条形图、权重自动定位
"""
import os
import io
import sys
from typing import Optional, List

import torch
from torch import nn
from PIL import Image
from torchvision import transforms

from PyQt5 import QtCore, QtGui, QtWidgets

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import font_manager as fm
import numpy as np

from models_factory import create_model
from data_transforms import get_transforms

ARCH_CHOICES = ['lenet', 'alexnet', 'vgg16', 'googlenet', 'resnet18', 'densenet121']
DATASET_CHOICES = ['mnist', 'svhn']

# -------------------------- 字体与中文支持 --------------------------
def _setup_matplotlib_fonts() -> bool:
    candidates = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'PingFang SC', 'STHeiti', 'WenQuanYi Zen Hei']
    have = [f.name for f in fm.fontManager.ttflist]
    for name in candidates:
        if any(name == n or name in n for n in have):
            matplotlib.rcParams['font.sans-serif'] = [name]
            matplotlib.rcParams['axes.unicode_minus'] = False
            return True
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    return False

ZH_FONT_OK = _setup_matplotlib_fonts()

# -------------------------- Matplotlib Canvas --------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, width=4.8, height=3.8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_tight_layout(True)
        super().__init__(self.fig)

# -------------------------- 小工具 --------------------------
def is_lenet(arch: str) -> bool:
    return arch.lower() in ['lenet', 'lenet5', 'lenet-5']

def locate_default_weights(dataset: str, arch: str) -> str:
    candidates = [
        os.path.join('outputs', dataset, arch, 'train', 'best.pt'),
        os.path.join('outputs', dataset, arch, 'best.pt'),
        os.path.join('outputs', arch, 'train', 'best.pt'),
        os.path.join('outputs', arch, 'best.pt'),
    ]
    if arch == 'lenet':
        candidates.append(os.path.join('outputs', 'lenet_best.pt'))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return ''

def build_transform_for_predict(arch: str, pretrained: bool, dataset: str) -> transforms.Compose:
    _, test_t = get_transforms(arch, pretrained, dataset)
    return test_t

def tensor_from_pil(pil_img: Image.Image, arch: str, pretrained: bool, dataset: str, source: str) -> torch.Tensor:
    if dataset == 'svhn' and not is_lenet(arch):
        if source == 'canvas' and pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
    test_t = build_transform_for_predict(arch, pretrained, dataset)
    x = test_t(pil_img)
    return x.unsqueeze(0)

def topk_from_logits(logits: torch.Tensor, k: int = 10):
    probs = torch.softmax(logits, dim=1).squeeze(0)
    topk_probs, topk_idx = torch.topk(probs, k=min(k, probs.numel()))
    return topk_idx.tolist(), topk_probs.tolist(), probs.tolist()

# -------------------------- 画板控件（黑底白字） --------------------------
class PaintCanvas(QtWidgets.QLabel):
    def __init__(self, width=360, height=360, pen_width=20, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.canvas = QtGui.QPixmap(width, height)
        self.canvas.fill(QtCore.Qt.black)     # 黑底
        self.setPixmap(self.canvas)
        self.last_pos = None
        self.pen_width = pen_width
        self.pen_color = QtCore.Qt.white      # 白笔
        self.setCursor(QtCore.Qt.CrossCursor)

    def set_pen_width(self, w: int):
        self.pen_width = max(1, int(w))

    def clear(self):
        self.canvas.fill(QtCore.Qt.black)
        self.setPixmap(self.canvas)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.last_pos = event.pos()
            self._draw_point(self.last_pos)

    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            return
        painter = QtGui.QPainter(self.canvas)
        pen = QtGui.QPen(self.pen_color, self.pen_width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(self.last_pos, event.pos())
        painter.end()
        self.setPixmap(self.canvas)
        self.last_pos = event.pos()

    def mouseReleaseEvent(self, event):
        self.last_pos = None

    def _draw_point(self, pos):
        painter = QtGui.QPainter(self.canvas)
        pen = QtGui.QPen(self.pen_color, self.pen_width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawPoint(pos)
        painter.end()
        self.setPixmap(self.canvas)

    def get_pil_image(self, need_rgb=False) -> Image.Image:
        qimg = self.canvas.toImage().convertToFormat(QtGui.QImage.Format_RGB32)
        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QIODevice.WriteOnly)
        qimg.save(buffer, "PNG")
        pil = Image.open(io.BytesIO(buffer.data()))
        pil = pil.convert('RGB') if need_rgb else pil.convert('L')
        return pil

# -------------------------- 主窗口 --------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("数字识别 · PyQt5 Demo")
        self.setMinimumSize(1220, 780)

        # 顶部配置
        self.arch_combo = QtWidgets.QComboBox(); self.arch_combo.addItems(ARCH_CHOICES)
        self.dataset_combo = QtWidgets.QComboBox(); self.dataset_combo.addItems(DATASET_CHOICES)
        self.use_relu_cb = QtWidgets.QCheckBox("LeNet 使用 ReLU")
        self.pretrained_cb = QtWidgets.QCheckBox("使用 ImageNet 归一化（非预训练）")
        self.weights_edit = QtWidgets.QLineEdit()
        self.weights_browse = QtWidgets.QPushButton("浏览权重")
        self.weights_auto = QtWidgets.QPushButton("自动定位")
        self.load_btn = QtWidgets.QPushButton("加载模型")

        # 画板
        self.canvas = PaintCanvas(width=400, height=400, pen_width=22)
        self.pen_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pen_slider.setRange(1, 72)
        self.pen_slider.setValue(self.canvas.pen_width)
        self.pen_slider.valueChanged.connect(self.canvas.set_pen_width)
        self.clear_btn = QtWidgets.QPushButton("清空画板")

        # 中部：推理与结果表
        self.predict_canvas_btn = QtWidgets.QPushButton("推理（画板）")
        self.open_img_btn = QtWidgets.QPushButton("打开图片…")
        self.predict_file_btn = QtWidgets.QPushButton("推理（图片）")
        self.pred_label = QtWidgets.QLabel("预测：—")
        self.pred_label.setStyleSheet("font-size: 26px; font-weight: bold;")
        self.prob_table = QtWidgets.QTableWidget(10, 2)
        self.prob_table.setHorizontalHeaderLabels(["类别", "概率"] if ZH_FONT_OK else ["Class", "Prob"])
        self.prob_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.prob_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        for r in range(10):
            self.prob_table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(r)))
            self.prob_table.setItem(r, 1, QtWidgets.QTableWidgetItem("—"))

        # 右侧 tabs
        right_tabs = QtWidgets.QTabWidget()

        # Tab1 概率图
        self.prob_canvas = MplCanvas()
        self.pred_big = QtWidgets.QLabel("Top-1：—" if ZH_FONT_OK else "Top-1: —")
        self.pred_big.setAlignment(QtCore.Qt.AlignCenter)
        self.pred_big.setStyleSheet("font-size: 28px; font-weight: bold;")
        prob_tab = QtWidgets.QWidget(); v1 = QtWidgets.QVBoxLayout(prob_tab)
        v1.addWidget(self.pred_big); v1.addWidget(self.prob_canvas, 1)
        right_tabs.addTab(prob_tab, "预测可视化" if ZH_FONT_OK else "Prediction")

        # Tab2 中间层特征
        self.layer_combo = QtWidgets.QComboBox()
        self.refresh_layers_btn = QtWidgets.QPushButton("刷新层列表" if ZH_FONT_OK else "Refresh Layers")
        self.feature_canvas = MplCanvas()
        feat_tab = QtWidgets.QWidget(); v2 = QtWidgets.QVBoxLayout(feat_tab)
        h2 = QtWidgets.QHBoxLayout()
        h2.addWidget(QtWidgets.QLabel("特征层：" if ZH_FONT_OK else "Layer:"))
        h2.addWidget(self.layer_combo, 1)
        h2.addWidget(self.refresh_layers_btn)
        v2.addLayout(h2); v2.addWidget(self.feature_canvas, 1)
        right_tabs.addTab(feat_tab, "中间层特征" if ZH_FONT_OK else "Feature Maps")

        # Tab3 卷积核权重（新增）
        self.kernel_info = QtWidgets.QLabel("—")
        self.kernel_canvas = MplCanvas()
        kernel_tab = QtWidgets.QWidget(); v3 = QtWidgets.QVBoxLayout(kernel_tab)
        v3.addWidget(self.kernel_info)
        v3.addWidget(self.kernel_canvas, 1)
        right_tabs.addTab(kernel_tab, "卷积核权重" if ZH_FONT_OK else "Kernels")

        # 设备/状态
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[nn.Module] = None
        self.hook_handle = None
        self.last_activation: Optional[torch.Tensor] = None

        # 记住最近一次推理输入（用于切层后自动重推）
        self.last_pil: Optional[Image.Image] = None
        self.last_source: Optional[str] = None
        self.last_arch: Optional[str] = None
        self.last_dataset: Optional[str] = None
        self.last_img_path: Optional[str] = None

        # 布局与信号
        self._build_layout(right_tabs)
        self._connect_signals()
        self.statusBar().showMessage(f"Device: {self.device}")

    def _build_layout(self, right_tabs: QtWidgets.QTabWidget):
        top = QtWidgets.QGridLayout()
        top.addWidget(QtWidgets.QLabel("模型：" if ZH_FONT_OK else "Model:"), 0, 0)
        top.addWidget(self.arch_combo, 0, 1)
        top.addWidget(QtWidgets.QLabel("数据集：" if ZH_FONT_OK else "Dataset:"), 0, 2)
        top.addWidget(self.dataset_combo, 0, 3)
        top.addWidget(self.use_relu_cb, 0, 4)
        top.addWidget(self.pretrained_cb, 0, 5)
        top.addWidget(QtWidgets.QLabel("权重路径：" if ZH_FONT_OK else "Weights:"), 1, 0)
        top.addWidget(self.weights_edit, 1, 1, 1, 3)
        top.addWidget(self.weights_browse, 1, 4)
        top.addWidget(self.weights_auto, 1, 5)
        top.addWidget(self.load_btn, 1, 6)

        left_box = QtWidgets.QGroupBox("画板" if ZH_FONT_OK else "Canvas")
        lv = QtWidgets.QVBoxLayout(left_box)
        lv.addWidget(self.canvas)
        lrow = QtWidgets.QHBoxLayout()
        lrow.addWidget(QtWidgets.QLabel("画笔粗细" if ZH_FONT_OK else "Pen"))
        lrow.addWidget(self.pen_slider, 1)
        lrow.addWidget(self.clear_btn)
        lv.addLayout(lrow)

        mid_box = QtWidgets.QGroupBox("推理与结果" if ZH_FONT_OK else "Inference & Results")
        mv = QtWidgets.QVBoxLayout(mid_box)
        brow = QtWidgets.QHBoxLayout()
        brow.addWidget(self.predict_canvas_btn)
        brow.addSpacing(10)
        brow.addWidget(self.open_img_btn)
        brow.addWidget(self.predict_file_btn)
        mv.addLayout(brow)
        mv.addWidget(self.pred_label)
        mv.addWidget(self.prob_table, 1)

        central = QtWidgets.QWidget(); grid = QtWidgets.QGridLayout(central)
        grid.addLayout(top, 0, 0, 1, 3)
        grid.addWidget(left_box, 1, 0)
        grid.addWidget(mid_box, 1, 1)
        grid.addWidget(right_tabs, 1, 2)
        self.setCentralWidget(central)

    def _connect_signals(self):
        self.weights_browse.clicked.connect(self.on_browse_weights)
        self.weights_auto.clicked.connect(self.on_auto_weights)
        self.load_btn.clicked.connect(self.on_load_model)
        self.clear_btn.clicked.connect(self.canvas.clear)
        self.predict_canvas_btn.clicked.connect(self.on_predict_canvas)
        self.open_img_btn.clicked.connect(self.on_open_image)
        self.predict_file_btn.clicked.connect(self.on_predict_file)
        self.refresh_layers_btn.clicked.connect(self.on_refresh_layers)
        self.layer_combo.currentIndexChanged.connect(self.on_change_layer)

    # -------------------------- 模型与 Hook --------------------------
    def on_browse_weights(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择权重文件" if ZH_FONT_OK else "Select Weights",
                                                        ".", "PyTorch Weights (*.pt *.pth)")
        if path:
            self.weights_edit.setText(path)

    def on_auto_weights(self):
        arch = self.arch_combo.currentText().lower()
        dataset = self.dataset_combo.currentText().lower()
        path = locate_default_weights(dataset, arch)
        if path:
            self.weights_edit.setText(path)
            self.statusBar().showMessage(f"Auto: {path}")
        else:
            QtWidgets.QMessageBox.warning(self, "未找到" if ZH_FONT_OK else "Not Found",
                                          "未能在默认位置找到 best.pt，请手动选择。" if ZH_FONT_OK else
                                          "Best weights not found. Please select manually.")

    def _clear_hook(self):
        if self.hook_handle is not None:
            try:
                self.hook_handle.remove()
            except Exception:
                pass
        self.hook_handle = None
        self.last_activation = None

    def _register_activation_hook(self, layer_name: str):
        self._clear_hook()
        if self.model is None:
            return
        name_to_module = dict(self.model.named_modules())
        if layer_name not in name_to_module:
            return
        target = name_to_module[layer_name]

        def _hook(module, inp, out):
            with torch.no_grad():
                t = out[0] if isinstance(out, (list, tuple)) else out
                if isinstance(t, torch.Tensor) and t.dim() == 4 and t.size(0) >= 1:
                    self.last_activation = t.detach().cpu()[0]  # [C,H,W]
                else:
                    self.last_activation = None

        self.hook_handle = target.register_forward_hook(_hook)

    def _list_conv_layers(self) -> List[str]:
        if self.model is None:
            return []
        return [n for n, m in self.model.named_modules() if isinstance(m, nn.Conv2d)]

    def on_refresh_layers(self):
        if self.model is None:
            QtWidgets.QMessageBox.information(self, "提示" if ZH_FONT_OK else "Info",
                                              "请先加载模型。" if ZH_FONT_OK else "Please load model first.")
            return
        layers = self._list_conv_layers()
        self.layer_combo.clear()
        if layers:
            self.layer_combo.addItems(layers)
            self._register_activation_hook(layers[-1])  # 默认最后一层
            # 卷积核：也立即绘制
            self._draw_kernels(layers[-1])
            # 若有上一张输入，自动复跑一次前向，拿到默认层激活
            if self.last_pil is not None:
                self._infer_and_show(self.last_pil.copy(), self.last_arch, self.last_dataset, self.last_source)
            else:
                self._draw_feature_maps(None, title=("特征层：%s（请推理）" if ZH_FONT_OK else "Layer: %s (infer)") % layers[-1])
        else:
            self.layer_combo.addItem("(无可视化层)" if ZH_FONT_OK else "(No Conv Layer)")
            self._clear_hook()
            self._draw_feature_maps(None, title="—")
            self._clear_kernels_info()

    def on_change_layer(self, idx: int):
        if self.model is None:
            return
        name = self.layer_combo.currentText()
        if not name or name.startswith("("):
            return
        self._register_activation_hook(name)
        # 立刻更新卷积核图
        self._draw_kernels(name)
        # 若已有上一张输入，立刻复跑一次前向并更新特征图
        if self.last_pil is not None and self.last_arch and self.last_dataset and self.last_source:
            self._infer_and_show(self.last_pil.copy(), self.last_arch, self.last_dataset, self.last_source)
        else:
            self._draw_feature_maps(None, title=("特征层：%s（请推理）" if ZH_FONT_OK else "Layer: %s (infer)") % name)

    # -------------------------- 推理流程 --------------------------
    def on_load_model(self):
        arch = self.arch_combo.currentText().lower()
        dataset = self.dataset_combo.currentText().lower()
        weights = self.weights_edit.text().strip()
        use_relu = self.use_relu_cb.isChecked()
        pretrained = self.pretrained_cb.isChecked()

        if not os.path.isfile(weights):
            QtWidgets.QMessageBox.warning(self, "错误" if ZH_FONT_OK else "Error",
                                          "请先选择有效的权重文件（.pt / .pth）" if ZH_FONT_OK else "Select a valid weights file.")
            return
        try:
            ckpt = torch.load(weights, map_location='cpu')
            state = ckpt.get('model_state_dict', ckpt)
            self.model = create_model(arch=arch, num_classes=10, pretrained=pretrained, use_relu=use_relu)
            self.model.load_state_dict(state)
            self.model.to(self.device).eval()
            self.statusBar().showMessage(f"Loaded: {arch} | {dataset} | {os.path.basename(weights)}")
            QtWidgets.QMessageBox.information(self, "成功" if ZH_FONT_OK else "Success",
                                              f"已加载 {arch} 的权重。" if ZH_FONT_OK else f"Loaded weights for {arch}.")
            # 刷新层列表（含卷积核图、特征图）
            self.on_refresh_layers()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "加载失败" if ZH_FONT_OK else "Load Failed", str(e))

    def on_open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图片" if ZH_FONT_OK else "Open Image",
                                                        ".", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.last_img_path = path
            QtWidgets.QMessageBox.information(self, "已选择" if ZH_FONT_OK else "Selected", os.path.basename(path))

    def on_predict_canvas(self):
        if self.model is None:
            QtWidgets.QMessageBox.warning(self, "未加载模型" if ZH_FONT_OK else "No Model",
                                          "请先加载模型权重。" if ZH_FONT_OK else "Please load model first.")
            return
        arch = self.arch_combo.currentText().lower()
        dataset = self.dataset_combo.currentText().lower()
        need_rgb = (dataset == 'svhn' and not is_lenet(arch))
        pil = self.canvas.get_pil_image(need_rgb=need_rgb)

        self.last_pil = pil.copy()
        self.last_source = 'canvas'
        self.last_arch = arch
        self.last_dataset = dataset

        self._infer_and_show(pil, arch, dataset, source='canvas')

    def on_predict_file(self):
        if self.model is None:
            QtWidgets.QMessageBox.warning(self, "未加载模型" if ZH_FONT_OK else "No Model",
                                          "请先加载模型权重。" if ZH_FONT_OK else "Please load model first.")
            return
        if not self.last_img_path or not os.path.isfile(self.last_img_path):
            QtWidgets.QMessageBox.warning(self, "未选择图片" if ZH_FONT_OK else "No Image",
                                          "请先选择一张图片。" if ZH_FONT_OK else "Please select an image.")
            return
        arch = self.arch_combo.currentText().lower()
        dataset = self.dataset_combo.currentText().lower()
        pil = Image.open(self.last_img_path)
        pil = pil.convert('L') if dataset == 'mnist' else pil.convert('RGB')

        self.last_pil = pil.copy()
        self.last_source = 'file'
        self.last_arch = arch
        self.last_dataset = dataset

        self._infer_and_show(pil, arch, dataset, source='file')

    def _infer_and_show(self, pil: Image.Image, arch: str, dataset: str, source: str):
        pretrained = self.pretrained_cb.isChecked()
        with torch.no_grad():
            x = tensor_from_pil(pil, arch, pretrained, dataset, source=source).to(self.device)
            logits = self.model(x)
            idx, probs_top, probs_all = topk_from_logits(logits, k=10)

        # 概率条形图 + 表格 + 文本
        self._draw_prob_bar(probs_all, top1=int(idx[0]), top1p=float(probs_top[0]))
        self._fill_prob_table(probs_all)
        if ZH_FONT_OK:
            self.pred_label.setText(f"预测：{int(idx[0])}（置信度 {float(probs_top[0]):.3f}）")
            self.pred_big.setText(f"Top-1：{int(idx[0])}  |  p={float(probs_top[0]):.3f}")
        else:
            self.pred_label.setText(f"Pred: {int(idx[0])} (p={float(probs_top[0]):.3f})")
            self.pred_big.setText(f"Top-1: {int(idx[0])}  |  p={float(probs_top[0]):.3f}")

        # 特征图（若 hook 捕获到）
        name = self.layer_combo.currentText()
        title = (f"特征层：{name}" if ZH_FONT_OK else f"Layer: {name}")
        if self.last_activation is not None:
            self._draw_feature_maps(self.last_activation, title=title)
        else:
            self._draw_feature_maps(None, title=(title + ("（无激活/请推理）" if ZH_FONT_OK else " (no activation)")))

    # -------------------------- 可视化：概率/特征/卷积核 --------------------------
    def _draw_prob_bar(self, probs: List[float], top1: int, top1p: float):
        self.prob_canvas.fig.clear()
        ax = self.prob_canvas.fig.add_subplot(111)
        ax.bar(range(10), probs)
        ax.set_xticks(range(10))
        if ZH_FONT_OK:
            ax.set_xlabel("类别"); ax.set_ylabel("概率")
            ax.set_title(f"概率分布（Top-1={top1}, p={top1p:.3f}）")
        else:
            ax.set_xlabel("Class"); ax.set_ylabel("Probability")
            ax.set_title(f"Prob. Distribution (Top-1={top1}, p={top1p:.3f})")
        for i, p in enumerate(probs):
            ax.text(i, min(0.98, p + 0.02), f"{p:.2f}", ha='center', va='bottom', fontsize=8)
        ax.set_ylim(0, 1.0)
        self.prob_canvas.draw()

    def _fill_prob_table(self, probs: List[float]):
        top1 = int(np.argmax(probs))
        for d in range(10):
            self.prob_table.setItem(d, 0, QtWidgets.QTableWidgetItem(str(d)))
            self.prob_table.setItem(d, 1, QtWidgets.QTableWidgetItem(f"{probs[d]:.4f}"))
            bg = QtGui.QColor(255, 255, 200) if d == top1 else QtGui.QColor(255, 255, 255)
            for c in range(2):
                item = self.prob_table.item(d, c)
                if item:
                    item.setBackground(QtGui.QBrush(bg))

    def _draw_feature_maps(self, feat: Optional[torch.Tensor], title: str = ""):
        self.feature_canvas.fig.clear()
        if feat is None or not isinstance(feat, torch.Tensor) or feat.dim() != 3:
            ax = self.feature_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, "暂无激活可展示" if ZH_FONT_OK else "No activation",
                    ha='center', va='center')
            ax.axis('off')
            self.feature_canvas.draw()
            return

        C, H, W = feat.shape
        n = min(16, C)
        idxs = np.linspace(0, C - 1, num=n, dtype=int).tolist()
        rows = cols = 4
        for i, ch in enumerate(idxs):
            ax = self.feature_canvas.fig.add_subplot(rows, cols, i + 1)
            fm = feat[ch].numpy()
            fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-9)
            ax.imshow(fm, cmap='viridis')
            ax.axis('off')
        self.feature_canvas.fig.suptitle(title)
        self.feature_canvas.draw()

    def _clear_kernels_info(self):
        self.kernel_info.setText("—")
        self.kernel_canvas.fig.clear()
        ax = self.kernel_canvas.fig.add_subplot(111)
        ax.text(0.5, 0.5, "无卷积层" if ZH_FONT_OK else "No conv layer", ha='center', va='center')
        ax.axis('off')
        self.kernel_canvas.draw()

    def _draw_kernels(self, layer_name: str):
        """可视化卷积核权重：对 Cin 做均值，等距采样 Cout 的16个核，seismic 零中心显示"""
        self.kernel_canvas.fig.clear()
        if self.model is None or not layer_name:
            self._clear_kernels_info()
            return
        mod = dict(self.model.named_modules()).get(layer_name, None)
        if not isinstance(mod, nn.Conv2d) or getattr(mod, 'weight', None) is None:
            self._clear_kernels_info()
            return

        with torch.no_grad():
            w = mod.weight.detach().cpu().numpy()  # [Cout, Cin, kH, kW]
        Cout, Cin, kH, kW = w.shape
        self.kernel_info.setText(
            f"{layer_name}  |  shape: [{Cout}, {Cin}, {kH}, {kW}]   stride={tuple(mod.stride)}   padding={tuple(mod.padding)}"
        )

        # 聚合到单通道核以便显示：对 Cin 取均值（也可改为 np.sum(np.abs(w), axis=1)）
        kernels = w.mean(axis=1)  # [Cout, kH, kW]

        n = min(16, Cout)
        idxs = np.linspace(0, Cout - 1, num=n, dtype=int).tolist()
        rows = cols = 4
        for i, oc in enumerate(idxs):
            ax = self.kernel_canvas.fig.add_subplot(rows, cols, i + 1)
            k = kernels[oc]
            # 对称零中心，突出正负
            vmax = float(np.max(np.abs(k))) + 1e-9
            ax.imshow(k, cmap='seismic', vmin=-vmax, vmax=vmax)
            ax.set_title(f"{oc}", fontsize=9)
            ax.axis('off')

        title = "卷积核权重（等距采样16个）" if ZH_FONT_OK else "Conv Kernels (sample 16)"
        self.kernel_canvas.fig.suptitle(title)
        self.kernel_canvas.draw()

# -------------------------- 入口 --------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
