from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

from cb_color_correct.filters import FilterPreset, presets
from cb_color_correct.image_ops import FilterParams, process_rgb8
from cb_color_correct.lut import CubeParseError, load_cube


def pil_to_rgb8(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def rgb8_to_qimage(rgb8: np.ndarray) -> QtGui.QImage:
    if rgb8.ndim != 3 or rgb8.shape[2] != 3 or rgb8.dtype != np.uint8:
        raise ValueError("Expected HxWx3 uint8 RGB array")
    h, w, _ = rgb8.shape
    bytes_per_line = 3 * w
    # Detach from Python/NumPy buffer lifecycle (QImage may otherwise reference freed memory).
    qimg = QtGui.QImage(rgb8.tobytes(), w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
    return qimg.copy()


@dataclass
class LoadedImage:
    path: Path
    original_rgb8: np.ndarray
    preview_rgb8: np.ndarray


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CB Color Correct")

        self._loaded: LoadedImage | None = None
        self._current_params = FilterParams()
        self._strength = 1.0
        self._base_pixmap: QtGui.QPixmap | None = None

        self._presets = presets()
        self._category_order = [
            "General",
            "Film / Chemical",
            "Lo-Fi",
            "Wild",
            "Black & White",
            "LUTs",
        ]

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)

        layout = QtWidgets.QHBoxLayout(root)

        # Left controls
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.load_btn = QtWidgets.QPushButton("Load Image…")
        self.load_lut_btn = QtWidgets.QPushButton("Load LUT…")
        self.save_btn = QtWidgets.QPushButton("Save As…")
        self.save_btn.setEnabled(False)

        left_layout.addWidget(self.load_btn)
        left_layout.addWidget(self.load_lut_btn)
        left_layout.addWidget(self.save_btn)

        self.preset_tree = QtWidgets.QTreeWidget()
        self.preset_tree.setMinimumWidth(260)
        self.preset_tree.setHeaderHidden(True)
        self.preset_tree.setRootIsDecorated(True)
        self._rebuild_preset_tree(select_name="None")

        left_layout.addWidget(QtWidgets.QLabel("Presets"))
        left_layout.addWidget(self.preset_tree, 1)

        self.strength_label = QtWidgets.QLabel("Strength: 100%")
        self.strength_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(100)

        left_layout.addWidget(self.strength_label)
        left_layout.addWidget(self.strength_slider)

        self.reset_btn = QtWidgets.QPushButton("Reset")
        left_layout.addWidget(self.reset_btn)

        layout.addWidget(left)

        # Right preview
        self.image_label = QtWidgets.QLabel("Load an image to begin")
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 480)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.image_label)
        layout.addWidget(self.scroll, 1)

        # Wire up
        self.load_btn.clicked.connect(self._on_load)
        self.load_lut_btn.clicked.connect(self._on_load_lut)
        self.save_btn.clicked.connect(self._on_save)
        self.reset_btn.clicked.connect(self._on_reset)
        self.preset_tree.currentItemChanged.connect(self._on_preset_item_changed)
        self.strength_slider.valueChanged.connect(self._on_strength_change)

        self._apply_current()

    def _on_load(self) -> None:
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp);;All Files (*)",
        )
        if not fn:
            return

        path = Path(fn)
        pil_img = Image.open(path)

        # Keep full-res original; for preview, cap size for interactive speed
        original_rgb8 = pil_to_rgb8(pil_img)

        preview_pil = pil_img.convert("RGB")
        preview_pil.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        preview_rgb8 = pil_to_rgb8(preview_pil)

        self._loaded = LoadedImage(path=path, original_rgb8=original_rgb8, preview_rgb8=preview_rgb8)
        self.save_btn.setEnabled(True)
        self._apply_current()

    def _on_save(self) -> None:
        if not self._loaded:
            return

        default_name = self._loaded.path.with_name(self._loaded.path.stem + "_filtered" + self._loaded.path.suffix)
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Image As",
            str(default_name),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff);;WEBP (*.webp);;BMP (*.bmp)",
        )
        if not fn:
            return

        out_path = Path(fn)
        rgb8 = process_rgb8(self._loaded.original_rgb8, self._current_params, self._strength)
        Image.fromarray(rgb8, mode="RGB").save(out_path)

    def _on_load_lut(self) -> None:
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open LUT (.cube)",
            str(Path.home()),
            "Cube LUT (*.cube);;All Files (*)",
        )
        if not fn:
            return

        try:
            lut = load_cube(fn)
        except (CubeParseError, OSError) as e:
            QtWidgets.QMessageBox.critical(self, "Failed to load LUT", str(e))
            return

        name = f"LUT: {Path(fn).stem}"
        preset = FilterPreset(name=name, params=FilterParams(lut=lut), category="LUTs")

        existing_index = next((i for i, p in enumerate(self._presets) if p.name == name), None)
        if existing_index is None:
            self._presets.append(preset)
        else:
            self._presets[existing_index] = preset
        self._rebuild_preset_tree(select_name=name)

    def _on_reset(self) -> None:
        self._select_preset_by_name("None")
        self.strength_slider.setValue(100)

    def _on_preset_item_changed(
        self,
        current: QtWidgets.QTreeWidgetItem | None,
        previous: QtWidgets.QTreeWidgetItem | None,
    ) -> None:
        if not current:
            return
        idx = current.data(0, int(QtCore.Qt.ItemDataRole.UserRole))
        if idx is None:
            return
        try:
            i = int(idx)
        except (TypeError, ValueError):
            return
        if i < 0 or i >= len(self._presets):
            return
        self._current_params = self._presets[i].params
        self._apply_current()

    def _select_preset_by_name(self, name: str) -> None:
        matches = [i for i, p in enumerate(self._presets) if p.name == name]
        if not matches:
            return
        target_idx = matches[0]

        it = QtWidgets.QTreeWidgetItemIterator(self.preset_tree)
        while it.value():
            item = it.value()
            idx = item.data(0, int(QtCore.Qt.ItemDataRole.UserRole))
            if idx is not None and int(idx) == target_idx:
                parent = item.parent()
                if parent:
                    parent.setExpanded(True)
                self.preset_tree.setCurrentItem(item)
                self.preset_tree.scrollToItem(item)
                return
            it += 1

    def _rebuild_preset_tree(self, select_name: str | None = None) -> None:
        self.preset_tree.clear()

        # Group presets by category
        cats: dict[str, list[tuple[int, FilterPreset]]] = {}
        for i, p in enumerate(self._presets):
            cats.setdefault(p.category, []).append((i, p))

        def add_category(cat_name: str) -> None:
            items = cats.get(cat_name)
            if not items:
                return

            cat_item = QtWidgets.QTreeWidgetItem([cat_name])
            cat_item.setData(0, int(QtCore.Qt.ItemDataRole.UserRole), None)
            cat_item.setFlags(cat_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
            self.preset_tree.addTopLevelItem(cat_item)

            # Expand a couple by default
            cat_item.setExpanded(cat_name in ("General", "Film / Chemical"))

            for idx, preset in items:
                child = QtWidgets.QTreeWidgetItem([preset.name])
                child.setData(0, int(QtCore.Qt.ItemDataRole.UserRole), idx)
                cat_item.addChild(child)

        for cat in self._category_order:
            add_category(cat)

        for cat in sorted(c for c in cats.keys() if c not in self._category_order):
            add_category(cat)

        self._select_preset_by_name(select_name or "None")

    def _on_strength_change(self, value: int) -> None:
        self._strength = float(value) / 100.0
        self.strength_label.setText(f"Strength: {value}%")
        self._apply_current()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        # Re-fit current pixmap to viewport
        self._refit_pixmap()

    def _apply_current(self) -> None:
        if not self._loaded:
            self.image_label.setText("Load an image to begin")
            self.image_label.setPixmap(QtGui.QPixmap())
            self._base_pixmap = None
            return

        rgb8 = process_rgb8(self._loaded.preview_rgb8, self._current_params, self._strength)
        qimg = rgb8_to_qimage(rgb8)
        self._base_pixmap = QtGui.QPixmap.fromImage(qimg)
        self._refit_pixmap()

    def _refit_pixmap(self) -> None:
        if not self._base_pixmap or self._base_pixmap.isNull():
            return
        # Fit to scroll viewport width while keeping aspect ratio
        viewport = self.scroll.viewport().size()
        if viewport.width() <= 10 or viewport.height() <= 10:
            return
        scaled = self._base_pixmap.scaled(
            viewport,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1200, 800)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
