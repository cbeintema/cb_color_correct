from __future__ import annotations

import sys
from dataclasses import dataclass
from dataclasses import replace
import json
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

from cb_color_correct.filters import FilterPreset, presets
from cb_color_correct.image_ops import FilterParams, process_rgb8_stack
from cb_color_correct.lut import CubeParseError, load_cube
from cb_color_correct.curve_editor import CurveEditor
from cb_color_correct.theme import apply_ableton_theme


class _RenderSignals(QtCore.QObject):
    finished = QtCore.Signal(int, object)  # generation, rgb8 ndarray
    failed = QtCore.Signal(int, str)  # generation, error text


class _RenderTask(QtCore.QRunnable):
    def __init__(
        self,
        generation: int,
        preview_rgb8: np.ndarray,
        base_params: FilterParams,
        adjust_params: FilterParams,
        strength: float,
    ) -> None:
        super().__init__()
        self.generation = generation
        self.preview_rgb8 = preview_rgb8
        self.base_params = base_params
        self.adjust_params = adjust_params
        self.strength = strength
        self.signals = _RenderSignals()

    def run(self) -> None:
        try:
            rgb8 = process_rgb8_stack(self.preview_rgb8, [self.base_params, self.adjust_params], self.strength)
            self.signals.finished.emit(self.generation, rgb8)
        except Exception:
            self.signals.failed.emit(self.generation, traceback.format_exc())


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
        self._base_params = FilterParams()
        self._adjust_params = FilterParams()
        self._strength = 1.0
        self._base_pixmap: QtGui.QPixmap | None = None

        self._apply_timer = QtCore.QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.timeout.connect(self._apply_current)

        self._thread_pool = QtCore.QThreadPool.globalInstance()
        # Single worker keeps UI consistent (latest result wins anyway).
        self._thread_pool.setMaxThreadCount(1)
        self._render_generation = 0

        self._presets = presets()
        self._category_order = [
            "General",
            "Film / Chemical",
            "Lo-Fi",
            "Wild",
            "Black & White",
            "User",
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
        self.load_preset_btn = QtWidgets.QPushButton("Load Preset…")
        self.save_preset_btn = QtWidgets.QPushButton("Save Preset…")
        self.save_btn = QtWidgets.QPushButton("Save As…")
        self.save_btn.setEnabled(False)

        left_layout.addWidget(self.load_btn)
        left_layout.addWidget(self.load_lut_btn)
        left_layout.addWidget(self.load_preset_btn)
        left_layout.addWidget(self.save_preset_btn)
        left_layout.addWidget(self.save_btn)

        # User presets folder
        self._settings = QtCore.QSettings("CB", "CB Color Correct")
        self.user_presets_dir = str(self._settings.value("userPresetsDir", ""))

        presets_dir_row = QtWidgets.QWidget()
        presets_dir_layout = QtWidgets.QHBoxLayout(presets_dir_row)
        presets_dir_layout.setContentsMargins(0, 0, 0, 0)
        presets_dir_layout.setSpacing(6)

        self.user_presets_edit = QtWidgets.QLineEdit()
        self.user_presets_edit.setPlaceholderText("User presets folder (optional)")
        self.user_presets_edit.setText(self.user_presets_dir)

        self.user_presets_browse_btn = QtWidgets.QToolButton()
        self.user_presets_browse_btn.setText("Browse")

        presets_dir_layout.addWidget(self.user_presets_edit, 1)
        presets_dir_layout.addWidget(self.user_presets_browse_btn)
        left_layout.addWidget(presets_dir_row)

        self.preset_tree = QtWidgets.QTreeWidget()
        self.preset_tree.setMinimumWidth(260)
        self.preset_tree.setHeaderHidden(True)
        self.preset_tree.setRootIsDecorated(True)
        self.preset_tree.setUniformRowHeights(True)
        self.preset_tree.setWordWrap(False)
        self.preset_tree.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self._rebuild_preset_tree(select_name="None")

        left_layout.addWidget(QtWidgets.QLabel("Presets"))
        left_layout.addWidget(self.preset_tree, 1)

        self.reset_btn = QtWidgets.QPushButton("Reset")
        left_layout.addWidget(self.reset_btn)

        layout.addWidget(left)

        # Right side: preview + adjustments sidebar
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QHBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        self.image_label = QtWidgets.QLabel("Load an image to begin")
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 480)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.image_label)
        right_layout.addWidget(self.scroll, 1)

        # Collapsible adjustments sidebar
        self.sidebar_container = QtWidgets.QWidget()
        self.sidebar_container.setObjectName("SidebarContainer")
        container_layout = QtWidgets.QVBoxLayout(self.sidebar_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(8)

        self.sidebar_toggle = QtWidgets.QToolButton()
        self.sidebar_toggle.setCheckable(True)
        self.sidebar_toggle.setChecked(False)  # collapsed by default
        self.sidebar_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.sidebar_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.sidebar_toggle.setText("Adjustments")
        container_layout.addWidget(self.sidebar_toggle)

        self.adjust_sidebar = QtWidgets.QWidget()
        self.adjust_sidebar.setObjectName("AdjustSidebar")
        sidebar_layout = QtWidgets.QVBoxLayout(self.adjust_sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(10)

        self.strength_label = QtWidgets.QLabel("Strength: 100%")
        self.strength_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(100)
        sidebar_layout.addWidget(self.strength_label)
        sidebar_layout.addWidget(self.strength_slider)

        self._adjust_scroll = QtWidgets.QScrollArea()
        self._adjust_scroll.setWidgetResizable(True)
        self._adjust_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        adjust_root = QtWidgets.QWidget()
        self._adjust_scroll.setWidget(adjust_root)
        self._adjust_layout = QtWidgets.QVBoxLayout(adjust_root)
        self._adjust_layout.setContentsMargins(0, 0, 0, 0)
        self._adjust_layout.setSpacing(8)
        self._build_adjustment_widgets()
        sidebar_layout.addWidget(self._adjust_scroll, 1)

        container_layout.addWidget(self.adjust_sidebar, 1)
        right_layout.addWidget(self.sidebar_container)

        self._set_adjustments_visible(False)

        layout.addWidget(right, 1)

        # Wire up
        self.load_btn.clicked.connect(self._on_load)
        self.load_lut_btn.clicked.connect(self._on_load_lut)
        self.load_preset_btn.clicked.connect(self._on_load_preset)
        self.save_preset_btn.clicked.connect(self._on_save_preset)
        self.save_btn.clicked.connect(self._on_save)
        self.reset_btn.clicked.connect(self._on_reset)
        self.preset_tree.currentItemChanged.connect(self._on_preset_item_changed)
        self.strength_slider.valueChanged.connect(self._on_strength_change)
        self.sidebar_toggle.toggled.connect(self._set_adjustments_visible)

        self.user_presets_browse_btn.clicked.connect(self._on_browse_user_presets_dir)
        self.user_presets_edit.editingFinished.connect(self._on_user_presets_dir_edited)

        self._apply_current()

        # Load user presets from configured folder on startup.
        self._reload_user_presets_from_folder()

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
        rgb8 = process_rgb8_stack(self._loaded.original_rgb8, [self._base_params, self._adjust_params], self._strength)
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

    def _on_save_preset(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "Preset Name", "Name:")
        if not ok or not name.strip():
            return
        name = name.strip()

        default_dir = Path(self.user_presets_dir) if self.user_presets_dir else Path.home()
        if not default_dir.exists():
            default_dir = Path.home()

        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Preset",
            str(default_dir / f"{name}.json"),
            "Preset JSON (*.json)",
        )
        if not fn:
            return

        payload = {
            "name": name,
            "category": "User",
            "base": self._filterparams_to_dict(self._base_params),
            "adjust": self._filterparams_to_dict(self._adjust_params),
        }
        Path(fn).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        # Also add it to the in-app list for quick use.
        combined = self._combine_params(self._base_params, self._adjust_params)
        preset = FilterPreset(name=name, params=combined, category="User")
        existing_index = next((i for i, p in enumerate(self._presets) if p.name == name and p.category == "User"), None)
        if existing_index is None:
            self._presets.append(preset)
        else:
            self._presets[existing_index] = preset
        self._rebuild_preset_tree(select_name=name)

    def _on_load_preset(self) -> None:
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Preset",
            str(Path.home()),
            "Preset JSON (*.json);;All Files (*)",
        )
        if not fn:
            return

        try:
            payload = json.loads(Path(fn).read_text(encoding="utf-8"))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Failed to load preset", str(e))
            return

        try:
            name = str(payload.get("name") or Path(fn).stem)
            category = str(payload.get("category") or "User")
            base = self._filterparams_from_dict(payload.get("base") or {})
            adjust = self._filterparams_from_dict(payload.get("adjust") or {})
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Invalid preset", str(e))
            return

        self._base_params = base
        self._adjust_params = adjust
        self._sync_adjustment_widgets_from_state()

        combined = self._combine_params(base, adjust)
        preset = FilterPreset(name=name, params=combined, category=category)
        self._presets.append(preset)
        if category not in self._category_order:
            self._category_order.insert(-1, category)
        self._rebuild_preset_tree(select_name=name)

    def _on_browse_user_presets_dir(self) -> None:
        start = self.user_presets_dir or str(Path.home())
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select User Presets Folder", start)
        if not folder:
            return
        self.user_presets_edit.setText(folder)
        self._set_user_presets_dir(folder)

    def _on_user_presets_dir_edited(self) -> None:
        self._set_user_presets_dir(self.user_presets_edit.text().strip())

    def _set_user_presets_dir(self, folder: str) -> None:
        self.user_presets_dir = folder
        self._settings.setValue("userPresetsDir", folder)
        self._reload_user_presets_from_folder()

    def _reload_user_presets_from_folder(self) -> None:
        # Remove previously folder-loaded user presets (keep in-session User presets).
        self._presets = [p for p in self._presets if not (p.category == "User" and p.source_path)]

        if not self.user_presets_dir:
            self._rebuild_preset_tree(select_name="None")
            return

        folder = Path(self.user_presets_dir)
        if not folder.exists() or not folder.is_dir():
            self._rebuild_preset_tree(select_name="None")
            return

        for path in sorted(folder.glob("*.json")):
            preset = self._load_user_preset_file(path)
            if preset is None:
                continue
            self._presets.append(preset)

        self._rebuild_preset_tree(select_name=None)

    def _load_user_preset_file(self, path: Path) -> FilterPreset | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

        # Supports our save format {name, base, adjust} and a simpler {name, params}.
        name = str(payload.get("name") or path.stem)
        category = str(payload.get("category") or "User")
        if category != "User":
            category = "User"

        if "base" in payload or "adjust" in payload:
            base = self._filterparams_from_dict(payload.get("base") or {})
            adjust = self._filterparams_from_dict(payload.get("adjust") or {})
            combined = self._combine_params(base, adjust)
        else:
            combined = self._filterparams_from_dict(payload.get("params") or payload)

        return FilterPreset(name=name, params=combined, category="User", source_path=str(path))

    def _on_reset(self) -> None:
        self._select_preset_by_name("None")
        self.strength_slider.setValue(100)
        self._adjust_params = FilterParams()
        self._sync_adjustment_widgets_from_state()
        self._schedule_apply()

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
        self._base_params = self._presets[i].params
        self._schedule_apply()

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
        self._schedule_apply()

    def _schedule_apply(self) -> None:
        # Debounce rapid slider changes.
        self._apply_timer.start(25)

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

        self._render_generation += 1
        gen = self._render_generation

        task = _RenderTask(
            generation=gen,
            preview_rgb8=self._loaded.preview_rgb8,
            base_params=self._base_params,
            adjust_params=self._adjust_params,
            strength=self._strength,
        )
        task.signals.finished.connect(self._on_render_finished)
        task.signals.failed.connect(self._on_render_failed)
        self._thread_pool.start(task)

    def _on_render_finished(self, generation: int, rgb8: object) -> None:
        if generation != self._render_generation:
            return
        arr = rgb8  # ndarray
        qimg = rgb8_to_qimage(arr)
        self._base_pixmap = QtGui.QPixmap.fromImage(qimg)
        self._refit_pixmap()

    def _on_render_failed(self, generation: int, err: str) -> None:
        if generation != self._render_generation:
            return
        # Keep it quiet in normal operation; show a dialog only for current render failure.
        QtWidgets.QMessageBox.critical(self, "Render Error", err)

    def _build_adjustment_widgets(self) -> None:
        # Tone group
        self.tone_group = QtWidgets.QGroupBox("Tone")
        self.tone_group.setCheckable(True)
        self.tone_group.setChecked(True)
        tone_form = QtWidgets.QFormLayout(self.tone_group)
        tone_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        tone_form.setContentsMargins(6, 6, 6, 6)
        tone_form.setHorizontalSpacing(8)
        tone_form.setVerticalSpacing(4)

        self.exposure_slider, self.exposure_value = self._make_slider(-200, 200, 0, suffix=" st")
        self.brightness_slider, self.brightness_value = self._make_slider(-20, 20, 0, suffix="")
        self.contrast_slider, self.contrast_value = self._make_slider(-50, 50, 0, suffix="")
        self.blacks_slider, self.blacks_value = self._make_slider(-100, 100, 0, suffix="")
        self.whites_slider, self.whites_value = self._make_slider(-100, 100, 0, suffix="")

        tone_form.addRow("Exposure", self._hbox(self.exposure_slider, self.exposure_value))
        tone_form.addRow("Brightness", self._hbox(self.brightness_slider, self.brightness_value))
        tone_form.addRow("Contrast", self._hbox(self.contrast_slider, self.contrast_value))
        tone_form.addRow("Blacks", self._hbox(self.blacks_slider, self.blacks_value))
        tone_form.addRow("Whites", self._hbox(self.whites_slider, self.whites_value))

        # Color group
        self.color_group = QtWidgets.QGroupBox("Color")
        self.color_group.setCheckable(True)
        self.color_group.setChecked(True)
        color_form = QtWidgets.QFormLayout(self.color_group)
        color_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        color_form.setContentsMargins(6, 6, 6, 6)
        color_form.setHorizontalSpacing(8)
        color_form.setVerticalSpacing(4)

        self.hue_slider, self.hue_value = self._make_slider(-180, 180, 0, suffix="°")
        self.sat_slider, self.sat_value = self._make_slider(-100, 100, 0, suffix="")
        self.vib_slider, self.vib_value = self._make_slider(-100, 100, 0, suffix="")

        color_form.addRow("Hue", self._hbox(self.hue_slider, self.hue_value))
        color_form.addRow("Saturation", self._hbox(self.sat_slider, self.sat_value))
        color_form.addRow("Vibrance", self._hbox(self.vib_slider, self.vib_value))

        # WB group
        self.wb_group = QtWidgets.QGroupBox("WB")
        self.wb_group.setCheckable(True)
        self.wb_group.setChecked(False)
        wb_form = QtWidgets.QFormLayout(self.wb_group)
        wb_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        wb_form.setContentsMargins(6, 6, 6, 6)
        wb_form.setHorizontalSpacing(8)
        wb_form.setVerticalSpacing(4)

        self.temp_slider, self.temp_value = self._make_slider(-100, 100, 0, suffix="")
        self.tint_slider, self.tint_value = self._make_slider(-100, 100, 0, suffix="")
        wb_form.addRow("Temp", self._hbox(self.temp_slider, self.temp_value))
        wb_form.addRow("Tint", self._hbox(self.tint_slider, self.tint_value))

        # Shadows / Highlights group
        self.sh_group = QtWidgets.QGroupBox("Shadows / Highlights")
        self.sh_group.setCheckable(True)
        self.sh_group.setChecked(False)
        sh_form = QtWidgets.QFormLayout(self.sh_group)
        sh_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        sh_form.setContentsMargins(6, 6, 6, 6)
        sh_form.setHorizontalSpacing(8)
        sh_form.setVerticalSpacing(4)

        self.shadows_slider, self.shadows_value = self._make_slider(-100, 100, 0, suffix="")
        self.highlights_slider, self.highlights_value = self._make_slider(-100, 100, 0, suffix="")
        sh_form.addRow("Shadows", self._hbox(self.shadows_slider, self.shadows_value))
        sh_form.addRow("Highlights", self._hbox(self.highlights_slider, self.highlights_value))

        # Split Tone group
        self.split_group = QtWidgets.QGroupBox("Split Tone")
        self.split_group.setCheckable(True)
        self.split_group.setChecked(False)
        split_form = QtWidgets.QFormLayout(self.split_group)
        split_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        split_form.setContentsMargins(6, 6, 6, 6)
        split_form.setHorizontalSpacing(8)
        split_form.setVerticalSpacing(4)

        self.split_shadows_btn = QtWidgets.QPushButton(" ")
        self.split_shadows_btn.setFixedHeight(22)
        self.split_highlights_btn = QtWidgets.QPushButton(" ")
        self.split_highlights_btn.setFixedHeight(22)

        self.split_amount_slider, self.split_amount_value = self._make_slider(0, 100, 0, suffix="")
        self.split_balance_slider, self.split_balance_value = self._make_slider(-100, 100, 0, suffix="")

        split_form.addRow("Shadows", self.split_shadows_btn)
        split_form.addRow("Highlights", self.split_highlights_btn)
        split_form.addRow("Amount", self._hbox(self.split_amount_slider, self.split_amount_value))
        split_form.addRow("Balance", self._hbox(self.split_balance_slider, self.split_balance_value))

        # Effects group
        self.effects_group = QtWidgets.QGroupBox("Effects")
        self.effects_group.setCheckable(True)
        self.effects_group.setChecked(False)
        fx_form = QtWidgets.QFormLayout(self.effects_group)
        fx_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        fx_form.setContentsMargins(6, 6, 6, 6)
        fx_form.setHorizontalSpacing(8)
        fx_form.setVerticalSpacing(4)

        self.clarity_slider, self.clarity_value = self._make_slider(-100, 100, 0, suffix="")
        self.dehaze_slider, self.dehaze_value = self._make_slider(-100, 100, 0, suffix="")
        self.vignette_slider, self.vignette_value = self._make_slider(-100, 100, 0, suffix="")
        self.vignette_mid_slider, self.vignette_mid_value = self._make_slider(0, 100, 50, suffix="")

        fx_form.addRow("Clarity", self._hbox(self.clarity_slider, self.clarity_value))
        fx_form.addRow("Dehaze", self._hbox(self.dehaze_slider, self.dehaze_value))
        fx_form.addRow("Vignette", self._hbox(self.vignette_slider, self.vignette_value))
        fx_form.addRow("Vig Mid", self._hbox(self.vignette_mid_slider, self.vignette_mid_value))

        # Levels group
        self.levels_group = QtWidgets.QGroupBox("Levels")
        self.levels_group.setCheckable(True)
        self.levels_group.setChecked(True)
        levels_form = QtWidgets.QFormLayout(self.levels_group)
        levels_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        levels_form.setContentsMargins(6, 6, 6, 6)
        levels_form.setHorizontalSpacing(8)
        levels_form.setVerticalSpacing(4)

        self.black_slider, self.black_value = self._make_slider(0, 20, 0, suffix="")
        self.white_slider, self.white_value = self._make_slider(80, 100, 100, suffix="")
        self.gamma_slider, self.gamma_value = self._make_slider(50, 150, 100, suffix="")

        levels_form.addRow("Black", self._hbox(self.black_slider, self.black_value))
        levels_form.addRow("White", self._hbox(self.white_slider, self.white_value))
        levels_form.addRow("Gamma", self._hbox(self.gamma_slider, self.gamma_value))

        # Curves group
        self.curves_group = QtWidgets.QGroupBox("Curves")
        self.curves_group.setCheckable(True)
        self.curves_group.setChecked(False)
        curves_layout = QtWidgets.QVBoxLayout(self.curves_group)
        curves_layout.setContentsMargins(6, 6, 6, 6)
        curves_layout.setSpacing(6)

        channel_row = QtWidgets.QHBoxLayout()
        channel_row.setContentsMargins(0, 0, 0, 0)
        channel_row.setSpacing(8)
        channel_row.addWidget(QtWidgets.QLabel("Channel"))
        self.curve_channel = QtWidgets.QComboBox()
        self.curve_channel.addItems(["Master", "Red", "Green", "Blue"])
        self.curve_channel.setCurrentIndex(0)
        channel_row.addWidget(self.curve_channel, 1)
        curves_layout.addLayout(channel_row)

        self.curve_editor = CurveEditor()
        curves_layout.addWidget(self.curve_editor)

        self.reset_adjust_btn = QtWidgets.QPushButton("Reset Adjustments")

        self._adjust_layout.addWidget(self.tone_group)
        self._adjust_layout.addWidget(self.color_group)
        self._adjust_layout.addWidget(self.wb_group)
        self._adjust_layout.addWidget(self.sh_group)
        self._adjust_layout.addWidget(self.split_group)
        self._adjust_layout.addWidget(self.effects_group)
        self._adjust_layout.addWidget(self.levels_group)
        self._adjust_layout.addWidget(self.curves_group)
        self._adjust_layout.addWidget(self.reset_adjust_btn)
        self._adjust_layout.addStretch(1)

        # Signals
        self.reset_adjust_btn.clicked.connect(self._on_reset_adjustments)

        self.tone_group.toggled.connect(self._on_adjust_change)
        self.color_group.toggled.connect(self._on_adjust_change)
        self.wb_group.toggled.connect(self._on_adjust_change)
        self.sh_group.toggled.connect(self._on_adjust_change)
        self.split_group.toggled.connect(self._on_adjust_change)
        self.effects_group.toggled.connect(self._on_adjust_change)
        self.levels_group.toggled.connect(self._on_adjust_change)

        self.exposure_slider.valueChanged.connect(self._on_adjust_change)
        self.brightness_slider.valueChanged.connect(self._on_adjust_change)
        self.contrast_slider.valueChanged.connect(self._on_adjust_change)
        self.blacks_slider.valueChanged.connect(self._on_adjust_change)
        self.whites_slider.valueChanged.connect(self._on_adjust_change)
        self.hue_slider.valueChanged.connect(self._on_adjust_change)
        self.sat_slider.valueChanged.connect(self._on_adjust_change)
        self.vib_slider.valueChanged.connect(self._on_adjust_change)
        self.temp_slider.valueChanged.connect(self._on_adjust_change)
        self.tint_slider.valueChanged.connect(self._on_adjust_change)
        self.shadows_slider.valueChanged.connect(self._on_adjust_change)
        self.highlights_slider.valueChanged.connect(self._on_adjust_change)
        self.split_amount_slider.valueChanged.connect(self._on_adjust_change)
        self.split_balance_slider.valueChanged.connect(self._on_adjust_change)
        self.clarity_slider.valueChanged.connect(self._on_adjust_change)
        self.dehaze_slider.valueChanged.connect(self._on_adjust_change)
        self.vignette_slider.valueChanged.connect(self._on_adjust_change)
        self.vignette_mid_slider.valueChanged.connect(self._on_adjust_change)
        self.black_slider.valueChanged.connect(self._on_adjust_change)
        self.white_slider.valueChanged.connect(self._on_adjust_change)
        self.gamma_slider.valueChanged.connect(self._on_adjust_change)
        self.curves_group.toggled.connect(self._on_adjust_change)
        self.curve_editor.pointsChanged.connect(self._on_curve_changed)
        self.curve_channel.currentIndexChanged.connect(self._on_curve_channel_changed)

        self.split_shadows_btn.clicked.connect(self._pick_split_shadows)
        self.split_highlights_btn.clicked.connect(self._pick_split_highlights)

        self._sync_adjustment_widgets_from_state()

    def _on_curve_changed(self, points: list) -> None:
        # Points are list[(x,y)] in 0..1.
        if not getattr(self, "curves_group", None) or not self.curves_group.isChecked():
            return
        pts = tuple((float(x), float(y)) for x, y in points)
        channel = self.curve_channel.currentText()
        if channel == "Master":
            self._adjust_params = replace(self._adjust_params, curve_points=pts)
        elif channel == "Red":
            self._adjust_params = replace(self._adjust_params, curve_points_r=pts)
        elif channel == "Green":
            self._adjust_params = replace(self._adjust_params, curve_points_g=pts)
        elif channel == "Blue":
            self._adjust_params = replace(self._adjust_params, curve_points_b=pts)
        self._schedule_apply()

    def _on_curve_channel_changed(self) -> None:
        # Load the selected channel's curve into the editor.
        if not getattr(self, "curve_editor", None):
            return
        self._sync_curve_editor_from_state()

    def _on_reset_adjustments(self) -> None:
        self._adjust_params = FilterParams()
        self._sync_adjustment_widgets_from_state()
        self._schedule_apply()

    def _on_adjust_change(self) -> None:
        # Read widgets -> update adjustment params.
        exposure = self.exposure_slider.value() / 100.0
        brightness = self.brightness_slider.value() / 100.0
        contrast = self.contrast_slider.value() / 100.0
        blacks = self.blacks_slider.value() / 100.0
        whites = self.whites_slider.value() / 100.0
        hue = float(self.hue_slider.value())
        sat = self.sat_slider.value() / 100.0
        vib = self.vib_slider.value() / 100.0

        temperature = self.temp_slider.value() / 100.0
        tint = self.tint_slider.value() / 100.0
        shadows = self.shadows_slider.value() / 100.0
        highlights = self.highlights_slider.value() / 100.0

        split_amount = self.split_amount_slider.value() / 100.0
        split_balance = self.split_balance_slider.value() / 100.0

        clarity = self.clarity_slider.value() / 100.0
        dehaze = self.dehaze_slider.value() / 100.0
        vignette = self.vignette_slider.value() / 100.0
        vignette_mid = self.vignette_mid_slider.value() / 100.0

        black = self.black_slider.value() / 100.0
        white = self.white_slider.value() / 100.0
        gamma = self.gamma_slider.value() / 100.0

        # Update value labels
        self.exposure_value.setText(f"{exposure:.2f} st")
        self.brightness_value.setText(f"{brightness:.2f}")
        self.contrast_value.setText(f"{contrast:.2f}")
        self.blacks_value.setText(f"{blacks:.2f}")
        self.whites_value.setText(f"{whites:.2f}")
        self.hue_value.setText(f"{int(hue)}°")
        self.sat_value.setText(f"{sat:.2f}")
        self.vib_value.setText(f"{vib:.2f}")
        self.temp_value.setText(f"{temperature:.2f}")
        self.tint_value.setText(f"{tint:.2f}")
        self.shadows_value.setText(f"{shadows:.2f}")
        self.highlights_value.setText(f"{highlights:.2f}")
        self.split_amount_value.setText(f"{split_amount:.2f}")
        self.split_balance_value.setText(f"{split_balance:.2f}")
        self.clarity_value.setText(f"{clarity:.2f}")
        self.dehaze_value.setText(f"{dehaze:.2f}")
        self.vignette_value.setText(f"{vignette:.2f}")
        self.vignette_mid_value.setText(f"{vignette_mid:.2f}")
        self.black_value.setText(f"{black:.2f}")
        self.white_value.setText(f"{white:.2f}")
        self.gamma_value.setText(f"{gamma:.2f}")

        # If group disabled, ignore its settings.
        if not self.tone_group.isChecked():
            exposure = 0.0
            brightness = 0.0
            contrast = 0.0
            blacks = 0.0
            whites = 0.0
        if not self.color_group.isChecked():
            hue = 0.0
            sat = 0.0
            vib = 0.0
        if not self.wb_group.isChecked():
            temperature = 0.0
            tint = 0.0
        if not self.sh_group.isChecked():
            shadows = 0.0
            highlights = 0.0
        if not self.split_group.isChecked():
            split_amount = 0.0
            split_balance = 0.0
        if not self.effects_group.isChecked():
            clarity = 0.0
            dehaze = 0.0
            vignette = 0.0
            vignette_mid = 0.5
        if not self.levels_group.isChecked():
            black = 0.0
            white = 1.0
            gamma = 1.0

        curve_points = None
        curve_r = None
        curve_g = None
        curve_b = None
        if getattr(self, "curves_group", None) and self.curves_group.isChecked():
            # Keep existing stored curves; the editor writes the selected channel via _on_curve_changed.
            curve_points = self._adjust_params.curve_points
            curve_r = self._adjust_params.curve_points_r
            curve_g = self._adjust_params.curve_points_g
            curve_b = self._adjust_params.curve_points_b
        else:
            # Disabled -> clear curves
            curve_points = None
            curve_r = None
            curve_g = None
            curve_b = None

        self._adjust_params = replace(
            self._adjust_params,
            exposure=exposure,
            brightness=brightness,
            contrast=contrast,
            blacks=blacks,
            whites=whites,
            hue_degrees=hue,
            saturation=sat,
            vibrance=vib,
            temperature=temperature,
            tint=tint,
            shadows=shadows,
            highlights=highlights,
            split_balance=split_balance,
            split_amount=split_amount,
            clarity=clarity,
            dehaze=dehaze,
            vignette=vignette,
            vignette_midpoint=vignette_mid,
            levels_black=black,
            levels_white=white,
            levels_gamma=gamma,
            curve_points=curve_points,
            curve_points_r=curve_r,
            curve_points_g=curve_g,
            curve_points_b=curve_b,
        )
        self._schedule_apply()

    def _sync_adjustment_widgets_from_state(self) -> None:
        # Block signals to avoid feedback loop.
        widgets = [
            self.exposure_slider,
            self.brightness_slider,
            self.contrast_slider,
            self.blacks_slider,
            self.whites_slider,
            self.hue_slider,
            self.sat_slider,
            self.vib_slider,
            self.temp_slider,
            self.tint_slider,
            self.shadows_slider,
            self.highlights_slider,
            self.split_amount_slider,
            self.split_balance_slider,
            self.clarity_slider,
            self.dehaze_slider,
            self.vignette_slider,
            self.vignette_mid_slider,
            self.black_slider,
            self.white_slider,
            self.gamma_slider,
        ]
        for w in widgets:
            w.blockSignals(True)

        self.exposure_slider.setValue(int(round(self._adjust_params.exposure * 100.0)))
        self.brightness_slider.setValue(int(round(self._adjust_params.brightness * 100.0)))
        self.contrast_slider.setValue(int(round(self._adjust_params.contrast * 100.0)))
        self.blacks_slider.setValue(int(round(self._adjust_params.blacks * 100.0)))
        self.whites_slider.setValue(int(round(self._adjust_params.whites * 100.0)))
        self.hue_slider.setValue(int(round(self._adjust_params.hue_degrees)))
        self.sat_slider.setValue(int(round(self._adjust_params.saturation * 100.0)))
        self.vib_slider.setValue(int(round(self._adjust_params.vibrance * 100.0)))

        self.temp_slider.setValue(int(round(self._adjust_params.temperature * 100.0)))
        self.tint_slider.setValue(int(round(self._adjust_params.tint * 100.0)))
        self.shadows_slider.setValue(int(round(self._adjust_params.shadows * 100.0)))
        self.highlights_slider.setValue(int(round(self._adjust_params.highlights * 100.0)))

        self.split_amount_slider.setValue(int(round(self._adjust_params.split_amount * 100.0)))
        self.split_balance_slider.setValue(int(round(self._adjust_params.split_balance * 100.0)))

        self.clarity_slider.setValue(int(round(self._adjust_params.clarity * 100.0)))
        self.dehaze_slider.setValue(int(round(self._adjust_params.dehaze * 100.0)))
        self.vignette_slider.setValue(int(round(self._adjust_params.vignette * 100.0)))
        self.vignette_mid_slider.setValue(int(round(self._adjust_params.vignette_midpoint * 100.0)))

        self.black_slider.setValue(int(round(self._adjust_params.levels_black * 100.0)))
        self.white_slider.setValue(int(round(self._adjust_params.levels_white * 100.0)))
        self.gamma_slider.setValue(int(round(self._adjust_params.levels_gamma * 100.0)))

        # Enable/disable optional groups based on state (important for preset load).
        self.wb_group.setChecked(bool(self._adjust_params.temperature or self._adjust_params.tint))
        self.sh_group.setChecked(bool(self._adjust_params.shadows or self._adjust_params.highlights))
        self.split_group.setChecked(bool(self._adjust_params.split_amount))
        self.effects_group.setChecked(
            bool(self._adjust_params.clarity or self._adjust_params.dehaze or self._adjust_params.vignette)
        )

        self._sync_split_buttons_from_state()

        # Curves
        if getattr(self, "curves_group", None):
            has_any_curve = (
                self._adjust_params.curve_points is not None
                or self._adjust_params.curve_points_r is not None
                or self._adjust_params.curve_points_g is not None
                or self._adjust_params.curve_points_b is not None
            )
            self.curves_group.setChecked(bool(has_any_curve))
            self._sync_curve_editor_from_state()

        for w in widgets:
            w.blockSignals(False)

        self._on_adjust_change()

    def _sync_curve_editor_from_state(self) -> None:
        if not getattr(self, "curve_editor", None):
            return
        channel = self.curve_channel.currentText() if getattr(self, "curve_channel", None) else "Master"
        pts = None
        if channel == "Master":
            pts = self._adjust_params.curve_points
        elif channel == "Red":
            pts = self._adjust_params.curve_points_r
        elif channel == "Green":
            pts = self._adjust_params.curve_points_g
        elif channel == "Blue":
            pts = self._adjust_params.curve_points_b

        if pts is None:
            self.curve_editor.set_identity(emit=False)
        else:
            self.curve_editor.set_points(pts, emit=False)

    def _sync_split_buttons_from_state(self) -> None:
        if not getattr(self, "split_shadows_btn", None):
            return
        sh = self._split_shadows_color()
        hi = self._split_highlights_color()
        self._set_color_button(self.split_shadows_btn, sh)
        self._set_color_button(self.split_highlights_btn, hi)

    def _set_color_button(self, btn: QtWidgets.QPushButton, color: QtGui.QColor) -> None:
        btn.setStyleSheet(
            "QPushButton {"
            f"background-color: {color.name()};"
            "border: 1px solid #3a3a3a;"
            "border-radius: 3px;"
            "}"
        )

    def _split_shadows_color(self) -> QtGui.QColor:
        r, g, b = self._adjust_params.split_shadows
        return QtGui.QColor.fromRgbF(float(r), float(g), float(b))

    def _split_highlights_color(self) -> QtGui.QColor:
        r, g, b = self._adjust_params.split_highlights
        return QtGui.QColor.fromRgbF(float(r), float(g), float(b))

    def _pick_split_shadows(self) -> None:
        c = QtWidgets.QColorDialog.getColor(self._split_shadows_color(), self, "Split Tone - Shadows")
        if not c.isValid():
            return
        rgb = (c.redF(), c.greenF(), c.blueF())
        self._adjust_params = replace(self._adjust_params, split_shadows=rgb)
        self._set_color_button(self.split_shadows_btn, c)
        if not self.split_group.isChecked():
            self.split_group.setChecked(True)
        self._schedule_apply()

    def _pick_split_highlights(self) -> None:
        c = QtWidgets.QColorDialog.getColor(self._split_highlights_color(), self, "Split Tone - Highlights")
        if not c.isValid():
            return
        rgb = (c.redF(), c.greenF(), c.blueF())
        self._adjust_params = replace(self._adjust_params, split_highlights=rgb)
        self._set_color_button(self.split_highlights_btn, c)
        if not self.split_group.isChecked():
            self.split_group.setChecked(True)
        self._schedule_apply()

    def _make_slider(self, min_v: int, max_v: int, value: int, suffix: str = "") -> tuple[QtWidgets.QSlider, QtWidgets.QLabel]:
        s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        s.setRange(min_v, max_v)
        s.setValue(value)
        s.setSingleStep(1)
        lab = QtWidgets.QLabel("0")
        lab.setMinimumWidth(52)
        lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        return s, lab

    def _set_adjustments_visible(self, visible: bool) -> None:
        self.adjust_sidebar.setVisible(visible)
        self.sidebar_toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow if visible else QtCore.Qt.ArrowType.RightArrow)
        # Keep professional/compact: narrow when collapsed, comfortable when expanded
        if visible:
            self.sidebar_container.setMinimumWidth(320)
            self.sidebar_container.setMaximumWidth(600)
        else:
            self.sidebar_container.setMinimumWidth(34)
            self.sidebar_container.setMaximumWidth(34)

    def _hbox(self, slider: QtWidgets.QSlider, label: QtWidgets.QLabel) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addWidget(slider, 1)
        lay.addWidget(label)
        return w

    def _filterparams_to_dict(self, p: FilterParams) -> dict:
        # Note: LUTs are intentionally not serialized (can be re-loaded via .cube).
        return {
            "exposure": p.exposure,
            "brightness": p.brightness,
            "contrast": p.contrast,
            "blacks": p.blacks,
            "whites": p.whites,
            "saturation": p.saturation,
            "vibrance": p.vibrance,
            "hue_degrees": p.hue_degrees,
            "temperature": p.temperature,
            "tint": p.tint,
            "shadows": p.shadows,
            "highlights": p.highlights,
            "clarity": p.clarity,
            "dehaze": p.dehaze,
            "vignette": p.vignette,
            "vignette_midpoint": p.vignette_midpoint,
            "levels_black": p.levels_black,
            "levels_white": p.levels_white,
            "levels_gamma": p.levels_gamma,
            "curve_points": list(p.curve_points) if p.curve_points is not None else None,
            "curve_points_r": list(p.curve_points_r) if p.curve_points_r is not None else None,
            "curve_points_g": list(p.curve_points_g) if p.curve_points_g is not None else None,
            "curve_points_b": list(p.curve_points_b) if p.curve_points_b is not None else None,
            "channel_mul": list(p.channel_mul),
            "split_shadows": list(p.split_shadows),
            "split_highlights": list(p.split_highlights),
            "split_balance": p.split_balance,
            "split_amount": p.split_amount,
        }

    def _filterparams_from_dict(self, d: dict) -> FilterParams:
        curve = d.get("curve_points")
        curve_points = None
        if curve is not None:
            curve_points = tuple((float(x), float(y)) for x, y in curve)

        def read_curve(key: str) -> tuple[tuple[float, float], ...] | None:
            v = d.get(key)
            if v is None:
                return None
            return tuple((float(x), float(y)) for x, y in v)

        curve_r = read_curve("curve_points_r")
        curve_g = read_curve("curve_points_g")
        curve_b = read_curve("curve_points_b")
        return FilterParams(
            exposure=float(d.get("exposure", 0.0)),
            brightness=float(d.get("brightness", 0.0)),
            contrast=float(d.get("contrast", 0.0)),
            blacks=float(d.get("blacks", 0.0)),
            whites=float(d.get("whites", 0.0)),
            saturation=float(d.get("saturation", 0.0)),
            vibrance=float(d.get("vibrance", 0.0)),
            hue_degrees=float(d.get("hue_degrees", 0.0)),
            temperature=float(d.get("temperature", 0.0)),
            tint=float(d.get("tint", 0.0)),
            shadows=float(d.get("shadows", 0.0)),
            highlights=float(d.get("highlights", 0.0)),
            clarity=float(d.get("clarity", 0.0)),
            dehaze=float(d.get("dehaze", 0.0)),
            vignette=float(d.get("vignette", 0.0)),
            vignette_midpoint=float(d.get("vignette_midpoint", 0.5)),
            levels_black=float(d.get("levels_black", 0.0)),
            levels_white=float(d.get("levels_white", 1.0)),
            levels_gamma=float(d.get("levels_gamma", 1.0)),
            curve_points=curve_points,
            curve_points_r=curve_r,
            curve_points_g=curve_g,
            curve_points_b=curve_b,
            channel_mul=tuple(d.get("channel_mul", [1.0, 1.0, 1.0])),
            split_shadows=tuple(d.get("split_shadows", [0.0, 0.0, 0.0])),
            split_highlights=tuple(d.get("split_highlights", [0.0, 0.0, 0.0])),
            split_balance=float(d.get("split_balance", 0.0)),
            split_amount=float(d.get("split_amount", 0.0)),
        )

    def _combine_params(self, base: FilterParams, adjust: FilterParams) -> FilterParams:
        # Combine into a single preset for saving into the tree. This is a simple additive/override blend.
        # For more advanced workflows we can store base+adjust separately.
        return FilterParams(
            exposure=base.exposure + adjust.exposure,
            brightness=base.brightness + adjust.brightness,
            contrast=base.contrast + adjust.contrast,
            blacks=base.blacks + adjust.blacks,
            whites=base.whites + adjust.whites,
            saturation=base.saturation + adjust.saturation,
            vibrance=base.vibrance + adjust.vibrance,
            hue_degrees=base.hue_degrees + adjust.hue_degrees,
            temperature=base.temperature + adjust.temperature,
            tint=base.tint + adjust.tint,
            shadows=base.shadows + adjust.shadows,
            highlights=base.highlights + adjust.highlights,
            clarity=base.clarity + adjust.clarity,
            dehaze=base.dehaze + adjust.dehaze,
            vignette=base.vignette + adjust.vignette,
            vignette_midpoint=adjust.vignette_midpoint if adjust.vignette_midpoint != 0.5 else base.vignette_midpoint,
            levels_black=max(0.0, base.levels_black + adjust.levels_black),
            levels_white=min(1.0, base.levels_white * (adjust.levels_white if adjust.levels_white != 1.0 else 1.0)),
            levels_gamma=base.levels_gamma * (adjust.levels_gamma if adjust.levels_gamma != 1.0 else 1.0),
            curve_points=adjust.curve_points or base.curve_points,
            curve_points_r=adjust.curve_points_r or base.curve_points_r,
            curve_points_g=adjust.curve_points_g or base.curve_points_g,
            curve_points_b=adjust.curve_points_b or base.curve_points_b,
            channel_mul=(
                base.channel_mul[0] * adjust.channel_mul[0],
                base.channel_mul[1] * adjust.channel_mul[1],
                base.channel_mul[2] * adjust.channel_mul[2],
            ),
            split_shadows=adjust.split_shadows if adjust.split_amount else base.split_shadows,
            split_highlights=adjust.split_highlights if adjust.split_amount else base.split_highlights,
            split_balance=base.split_balance + adjust.split_balance,
            split_amount=max(base.split_amount, adjust.split_amount),
            lut=base.lut if base.lut is not None else adjust.lut,
        )

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

    apply_ableton_theme(app)

    w = MainWindow()
    w.resize(1200, 800)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
