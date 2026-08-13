from __future__ import annotations

import sys
from dataclasses import dataclass
from dataclasses import replace
import json
import shlex
import shutil
import tempfile
import traceback
import urllib.request
from typing import Iterable
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

from cb_color_correct.censor import CensorCircle, apply_censor_blur
from cb_color_correct.filters import FilterPreset, presets
from cb_color_correct.image_ops import FilterParams, process_rgb8_stack
from cb_color_correct.image_metadata import save_metadata_free_rgb8
from cb_color_correct.lut import CubeParseError, load_cube
from cb_color_correct.curve_editor import CurveEditor
from cb_color_correct.theme import apply_ableton_theme
from cb_color_correct.upscale import (
    ComfyUILaunchWatcher,
    ComfyUpscaleWorker,
    UpscalePackagePaths,
    UpscaleSettings,
    build_upscale_package_paths,
    create_upscale_zip,
)


DEFAULT_CENSOR_BLUR_RADIUS = 24


def _default_comfyui_launch_settings() -> tuple[str, str, str]:
    comfy_root = Path.home() / "Documents" / "ComfyUI"
    python_path = comfy_root / ".venv" / "Scripts" / "python.exe"
    main_path = Path.home() / "AppData" / "Local" / "Programs" / "ComfyUI" / "resources" / "ComfyUI" / "main.py"
    user_path = comfy_root / "user"
    model_paths = comfy_root / "extra_model_paths.yaml"

    extra_args: list[str] = []
    if user_path.is_dir():
        extra_args.extend(["--user-directory", f'"{user_path}"'])
    if model_paths.is_file():
        extra_args.extend(["--extra-model-paths-config", f'"{model_paths}"'])
    return (
        str(python_path) if python_path.is_file() else "",
        str(main_path) if main_path.is_file() else "",
        " ".join(extra_args),
    )


def validate_packages() -> None:
    """Validate that required and optional packages are available."""
    missing_required = []
    missing_optional = []
    
    # Check required packages
    required = [
        ("numpy", "numpy"),
        ("PIL", "Pillow"),
        ("PySide6", "PySide6"),
    ]
    
    for module_name, package_name in required:
        try:
            __import__(module_name)
        except ImportError:
            missing_required.append(package_name)
    
    # Check optional packages
    optional = [
        ("pilgram2", "pilgram2", "Instagram filters will not be available"),
    ]
    
    for module_name, package_name, description in optional:
        try:
            __import__(module_name)
        except ImportError:
            missing_optional.append((package_name, description))
    
    # Report missing packages
    if missing_required:
        print(f"ERROR: Missing required packages: {', '.join(missing_required)}", file=sys.stderr)
        print("Please run: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)
    
    if missing_optional:
        print("WARNING: Optional packages missing:", file=sys.stderr)
        for package_name, description in missing_optional:
            print(f"  - {package_name}: {description}", file=sys.stderr)
        print("To install optional packages: pip install -r requirements.txt", file=sys.stderr)
        print()


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
        censor_circles: tuple[CensorCircle, ...] = (),
        censor_blur_radius: float = 0.0,
    ) -> None:
        super().__init__()
        self.generation = generation
        self.preview_rgb8 = preview_rgb8
        self.base_params = base_params
        self.adjust_params = adjust_params
        self.strength = strength
        self.censor_circles = tuple(censor_circles)
        self.censor_blur_radius = float(censor_blur_radius)
        self.signals = _RenderSignals()

    def run(self) -> None:
        try:
            rgb8 = process_rgb8_stack(self.preview_rgb8, [self.base_params, self.adjust_params], self.strength)
            if self.censor_circles:
                rgb8 = apply_censor_blur(rgb8, self.censor_circles, self.censor_blur_radius)
            self.signals.finished.emit(self.generation, rgb8)
        except Exception:
            self.signals.failed.emit(self.generation, traceback.format_exc())


class _BatchSignals(QtCore.QObject):
    progress = QtCore.Signal(int, int, str)  # done, total, filename
    finished = QtCore.Signal(int, int)  # ok, failed
    failed = QtCore.Signal(str)  # fatal error


class _BatchTask(QtCore.QRunnable):
    def __init__(
        self,
        files: list[Path],
        output_dir: Path,
        base_params: FilterParams,
        adjust_params: FilterParams,
        strength: float,
    ) -> None:
        super().__init__()
        self.files = files
        self.output_dir = output_dir
        self.base_params = base_params
        self.adjust_params = adjust_params
        self.strength = strength
        self.signals = _BatchSignals()

    def run(self) -> None:
        try:
            from PIL import Image

            ok = 0
            failed = 0
            total = len(self.files)

            self.output_dir.mkdir(parents=True, exist_ok=True)

            for i, path in enumerate(self.files, start=1):
                try:
                    pil_img = Image.open(path)
                    rgb8 = pil_to_rgb8(pil_img)
                    out = process_rgb8_stack(rgb8, [self.base_params, self.adjust_params], self.strength)

                    out_name = path.stem + "_filtered" + path.suffix
                    out_path = self.output_dir / out_name
                    Image.fromarray(out, mode="RGB").save(out_path)
                    ok += 1
                except Exception:
                    failed += 1

                self.signals.progress.emit(i, total, path.name)

            self.signals.finished.emit(ok, failed)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


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


@dataclass(frozen=True)
class _HistoryState:
    base_params: FilterParams
    adjust_params: FilterParams
    strength: float
    censor_circles: tuple[CensorCircle, ...] = ()
    censor_blur_radius: int = DEFAULT_CENSOR_BLUR_RADIUS


class _FixedRowHeightDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, row_height: int, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._row_height = int(row_height)

    def sizeHint(
        self,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> QtCore.QSize:
        size = super().sizeHint(option, index)
        size.setHeight(self._row_height)
        return size


class _PanScrollArea(QtWidgets.QScrollArea):
    pinchZoomRequested = QtCore.Signal(float, object)  # factor, viewport QPoint
    wheelZoomRequested = QtCore.Signal(float, object)  # factor, viewport QPoint

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._space_pan_enabled = False
        self._panning = False
        self._pan_start = QtCore.QPoint(0, 0)
        self._pan_h = 0
        self._pan_v = 0
        self.viewport().installEventFilter(self)

    def set_space_pan_enabled(self, enabled: bool) -> None:
        self._space_pan_enabled = bool(enabled)
        if self._panning:
            return
        if self._space_pan_enabled:
            self.viewport().setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        else:
            self.viewport().unsetCursor()

    def _begin_pan(self, pos: QtCore.QPoint) -> None:
        self._panning = True
        self._pan_start = QtCore.QPoint(pos)
        self._pan_h = self.horizontalScrollBar().value()
        self._pan_v = self.verticalScrollBar().value()
        self.viewport().setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)

    def _end_pan(self) -> None:
        self._panning = False
        if self._space_pan_enabled:
            self.viewport().setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        else:
            self.viewport().unsetCursor()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._begin_pan(event.position().toPoint())
            event.accept()
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._space_pan_enabled:
            self._begin_pan(event.position().toPoint())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._panning:
            delta = event.position().toPoint() - self._pan_start
            self.horizontalScrollBar().setValue(self._pan_h - delta.x())
            self.verticalScrollBar().setValue(self._pan_v - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._panning and event.button() in (QtCore.Qt.MouseButton.MiddleButton, QtCore.Qt.MouseButton.LeftButton):
            self._end_pan()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self.viewport() and event.type() == QtCore.QEvent.Type.NativeGesture:
            ng = event  # QNativeGestureEvent
            if ng.gestureType() == QtCore.Qt.NativeGestureType.ZoomNativeGesture:
                value = float(ng.value())
                factor = max(0.2, min(5.0, 1.0 + value))
                self.pinchZoomRequested.emit(factor, ng.position().toPoint())
                return True
        return super().eventFilter(watched, event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        mods = event.modifiers()
        if mods & (QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.MetaModifier):
            delta = event.angleDelta().y()
            if delta == 0:
                delta = event.pixelDelta().y()
            if delta != 0:
                factor = 1.12 if delta > 0 else (1.0 / 1.12)
                self.wheelZoomRequested.emit(factor, event.position().toPoint())
                event.accept()
                return
        super().wheelEvent(event)


class CensorImageLabel(QtWidgets.QLabel):
    circleCreated = QtCore.Signal(object)

    def __init__(self, text: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._censor_enabled = False
        self._space_pan_active = False
        self._censor_circles: tuple[CensorCircle, ...] = ()
        self._drag_start: QtCore.QPointF | None = None
        self._drag_current: QtCore.QPointF | None = None
        self.setMouseTracking(True)

    def set_censor_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if not enabled:
            self.cancel_censor_drag()
        self._censor_enabled = enabled
        self._set_idle_cursor()
        self.update()

    def set_space_pan_active(self, active: bool) -> None:
        self._space_pan_active = bool(active)
        self._set_idle_cursor()

    def set_censor_circles(self, circles: tuple[CensorCircle, ...]) -> None:
        self._censor_circles = tuple(circles)
        self.update()

    def cancel_censor_drag(self) -> bool:
        if self._drag_start is None:
            return False
        self._drag_start = None
        self._drag_current = None
        self.releaseMouse()
        self._set_idle_cursor()
        self.update()
        return True

    def _set_idle_cursor(self) -> None:
        if not self._censor_enabled:
            self.unsetCursor()
        elif self._space_pan_active:
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(QtCore.Qt.CursorShape.CrossCursor)

    def _circle_rect(self, circle: CensorCircle) -> QtCore.QRectF:
        radius = float(circle.radius) * self.width()
        center = QtCore.QPointF(float(circle.center_x) * self.width(), float(circle.center_y) * self.height())
        return QtCore.QRectF(center.x() - radius, center.y() - radius, radius * 2.0, radius * 2.0)

    def _drag_circle(self) -> CensorCircle | None:
        if self._drag_start is None or self._drag_current is None or self.width() <= 0:
            return None
        dx = self._drag_current.x() - self._drag_start.x()
        dy = self._drag_current.y() - self._drag_start.y()
        distance = float(np.hypot(dx, dy))
        return CensorCircle(
            center_x=max(0.0, min(1.0, self._drag_start.x() / self.width())),
            center_y=max(0.0, min(1.0, self._drag_start.y() / max(1, self.height()))),
            radius=distance / self.width(),
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            if self._censor_enabled:
                self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            event.ignore()
            return
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and self._censor_enabled
            and not self._space_pan_active
            and self.pixmap()
            and not self.pixmap().isNull()
        ):
            self._drag_start = QtCore.QPointF(event.position())
            self._drag_current = QtCore.QPointF(event.position())
            self.grabMouse()
            event.accept()
            self.update()
            return
        event.ignore()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_start is not None:
            self._drag_current = QtCore.QPointF(event.position())
            event.accept()
            self.update()
            return
        event.ignore()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_start is not None and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_current = QtCore.QPointF(event.position())
            circle = self._drag_circle()
            self._drag_start = None
            self._drag_current = None
            self.releaseMouse()
            self._set_idle_cursor()
            event.accept()
            self.update()
            if circle is not None and circle.radius * self.width() >= 4.0:
                self.circleCreated.emit(circle)
            return
        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._set_idle_cursor()
        event.ignore()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        if not self._censor_enabled:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 210, 80, 230), 2))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 210, 80, 45)))
        for circle in self._censor_circles:
            painter.drawEllipse(self._circle_rect(circle))

        drag_circle = self._drag_circle()
        if drag_circle is not None:
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 230), 2, QtCore.Qt.PenStyle.DashLine))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 35)))
            painter.drawEllipse(self._circle_rect(drag_circle))
        painter.end()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CB Color Correct")

        self._loaded: LoadedImage | None = None
        self._base_params = FilterParams()
        self._adjust_params = FilterParams()
        self._strength = 1.0
        self._base_pixmap: QtGui.QPixmap | None = None
        self._original_preview_pixmap: QtGui.QPixmap | None = None
        self._zoom_mode: str = "fit"  # fit | custom
        self._zoom_factor: float = 1.0
        self._space_pan_active = False
        self._split_preview_enabled = False
        self._split_divider_ratio = 0.5
        self._dragging_split_divider = False
        self._censor_enabled = False
        self._censor_circles: tuple[CensorCircle, ...] = ()
        self._censor_blur_radius = DEFAULT_CENSOR_BLUR_RADIUS
        self._undo_stack: list[_HistoryState] = []
        self._redo_stack: list[_HistoryState] = []
        self._history_restoring = False
        self._history_last_state = _HistoryState(self._base_params, self._adjust_params, self._strength)

        self._apply_timer = QtCore.QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.timeout.connect(self._apply_current)

        self._thread_pool = QtCore.QThreadPool.globalInstance()
        # Single worker keeps UI consistent (latest result wins anyway).
        self._thread_pool.setMaxThreadCount(1)
        self._render_generation = 0
        self._upscale_worker: ComfyUpscaleWorker | None = None
        self._comfyui_launch_watcher: ComfyUILaunchWatcher | None = None
        self._upscale_source_temp: Path | None = None
        self._upscale_output_auto = True
        self._upscale_cancel_requested = False
        self._upscale_comfyui_stopping = False
        self._upscale_package_mode = False
        self._upscale_package_censored_temp: Path | None = None
        self._upscale_package_paths: UpscalePackagePaths | None = None

        self._presets = presets()
        self._category_order = [
            "General",
            "Instagram",
            "Film / Chemical",
            "Lo-Fi",
            "Wild",
            "Black & White",
            "User",
            "LUTs",
        ]
        self._favorite_keys: set[str] = set()
        self._preset_tree_expand_state_before_favorites: dict[str, bool] | None = None
        self._slider_width = 270

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
        stored_censor_blur = self._settings.value("censor/blurRadius", DEFAULT_CENSOR_BLUR_RADIUS)
        try:
            self._censor_blur_radius = max(1, min(100, int(str(stored_censor_blur))))
        except (TypeError, ValueError):
            self._censor_blur_radius = DEFAULT_CENSOR_BLUR_RADIUS
        self.user_presets_dir = str(self._settings.value("userPresetsDir", ""))
        self.last_open_image_dir = str(self._settings.value("lastOpenImageDir", ""))
        try:
            raw_favs = str(self._settings.value("favoritePresetKeys", "[]"))
            fav_list = json.loads(raw_favs)
            if isinstance(fav_list, list):
                self._favorite_keys = {str(v) for v in fav_list}
        except Exception:
            self._favorite_keys = set()

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

        # Batch panel (collapsed by default)
        self.batch_group = QtWidgets.QGroupBox("Batch")
        self.batch_group.setCheckable(True)
        self.batch_group.setChecked(False)
        batch_layout = QtWidgets.QVBoxLayout(self.batch_group)
        batch_layout.setContentsMargins(6, 6, 6, 6)
        batch_layout.setSpacing(6)

        self.batch_content = QtWidgets.QWidget()
        bc = QtWidgets.QVBoxLayout(self.batch_content)
        bc.setContentsMargins(0, 0, 0, 0)
        bc.setSpacing(6)

        self.batch_input_dir = str(self._settings.value("batchInputDir", ""))
        self.batch_output_dir = str(self._settings.value("batchOutputDir", ""))

        in_row = QtWidgets.QWidget()
        in_lay = QtWidgets.QHBoxLayout(in_row)
        in_lay.setContentsMargins(0, 0, 0, 0)
        in_lay.setSpacing(6)
        self.batch_in_edit = QtWidgets.QLineEdit(self.batch_input_dir)
        self.batch_in_edit.setPlaceholderText("Batch input folder")
        self.batch_in_browse = QtWidgets.QToolButton()
        self.batch_in_browse.setText("Browse")
        in_lay.addWidget(self.batch_in_edit, 1)
        in_lay.addWidget(self.batch_in_browse)

        out_row = QtWidgets.QWidget()
        out_lay = QtWidgets.QHBoxLayout(out_row)
        out_lay.setContentsMargins(0, 0, 0, 0)
        out_lay.setSpacing(6)
        self.batch_out_edit = QtWidgets.QLineEdit(self.batch_output_dir)
        self.batch_out_edit.setPlaceholderText("Batch output folder")
        self.batch_out_browse = QtWidgets.QToolButton()
        self.batch_out_browse.setText("Browse")
        out_lay.addWidget(self.batch_out_edit, 1)
        out_lay.addWidget(self.batch_out_browse)

        self.batch_run_btn = QtWidgets.QPushButton("Run Batch")
        self.batch_progress = QtWidgets.QProgressBar()
        self.batch_progress.setRange(0, 100)
        self.batch_progress.setValue(0)
        self.batch_progress.setTextVisible(True)
        self.batch_progress.setFormat("Idle")

        bc.addWidget(in_row)
        bc.addWidget(out_row)
        bc.addWidget(self.batch_run_btn)
        bc.addWidget(self.batch_progress)

        batch_layout.addWidget(self.batch_content)
        left_layout.addWidget(self.batch_group)
        self.batch_content.setVisible(False)

        self.preset_tree = QtWidgets.QTreeWidget()
        self.preset_tree.setMinimumWidth(260)
        self.preset_tree.setHeaderHidden(True)
        self.preset_tree.setColumnCount(2)
        header = self.preset_tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Fixed)
        header.setMinimumSectionSize(8)
        header.resizeSection(1, 28)
        self.preset_tree.setRootIsDecorated(True)
        self.preset_tree.setUniformRowHeights(True)
        self.preset_tree.setWordWrap(False)
        self.preset_tree.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self.preset_tree.setItemDelegate(_FixedRowHeightDelegate(22, self.preset_tree))

        presets_hdr = QtWidgets.QWidget()
        presets_hdr_l = QtWidgets.QHBoxLayout(presets_hdr)
        presets_hdr_l.setContentsMargins(0, 0, 0, 0)
        presets_hdr_l.setSpacing(6)
        presets_hdr_l.addWidget(QtWidgets.QLabel("Presets"))
        presets_hdr_l.addStretch(1)

        self.preset_favorites_only = QtWidgets.QCheckBox("Fav only")
        self.preset_favorites_only.setChecked(False)

        presets_hdr_l.addWidget(self.preset_favorites_only)

        left_layout.addWidget(presets_hdr)
        left_layout.addWidget(self.preset_tree, 1)
        self._rebuild_preset_tree(select_name="None")

        self.reset_btn = QtWidgets.QPushButton("Reset")
        left_layout.addWidget(self.reset_btn)

        layout.addWidget(left)

        # Right side: editor and upscale tabs
        right_tabs = QtWidgets.QTabWidget()
        editor_tab = QtWidgets.QWidget()
        right_layout = QtWidgets.QHBoxLayout(editor_tab)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        self.image_label = CensorImageLabel("Load an image to begin")
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(1, 1)
        self.image_label.installEventFilter(self)

        preview_panel = QtWidgets.QWidget()
        preview_layout = QtWidgets.QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(6)

        zoom_row = QtWidgets.QWidget()
        zoom_row_layout = QtWidgets.QHBoxLayout(zoom_row)
        zoom_row_layout.setContentsMargins(0, 0, 0, 0)
        zoom_row_layout.setSpacing(6)

        self.zoom_out_btn = QtWidgets.QPushButton("-")
        self.zoom_out_btn.setFixedWidth(28)
        self.zoom_in_btn = QtWidgets.QPushButton("+")
        self.zoom_in_btn.setFixedWidth(28)
        self.undo_btn = QtWidgets.QPushButton("<")
        self.undo_btn.setFixedWidth(28)
        self.redo_btn = QtWidgets.QPushButton(">")
        self.redo_btn.setFixedWidth(28)
        self.zoom_undo_divider = QtWidgets.QFrame()
        self.zoom_undo_divider.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        self.zoom_undo_divider.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.zoom_fit_btn = QtWidgets.QPushButton("Fit")
        self.zoom_actual_btn = QtWidgets.QPushButton("Actual Size")
        self.split_preview_btn = QtWidgets.QPushButton("Split Preview")
        self.split_preview_btn.setCheckable(True)
        self.zoom_value_label = QtWidgets.QLabel("Fit")

        zoom_row_layout.addWidget(self.undo_btn)
        zoom_row_layout.addWidget(self.redo_btn)
        zoom_row_layout.addSpacing(6)
        zoom_row_layout.addWidget(self.zoom_undo_divider)
        zoom_row_layout.addSpacing(6)
        zoom_row_layout.addStretch(1)
        zoom_row_layout.addWidget(self.split_preview_btn)
        zoom_row_layout.addStretch(1)
        zoom_row_layout.addWidget(self.zoom_out_btn)
        zoom_row_layout.addWidget(self.zoom_in_btn)
        zoom_row_layout.addWidget(self.zoom_fit_btn)
        zoom_row_layout.addWidget(self.zoom_actual_btn)
        zoom_row_layout.addWidget(self.zoom_value_label)

        self.censor_toolbar = QtWidgets.QWidget()
        censor_toolbar_layout = QtWidgets.QHBoxLayout(self.censor_toolbar)
        censor_toolbar_layout.setContentsMargins(0, 0, 0, 0)
        censor_toolbar_layout.setSpacing(6)
        self.censor_btn = QtWidgets.QPushButton("Censor")
        self.censor_btn.setCheckable(True)
        self.censor_blur_label = QtWidgets.QLabel("Blur")
        self.censor_blur_spin = QtWidgets.QSpinBox()
        self.censor_blur_spin.setRange(1, 100)
        self.censor_blur_spin.setValue(self._censor_blur_radius)
        self.censor_blur_spin.setSuffix(" px")
        self.censor_blur_spin.setFixedWidth(82)
        self.censor_remove_btn = QtWidgets.QPushButton("Remove Last")
        self.censor_clear_btn = QtWidgets.QPushButton("Clear")
        censor_toolbar_layout.addWidget(self.censor_btn)
        censor_toolbar_layout.addWidget(self.censor_blur_label)
        censor_toolbar_layout.addWidget(self.censor_blur_spin)
        censor_toolbar_layout.addWidget(self.censor_remove_btn)
        censor_toolbar_layout.addWidget(self.censor_clear_btn)
        censor_toolbar_layout.addStretch(1)
        self.censor_toolbar.setEnabled(False)

        self.scroll = _PanScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.image_label)
        preview_layout.addWidget(zoom_row)
        preview_layout.addWidget(self.censor_toolbar)
        preview_layout.addWidget(self.scroll, 1)
        right_layout.addWidget(preview_panel, 1)

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
        self.sidebar_toggle.setText("Modifications")
        container_layout.addWidget(self.sidebar_toggle)

        self.adjust_sidebar = QtWidgets.QWidget()
        self.adjust_sidebar.setObjectName("AdjustSidebar")
        sidebar_layout = QtWidgets.QVBoxLayout(self.adjust_sidebar)
        sidebar_layout.setContentsMargins(0, 0, 8, 0)
        sidebar_layout.setSpacing(10)

        self.strength_label = QtWidgets.QLabel("Strength: 100%")
        self.strength_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(100)
        self.strength_slider.setMinimumWidth(180)
        sidebar_layout.addWidget(self.strength_label)
        sidebar_layout.addWidget(self.strength_slider)

        self._adjust_scroll = QtWidgets.QScrollArea()
        self._adjust_scroll.setWidgetResizable(True)
        self._adjust_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        adjust_root = QtWidgets.QWidget()
        self._adjust_scroll.setWidget(adjust_root)
        self._adjust_layout = QtWidgets.QVBoxLayout(adjust_root)
        self._adjust_layout.setContentsMargins(0, 0, 8, 0)
        self._adjust_layout.setSpacing(8)
        self._build_adjustment_widgets()
        sidebar_layout.addWidget(self._adjust_scroll, 1)

        container_layout.addWidget(self.adjust_sidebar, 1)
        right_layout.addWidget(self.sidebar_container)

        self._set_adjustments_visible(False)

        right_tabs.addTab(editor_tab, "Edit")
        upscale_tab = QtWidgets.QWidget()
        self._build_upscale_tab(upscale_tab)
        right_tabs.addTab(upscale_tab, "Upscale")
        layout.addWidget(right_tabs, 1)

        # Wire up
        self.load_btn.clicked.connect(self._on_load)
        self.load_lut_btn.clicked.connect(self._on_load_lut)
        self.load_preset_btn.clicked.connect(self._on_load_preset)
        self.save_preset_btn.clicked.connect(self._on_save_preset)
        self.save_btn.clicked.connect(self._on_save)
        self.reset_btn.clicked.connect(self._on_reset)
        self.preset_tree.currentItemChanged.connect(self._on_preset_item_changed)
        self.preset_tree.itemClicked.connect(self._on_preset_item_clicked)
        self.preset_favorites_only.toggled.connect(self._on_favorites_only_toggled)
        self.strength_slider.valueChanged.connect(self._on_strength_change)
        self.sidebar_toggle.toggled.connect(self._set_adjustments_visible)
        self.zoom_in_btn.clicked.connect(self._on_zoom_in)
        self.zoom_out_btn.clicked.connect(self._on_zoom_out)
        self.zoom_fit_btn.clicked.connect(self._on_zoom_fit)
        self.zoom_actual_btn.clicked.connect(self._on_zoom_actual)
        self.undo_btn.clicked.connect(self._on_undo)
        self.redo_btn.clicked.connect(self._on_redo)
        self.split_preview_btn.toggled.connect(self._on_split_preview_toggled)
        self.censor_btn.toggled.connect(self._on_censor_toggled)
        self.censor_blur_spin.valueChanged.connect(self._on_censor_blur_changed)
        self.censor_remove_btn.clicked.connect(self._on_censor_remove_last)
        self.censor_clear_btn.clicked.connect(self._on_censor_clear)
        self.image_label.circleCreated.connect(self._on_censor_circle_created)
        self.scroll.pinchZoomRequested.connect(self._on_pinch_zoom)
        self.scroll.wheelZoomRequested.connect(self._on_wheel_zoom)

        self.user_presets_browse_btn.clicked.connect(self._on_browse_user_presets_dir)
        self.user_presets_edit.editingFinished.connect(self._on_user_presets_dir_edited)

        self.batch_group.toggled.connect(self.batch_content.setVisible)
        self.batch_in_browse.clicked.connect(self._on_browse_batch_input)
        self.batch_out_browse.clicked.connect(self._on_browse_batch_output)
        self.batch_in_edit.editingFinished.connect(self._on_batch_paths_edited)
        self.batch_out_edit.editingFinished.connect(self._on_batch_paths_edited)
        self.batch_run_btn.clicked.connect(self._on_run_batch)

        self.open_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Open, self)
        self.open_shortcut.activated.connect(self._on_load)
        self.save_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Save, self)
        self.save_shortcut.activated.connect(self._on_save)
        self.censor_escape_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Escape"), self)
        self.censor_escape_shortcut.activated.connect(self.image_label.cancel_censor_drag)
        self.censor_escape_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self.undo_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.activated.connect(self._on_undo)
        self.undo_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self.redo_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Redo, self)
        self.redo_shortcut.activated.connect(self._on_redo)
        self.redo_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self.undo_shortcut_meta = QtGui.QShortcut(QtGui.QKeySequence("Meta+Z"), self)
        self.undo_shortcut_meta.activated.connect(self._on_undo)
        self.undo_shortcut_meta.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self.redo_shortcut_meta = QtGui.QShortcut(QtGui.QKeySequence("Meta+Shift+Z"), self)
        self.redo_shortcut_meta.activated.connect(self._on_redo)
        self.redo_shortcut_meta.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)

        self.zoom_in_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Meta++"), self)
        self.zoom_in_shortcut.activated.connect(self._on_zoom_in)
        self.zoom_in_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self.zoom_in_shortcut_alt = QtGui.QShortcut(QtGui.QKeySequence("Meta+="), self)
        self.zoom_in_shortcut_alt.activated.connect(self._on_zoom_in)
        self.zoom_in_shortcut_alt.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)

        self.zoom_out_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Meta+-"), self)
        self.zoom_out_shortcut.activated.connect(self._on_zoom_out)
        self.zoom_out_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)

        self.actual_size_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Meta+A"), self)
        self.actual_size_shortcut.activated.connect(self._on_zoom_actual)
        self.actual_size_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)

        self.fit_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Meta+F"), self)
        self.fit_shortcut.activated.connect(self._on_zoom_fit)
        self.fit_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)

        self.zoom_in_shortcut_ctrl = QtGui.QShortcut(QtGui.QKeySequence("Ctrl++"), self)
        self.zoom_in_shortcut_ctrl.activated.connect(self._on_zoom_in)
        self.zoom_in_shortcut_ctrl.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self.zoom_in_shortcut_ctrl_alt = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+="), self)
        self.zoom_in_shortcut_ctrl_alt.activated.connect(self._on_zoom_in)
        self.zoom_in_shortcut_ctrl_alt.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)

        self.zoom_out_shortcut_ctrl = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+-"), self)
        self.zoom_out_shortcut_ctrl.activated.connect(self._on_zoom_out)
        self.zoom_out_shortcut_ctrl.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)

        self.actual_size_shortcut_ctrl = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+A"), self)
        self.actual_size_shortcut_ctrl.activated.connect(self._on_zoom_actual)
        self.actual_size_shortcut_ctrl.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)

        self.fit_shortcut_ctrl = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+F"), self)
        self.fit_shortcut_ctrl.activated.connect(self._on_zoom_fit)
        self.fit_shortcut_ctrl.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        open_hint = self.open_shortcut.key().toString(QtGui.QKeySequence.SequenceFormat.NativeText)
        save_hint = self.save_shortcut.key().toString(QtGui.QKeySequence.SequenceFormat.NativeText)
        self.load_btn.setToolTip(f"Load image ({open_hint})")
        self.save_btn.setToolTip(f"Save image ({save_hint})")

        self._history_last_state = self._make_history_state()
        self._update_censor_controls()

        self._apply_current()

        # Load user presets from configured folder on startup.
        self._reload_user_presets_from_folder()

        self._batch_running = False

    def _on_load(self) -> None:
        start_dir = self.last_open_image_dir.strip() if getattr(self, "last_open_image_dir", "") else ""
        if not start_dir or not Path(start_dir).exists():
            start_dir = str(Path.home())
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Image",
            start_dir,
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp);;All Files (*)",
        )
        if not fn:
            return

        path = Path(fn)
        self.last_open_image_dir = str(path.parent)
        self._settings.setValue("lastOpenImageDir", self.last_open_image_dir)
        pil_img = Image.open(path)

        # Keep full-res original; for preview, cap size for interactive speed
        original_rgb8 = pil_to_rgb8(pil_img)

        preview_pil = pil_img.convert("RGB")
        preview_pil.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        preview_rgb8 = pil_to_rgb8(preview_pil)

        self._loaded = LoadedImage(path=path, original_rgb8=original_rgb8, preview_rgb8=preview_rgb8)
        self._original_preview_pixmap = QtGui.QPixmap.fromImage(rgb8_to_qimage(preview_rgb8))
        self._censor_circles = ()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._history_last_state = self._make_history_state()
        self._update_censor_controls()
        self._update_upscale_source_label()
        if self._upscale_output_auto:
            self._set_default_upscale_output_path()
        self._update_upscale_controls()
        self.save_btn.setEnabled(True)
        self._zoom_mode = "fit"
        self._zoom_factor = 1.0
        self._apply_current()

    def _on_save(self) -> None:
        if not self._loaded:
            return

        save_suffix = "_censored" if self._censor_circles else "_filtered"
        default_name = self._loaded.path.with_name(self._loaded.path.stem + save_suffix + self._loaded.path.suffix)
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Image As",
            str(default_name),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff);;WEBP (*.webp);;BMP (*.bmp)",
        )
        if not fn:
            return

        out_path = Path(fn)
        rgb8 = self._render_censored_output_rgb8()
        save_metadata_free_rgb8(rgb8, out_path)

    def _on_censor_toggled(self, checked: bool) -> None:
        self._censor_enabled = bool(checked)
        if self._censor_enabled and self.split_preview_btn.isChecked():
            self.split_preview_btn.setChecked(False)
        self.split_preview_btn.setEnabled(not self._censor_enabled)
        self.image_label.set_censor_enabled(self._censor_enabled)
        self._update_censor_controls()

    def _on_censor_circle_created(self, circle: object) -> None:
        if not isinstance(circle, CensorCircle):
            return
        self._censor_circles = self._censor_circles + (circle,)
        self._record_history_if_changed()
        self._update_censor_controls()
        self._schedule_apply()

    def _on_censor_blur_changed(self, value: int) -> None:
        self._censor_blur_radius = int(value)
        self._settings.setValue("censor/blurRadius", self._censor_blur_radius)
        self._settings.sync()
        self._record_history_if_changed()
        self._schedule_apply()

    def _on_censor_remove_last(self) -> None:
        if not self._censor_circles:
            return
        self._censor_circles = self._censor_circles[:-1]
        self._record_history_if_changed()
        self._update_censor_controls()
        self._schedule_apply()

    def _on_censor_clear(self) -> None:
        if not self._censor_circles:
            return
        if len(self._censor_circles) > 1:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Clear Censor Regions",
                "Remove all censor regions?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self._censor_circles = ()
        self._record_history_if_changed()
        self._update_censor_controls()
        self._schedule_apply()

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

    def _on_browse_batch_input(self) -> None:
        start = self.batch_in_edit.text().strip() or str(Path.home())
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Batch Input Folder", start)
        if not folder:
            return
        self.batch_in_edit.setText(folder)
        self._on_batch_paths_edited()

    def _on_browse_batch_output(self) -> None:
        start = self.batch_out_edit.text().strip() or str(Path.home())
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Batch Output Folder", start)
        if not folder:
            return
        self.batch_out_edit.setText(folder)
        self._on_batch_paths_edited()

    def _on_batch_paths_edited(self) -> None:
        self.batch_input_dir = self.batch_in_edit.text().strip()
        self.batch_output_dir = self.batch_out_edit.text().strip()
        self._settings.setValue("batchInputDir", self.batch_input_dir)
        self._settings.setValue("batchOutputDir", self.batch_output_dir)

    def _iter_batch_files(self, folder: Path) -> Iterable[Path]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                yield p

    def _on_run_batch(self) -> None:
        if self._batch_running:
            return

        in_dir = Path(self.batch_in_edit.text().strip())
        out_dir = Path(self.batch_out_edit.text().strip())
        if not in_dir.exists() or not in_dir.is_dir():
            QtWidgets.QMessageBox.warning(self, "Batch", "Please choose a valid input folder.")
            return
        if not out_dir:
            QtWidgets.QMessageBox.warning(self, "Batch", "Please choose an output folder.")
            return

        files = list(self._iter_batch_files(in_dir))
        if not files:
            QtWidgets.QMessageBox.information(self, "Batch", "No supported images found in input folder.")
            return

        self._batch_running = True
        self.batch_run_btn.setEnabled(False)
        self.batch_progress.setRange(0, len(files))
        self.batch_progress.setValue(0)
        self.batch_progress.setFormat("Starting…")

        task = _BatchTask(
            files=files,
            output_dir=out_dir,
            base_params=self._base_params,
            adjust_params=self._effective_adjust_params(),
            strength=self._strength,
        )
        task.signals.progress.connect(self._on_batch_progress)
        task.signals.finished.connect(self._on_batch_finished)
        task.signals.failed.connect(self._on_batch_failed)
        self._thread_pool.start(task)

    def _on_batch_progress(self, done: int, total: int, filename: str) -> None:
        self.batch_progress.setValue(done)
        self.batch_progress.setFormat(f"{done}/{total} — {filename}")

    def _on_batch_finished(self, ok: int, failed: int) -> None:
        self._batch_running = False
        self.batch_run_btn.setEnabled(True)
        self.batch_progress.setFormat(f"Done — OK: {ok}, Failed: {failed}")
        QtWidgets.QMessageBox.information(self, "Batch", f"Finished.\n\nOK: {ok}\nFailed: {failed}")

    def _on_batch_failed(self, err: str) -> None:
        self._batch_running = False
        self.batch_run_btn.setEnabled(True)
        self.batch_progress.setFormat("Failed")
        QtWidgets.QMessageBox.critical(self, "Batch Error", err)

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
        self._record_history_if_changed()
        self._schedule_apply()

    def _on_preset_item_clicked(
        self,
        item: QtWidgets.QTreeWidgetItem,
        column: int,
    ) -> None:
        idx = item.data(0, int(QtCore.Qt.ItemDataRole.UserRole))
        if idx is not None and column == 1:
            try:
                i = int(idx)
            except (TypeError, ValueError):
                return
            self._on_favorite_button_for_index(i)
            return

        # Toggle expansion when clicking on category names (not presets)
        if idx is None:  # This is a category item
            item.setExpanded(not item.isExpanded())

    def _on_favorite_button_for_index(self, i: int) -> None:
        if i < 0 or i >= len(self._presets):
            return
        preset = self._presets[i]
        key = self._preset_key(preset)
        if key in self._favorite_keys:
            self._favorite_keys.remove(key)
        else:
            self._favorite_keys.add(key)
        self._save_favorites()
        self._rebuild_preset_tree(select_name=preset.name)

    def _current_preset_index(self) -> int | None:
        item = self.preset_tree.currentItem()
        if not item:
            return None
        idx = item.data(0, int(QtCore.Qt.ItemDataRole.UserRole))
        if idx is None:
            return None
        try:
            i = int(idx)
        except (TypeError, ValueError):
            return None
        if 0 <= i < len(self._presets):
            return i
        return None

    def _preset_key(self, preset: FilterPreset) -> str:
        src = preset.source_path or ""
        pilgram_name = preset.params.pilgram_filter or ""
        return f"{preset.category}|{preset.name}|{src}|{pilgram_name}"

    def _is_favorite_preset(self, preset: FilterPreset) -> bool:
        return self._preset_key(preset) in self._favorite_keys

    def _save_favorites(self) -> None:
        self._settings.setValue("favoritePresetKeys", json.dumps(sorted(self._favorite_keys)))

    def _on_favorites_only_toggled(self, checked: bool) -> None:
        current_idx = self._current_preset_index()
        current_name = self._presets[current_idx].name if current_idx is not None else None
        if checked:
            self._preset_tree_expand_state_before_favorites = self._capture_preset_tree_expand_state()
        self._rebuild_preset_tree(select_name=current_name)
        if checked:
            self._set_all_preset_categories_expanded(True)
        elif self._preset_tree_expand_state_before_favorites is not None:
            self._apply_preset_tree_expand_state(self._preset_tree_expand_state_before_favorites)
            self._preset_tree_expand_state_before_favorites = None

    def _capture_preset_tree_expand_state(self) -> dict[str, bool]:
        state: dict[str, bool] = {}
        for i in range(self.preset_tree.topLevelItemCount()):
            item = self.preset_tree.topLevelItem(i)
            state[item.text(0)] = item.isExpanded()
        return state

    def _apply_preset_tree_expand_state(self, state: dict[str, bool]) -> None:
        for i in range(self.preset_tree.topLevelItemCount()):
            item = self.preset_tree.topLevelItem(i)
            name = item.text(0)
            if name in state:
                item.setExpanded(bool(state[name]))

    def _set_all_preset_categories_expanded(self, expanded: bool) -> None:
        for i in range(self.preset_tree.topLevelItemCount()):
            self.preset_tree.topLevelItem(i).setExpanded(bool(expanded))

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
        favorites_only = self.preset_favorites_only.isChecked() if hasattr(self, "preset_favorites_only") else False

        # Group presets by category
        cats: dict[str, list[tuple[int, FilterPreset]]] = {}
        for i, p in enumerate(self._presets):
            if favorites_only and not self._is_favorite_preset(p):
                continue
            cats.setdefault(p.category, []).append((i, p))

        def add_category(cat_name: str) -> None:
            items = cats.get(cat_name)
            if not items:
                return

            cat_item = QtWidgets.QTreeWidgetItem([cat_name])
            cat_item.setData(0, int(QtCore.Qt.ItemDataRole.UserRole), None)
            cat_item.setFlags(cat_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
            cat_item.setSizeHint(0, QtCore.QSize(0, 22))
            self.preset_tree.addTopLevelItem(cat_item)

            # All categories collapsed by default
            cat_item.setExpanded(False)

            for idx, preset in items:
                is_fav = self._is_favorite_preset(preset)
                star = "★" if is_fav else "☆"
                child = QtWidgets.QTreeWidgetItem([preset.name, star])
                child.setData(0, int(QtCore.Qt.ItemDataRole.UserRole), idx)
                child.setSizeHint(0, QtCore.QSize(0, 22))
                child.setTextAlignment(1, int(QtCore.Qt.AlignmentFlag.AlignCenter))
                star_color = QtGui.QColor("#f1c84a") if is_fav else QtGui.QColor("#a0a0a0")
                child.setForeground(1, QtGui.QBrush(star_color))
                cat_item.addChild(child)

        for cat in self._category_order:
            add_category(cat)

        for cat in sorted(c for c in cats.keys() if c not in self._category_order):
            add_category(cat)

        if favorites_only:
            self._set_all_preset_categories_expanded(True)

        self._select_preset_by_name(select_name or "None")

    def _on_strength_change(self, value: int) -> None:
        self._strength = float(value) / 100.0
        self.strength_label.setText(f"Strength: {value}%")
        self._record_history_if_changed()
        self._schedule_apply()

    def _schedule_apply(self) -> None:
        # Debounce rapid slider changes.
        self._apply_timer.start(25)

    def _make_history_state(self) -> _HistoryState:
        return _HistoryState(
            base_params=self._base_params,
            adjust_params=self._adjust_params,
            strength=self._strength,
            censor_circles=tuple(self._censor_circles),
            censor_blur_radius=int(self._censor_blur_radius),
        )

    def _history_state_equal(self, a: _HistoryState, b: _HistoryState) -> bool:
        return (
            a.base_params == b.base_params
            and a.adjust_params == b.adjust_params
            and abs(a.strength - b.strength) < 1e-9
            and a.censor_circles == b.censor_circles
            and a.censor_blur_radius == b.censor_blur_radius
        )

    def _record_history_if_changed(self) -> None:
        if self._history_restoring:
            return
        current = self._make_history_state()
        if self._history_state_equal(current, self._history_last_state):
            return
        self._undo_stack.append(self._history_last_state)
        if len(self._undo_stack) > 200:
            self._undo_stack = self._undo_stack[-200:]
        self._redo_stack.clear()
        self._history_last_state = current

    def _apply_history_state(self, state: _HistoryState) -> None:
        self._history_restoring = True
        self._base_params = state.base_params
        self._adjust_params = state.adjust_params
        self._strength = float(state.strength)
        self._censor_circles = tuple(state.censor_circles)
        self._censor_blur_radius = int(state.censor_blur_radius)
        self.strength_slider.setValue(int(round(self._strength * 100.0)))
        self.strength_label.setText(f"Strength: {int(round(self._strength * 100.0))}%")
        self.censor_blur_spin.blockSignals(True)
        self.censor_blur_spin.setValue(self._censor_blur_radius)
        self.censor_blur_spin.blockSignals(False)
        self._sync_adjustment_widgets_from_state()
        self._update_censor_controls()
        self._history_restoring = False
        self._history_last_state = self._make_history_state()
        self._schedule_apply()

    def _on_undo(self) -> None:
        if not self._undo_stack:
            return
        current = self._make_history_state()
        state = self._undo_stack.pop()
        self._redo_stack.append(current)
        self._apply_history_state(state)

    def _on_redo(self) -> None:
        if not self._redo_stack:
            return
        current = self._make_history_state()
        state = self._redo_stack.pop()
        self._undo_stack.append(current)
        self._apply_history_state(state)

    def _fit_zoom_factor(self) -> float:
        src = self._display_source_pixmap()
        if not src or src.isNull():
            return 1.0
        viewport = self.scroll.viewport().size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            return 1.0
        bw = src.width()
        bh = src.height()
        if bw <= 0 or bh <= 0:
            return 1.0
        return max(0.01, min(float(viewport.width()) / float(bw), float(viewport.height()) / float(bh)))

    def _set_zoom_label(self) -> None:
        if self._zoom_mode == "fit":
            pct = int(round(self._fit_zoom_factor() * 100.0))
            self.zoom_value_label.setText(f"Fit ({pct}%)")
        else:
            pct = int(round(self._zoom_factor * 100.0))
            self.zoom_value_label.setText(f"{pct}%")

    def _on_zoom_in(self) -> None:
        if not self._base_pixmap or self._base_pixmap.isNull():
            return
        self._apply_zoom(self._current_zoom_factor() * 1.25)

    def _on_zoom_out(self) -> None:
        if not self._base_pixmap or self._base_pixmap.isNull():
            return
        self._apply_zoom(self._current_zoom_factor() / 1.25)

    def _on_zoom_fit(self) -> None:
        self._zoom_mode = "fit"
        self._refit_pixmap()

    def _on_zoom_actual(self) -> None:
        if not self._base_pixmap or self._base_pixmap.isNull():
            return
        self._apply_zoom(1.0)

    def _on_pinch_zoom(self, factor: float, anchor_pos: object) -> None:
        src = self._display_source_pixmap()
        if not src or src.isNull():
            return
        anchor = anchor_pos if isinstance(anchor_pos, QtCore.QPoint) else None
        self._apply_zoom(self._current_zoom_factor() * float(factor), anchor)

    def _on_wheel_zoom(self, factor: float, anchor_pos: object) -> None:
        src = self._display_source_pixmap()
        if not src or src.isNull():
            return
        anchor = anchor_pos if isinstance(anchor_pos, QtCore.QPoint) else None
        self._apply_zoom(self._current_zoom_factor() * float(factor), anchor)

    def _on_split_preview_toggled(self, checked: bool) -> None:
        if checked and self._censor_enabled:
            self.split_preview_btn.setChecked(False)
            return
        self._split_preview_enabled = bool(checked)
        self._dragging_split_divider = False
        if not self._split_preview_enabled:
            self.image_label.unsetCursor()
        self._refit_pixmap()

    def _update_split_divider_from_label_x(self, x: int) -> None:
        w = max(1, self.image_label.width())
        ratio = float(x) / float(w)
        self._split_divider_ratio = max(0.02, min(0.98, ratio))
        self._refit_pixmap()

    def _current_zoom_factor(self) -> float:
        if self._zoom_mode == "fit":
            return self._fit_zoom_factor()
        return max(0.05, min(16.0, float(self._zoom_factor)))

    def _apply_zoom(self, new_factor: float, anchor_pos: QtCore.QPoint | None = None) -> None:
        src = self._display_source_pixmap()
        if not src or src.isNull():
            return
        viewport = self.scroll.viewport().size()
        if viewport.width() <= 2 or viewport.height() <= 2:
            return

        old_factor = self._current_zoom_factor()
        old_w = max(1, int(round(src.width() * old_factor)))
        old_h = max(1, int(round(src.height() * old_factor)))

        if anchor_pos is None:
            anchor_pos = QtCore.QPoint(viewport.width() // 2, viewport.height() // 2)

        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        old_content_x = hbar.value() + anchor_pos.x()
        old_content_y = vbar.value() + anchor_pos.y()
        ratio_x = old_content_x / float(old_w)
        ratio_y = old_content_y / float(old_h)

        self._zoom_mode = "custom"
        self._zoom_factor = max(0.05, min(16.0, float(new_factor)))
        self._refit_pixmap()

        new_factor_applied = self._current_zoom_factor()
        src_after = self._display_source_pixmap()
        if not src_after or src_after.isNull():
            return
        new_w = max(1, int(round(src_after.width() * new_factor_applied)))
        new_h = max(1, int(round(src_after.height() * new_factor_applied)))
        hbar.setValue(int(round(ratio_x * new_w - anchor_pos.x())))
        vbar.setValue(int(round(ratio_y * new_h - anchor_pos.y())))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        # Re-fit current pixmap to viewport
        self._refit_pixmap()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == int(QtCore.Qt.Key.Key_Escape) and self.image_label.cancel_censor_drag():
            event.accept()
            return
        if event.key() == int(QtCore.Qt.Key.Key_Space) and not event.isAutoRepeat():
            self._space_pan_active = True
            self.scroll.set_space_pan_enabled(True)
            self.image_label.set_space_pan_active(True)
            event.accept()
            return
        if (
            event.key() == int(QtCore.Qt.Key.Key_M)
            and not event.isAutoRepeat()
            and event.modifiers() == QtCore.Qt.KeyboardModifier.NoModifier
        ):
            fw = QtWidgets.QApplication.focusWidget()
            if not isinstance(fw, (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)):
                self.sidebar_toggle.setChecked(not self.sidebar_toggle.isChecked())
                event.accept()
                return
        if not event.isAutoRepeat():
            mods = event.modifiers()
            has_cmd_ctrl = bool(mods & (QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.MetaModifier))
            is_undo = (
                event.key() == int(QtCore.Qt.Key.Key_Z)
                and has_cmd_ctrl
                and not bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
            )
            is_redo = has_cmd_ctrl and (
                (event.key() == int(QtCore.Qt.Key.Key_Z) and bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier))
                or event.key() == int(QtCore.Qt.Key.Key_Y)
            )
            if is_undo:
                self._on_undo()
                event.accept()
                return
            if is_redo:
                self._on_redo()
                event.accept()
                return
        super().keyPressEvent(event)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self.image_label and self._split_preview_enabled:
            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                me = event
                if me.button() == QtCore.Qt.MouseButton.LeftButton and not self._space_pan_active:
                    split_x = int(round(self.image_label.width() * self._split_divider_ratio))
                    if abs(me.position().x() - split_x) <= 8:
                        self._dragging_split_divider = True
                        self.image_label.setCursor(QtCore.Qt.CursorShape.SplitHCursor)
                        self._update_split_divider_from_label_x(int(me.position().x()))
                        return True
            elif event.type() == QtCore.QEvent.Type.MouseMove:
                me = event
                if self._dragging_split_divider:
                    self._update_split_divider_from_label_x(int(me.position().x()))
                    return True
                split_x = int(round(self.image_label.width() * self._split_divider_ratio))
                if abs(me.position().x() - split_x) <= 8:
                    self.image_label.setCursor(QtCore.Qt.CursorShape.SplitHCursor)
                else:
                    self.image_label.unsetCursor()
            elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                me = event
                if self._dragging_split_divider and me.button() == QtCore.Qt.MouseButton.LeftButton:
                    self._dragging_split_divider = False
                    self.image_label.unsetCursor()
                    return True
        return super().eventFilter(watched, event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == int(QtCore.Qt.Key.Key_Space) and not event.isAutoRepeat():
            self._space_pan_active = False
            self.scroll.set_space_pan_enabled(False)
            self.image_label.set_space_pan_active(False)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _apply_current(self) -> None:
        if not self._loaded:
            self.image_label.setText("Load an image to begin")
            self.image_label.setPixmap(QtGui.QPixmap())
            self.image_label.adjustSize()
            self._base_pixmap = None
            self._original_preview_pixmap = None
            self._set_zoom_label()
            return

        self._render_generation += 1
        gen = self._render_generation

        task = _RenderTask(
            generation=gen,
            preview_rgb8=self._loaded.preview_rgb8,
            base_params=self._base_params,
            adjust_params=self._effective_adjust_params(),
            strength=self._strength,
            censor_circles=self._censor_circles,
            censor_blur_radius=self._preview_censor_blur_radius(),
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

    def _build_upscale_tab(self, parent: QtWidgets.QWidget) -> None:
        layout = QtWidgets.QVBoxLayout(parent)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        hint = QtWidgets.QLabel(
            "Upscale the currently loaded image after its color adjustments. "
            "Censor regions are excluded from the upscale source, and the original file is never modified."
        )
        hint.setWordWrap(True)
        hint.setObjectName("hintLabel")
        layout.addWidget(hint)
        self.upscale_source_label = QtWidgets.QLabel("Source: No image loaded")
        self.upscale_source_label.setObjectName("sourceLabel")
        layout.addWidget(self.upscale_source_label)

        connection_group = QtWidgets.QGroupBox("ComfyUI")
        connection_form = QtWidgets.QFormLayout(connection_group)
        connection_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.upscale_url_edit = QtWidgets.QLineEdit(
            str(self._settings.value("upscale/comfyUrl", "http://127.0.0.1:8000"))
        )
        self.upscale_url_edit.setPlaceholderText("http://127.0.0.1:8000")
        self.upscale_engine_combo = QtWidgets.QComboBox()
        self.upscale_engine_combo.addItems(["NVIDIA RTX", "SeedVR"])
        connection_form.addRow("URL", self.upscale_url_edit)
        connection_form.addRow("Engine", self.upscale_engine_combo)
        self.upscale_connection_group = connection_group
        layout.addWidget(connection_group)

        rtx_group = QtWidgets.QGroupBox("NVIDIA RTX Settings")
        rtx_form = QtWidgets.QFormLayout(rtx_group)
        self.upscale_rtx_resize_type_combo = QtWidgets.QComboBox()
        self.upscale_rtx_resize_type_combo.addItems(["scale by multiplier", "resize to width and height"])
        self.upscale_rtx_scale_spin = QtWidgets.QDoubleSpinBox()
        self.upscale_rtx_scale_spin.setRange(1.0, 8.0)
        self.upscale_rtx_scale_spin.setSingleStep(0.5)
        self.upscale_rtx_scale_spin.setDecimals(1)
        self.upscale_rtx_scale_spin.setValue(2.0)
        self.upscale_rtx_scale_spin.setSuffix("x")
        self.upscale_rtx_width_spin = QtWidgets.QSpinBox()
        self.upscale_rtx_width_spin.setRange(64, 16384)
        self.upscale_rtx_width_spin.setSingleStep(64)
        self.upscale_rtx_width_spin.setValue(3840)
        self.upscale_rtx_height_spin = QtWidgets.QSpinBox()
        self.upscale_rtx_height_spin.setRange(64, 16384)
        self.upscale_rtx_height_spin.setSingleStep(64)
        self.upscale_rtx_height_spin.setValue(2160)
        self.upscale_rtx_quality_combo = QtWidgets.QComboBox()
        self.upscale_rtx_quality_combo.addItems(["ULTRA", "HIGH", "MEDIUM", "LOW"])

        rtx_wh_row = QtWidgets.QWidget()
        rtx_wh_layout = QtWidgets.QHBoxLayout(rtx_wh_row)
        rtx_wh_layout.setContentsMargins(0, 0, 0, 0)
        rtx_wh_layout.addWidget(self.upscale_rtx_width_spin)
        rtx_wh_layout.addWidget(QtWidgets.QLabel("x"))
        rtx_wh_layout.addWidget(self.upscale_rtx_height_spin)
        self.upscale_rtx_scale_label = QtWidgets.QLabel("Scale")
        self.upscale_rtx_wh_label = QtWidgets.QLabel("Width x Height")
        rtx_form.addRow("Resize Type", self.upscale_rtx_resize_type_combo)
        rtx_form.addRow(self.upscale_rtx_scale_label, self.upscale_rtx_scale_spin)
        rtx_form.addRow(self.upscale_rtx_wh_label, rtx_wh_row)
        rtx_form.addRow("Quality", self.upscale_rtx_quality_combo)
        layout.addWidget(rtx_group)

        seedvr_group = QtWidgets.QGroupBox("SeedVR Settings")
        seedvr_form = QtWidgets.QFormLayout(seedvr_group)
        self.upscale_seedvr_resolution_spin = QtWidgets.QSpinBox()
        self.upscale_seedvr_resolution_spin.setRange(256, 16384)
        self.upscale_seedvr_resolution_spin.setSingleStep(256)
        self.upscale_seedvr_resolution_spin.setValue(4096)
        self.upscale_seedvr_max_resolution_spin = QtWidgets.QSpinBox()
        self.upscale_seedvr_max_resolution_spin.setRange(256, 16384)
        self.upscale_seedvr_max_resolution_spin.setSingleStep(256)
        self.upscale_seedvr_max_resolution_spin.setValue(4096)
        self.upscale_seedvr_seed_spin = QtWidgets.QSpinBox()
        self.upscale_seedvr_seed_spin.setRange(0, 2147483647)
        self.upscale_seedvr_seed_spin.setValue(42)
        seedvr_form.addRow("Resolution", self.upscale_seedvr_resolution_spin)
        seedvr_form.addRow("Max Resolution", self.upscale_seedvr_max_resolution_spin)
        seedvr_form.addRow("Seed", self.upscale_seedvr_seed_spin)
        layout.addWidget(seedvr_group)

        launch_group = QtWidgets.QGroupBox("ComfyUI Launch")
        launch_form = QtWidgets.QFormLayout(launch_group)
        default_comfyui_python, default_comfyui_main, default_comfyui_extra_args = _default_comfyui_launch_settings()
        self.upscale_python_edit = QtWidgets.QLineEdit(default_comfyui_python)
        self.upscale_python_edit.setPlaceholderText("Optional path to ComfyUI venv python.exe")
        self.upscale_python_browse_btn = QtWidgets.QPushButton("Browse")
        python_row = QtWidgets.QWidget()
        python_layout = QtWidgets.QHBoxLayout(python_row)
        python_layout.setContentsMargins(0, 0, 0, 0)
        python_layout.addWidget(self.upscale_python_edit, 1)
        python_layout.addWidget(self.upscale_python_browse_btn)
        self.upscale_main_edit = QtWidgets.QLineEdit(default_comfyui_main)
        self.upscale_main_edit.setPlaceholderText("Optional path to ComfyUI main.py")
        self.upscale_main_browse_btn = QtWidgets.QPushButton("Browse")
        main_row = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(main_row)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.upscale_main_edit, 1)
        main_layout.addWidget(self.upscale_main_browse_btn)
        self.upscale_extra_args_edit = QtWidgets.QLineEdit(default_comfyui_extra_args)
        self.upscale_extra_args_edit.setPlaceholderText("Optional extra launch args")
        launch_form.addRow("Python Exe", python_row)
        launch_form.addRow("ComfyUI main.py", main_row)
        launch_form.addRow("Extra Args", self.upscale_extra_args_edit)
        layout.addWidget(launch_group)
        self.upscale_launch_group = launch_group

        output_group = QtWidgets.QGroupBox("Output")
        output_form = QtWidgets.QFormLayout(output_group)
        self.upscale_output_edit = QtWidgets.QLineEdit()
        self.upscale_output_edit.setPlaceholderText("Defaults to <source>_upscaled.png")
        self.upscale_output_browse_btn = QtWidgets.QPushButton("Browse")
        output_row = QtWidgets.QWidget()
        output_layout = QtWidgets.QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.upscale_output_edit, 1)
        output_layout.addWidget(self.upscale_output_browse_btn)
        output_form.addRow("Output File", output_row)
        layout.addWidget(output_group)
        self.upscale_output_group = output_group

        action_row = QtWidgets.QHBoxLayout()
        self.upscale_launch_btn = QtWidgets.QPushButton("Launch ComfyUI")
        self.upscale_stop_btn = QtWidgets.QPushButton("Stop ComfyUI")
        self.upscale_stop_btn.setObjectName("stopButton")
        self.upscale_cancel_btn = QtWidgets.QPushButton("Cancel")
        self.upscale_run_btn = QtWidgets.QPushButton("Upscale Current Image")
        self.upscale_run_btn.setObjectName("upscaleButton")
        self.upscale_package_btn = QtWidgets.QPushButton("Upscale and Zip")
        self.upscale_package_btn.setObjectName("upscalePackageButton")
        self.upscale_package_btn.setToolTip(
            "Create a censored preview and a ZIP containing the uncensored upscale"
        )
        action_row.addWidget(self.upscale_launch_btn)
        action_row.addWidget(self.upscale_stop_btn)
        action_row.addStretch(1)
        action_row.addWidget(self.upscale_cancel_btn)
        action_row.addWidget(self.upscale_run_btn)
        action_row.addWidget(self.upscale_package_btn)
        layout.addLayout(action_row)

        self.upscale_progress = QtWidgets.QProgressBar()
        self.upscale_progress.setRange(0, 1)
        self.upscale_progress.setValue(0)
        self.upscale_status_label = QtWidgets.QLabel("Load an image to begin")
        self.upscale_log = QtWidgets.QPlainTextEdit()
        self.upscale_log.setReadOnly(True)
        self.upscale_log.setMinimumHeight(100)
        layout.addWidget(self.upscale_progress)
        layout.addWidget(self.upscale_status_label)
        layout.addWidget(self.upscale_log, 1)

        self.upscale_rtx_group = rtx_group
        self.upscale_seedvr_group = seedvr_group
        self.upscale_rtx_resize_type_combo.currentTextChanged.connect(self._on_upscale_resize_type_changed)
        self.upscale_engine_combo.currentTextChanged.connect(self._on_upscale_engine_changed)
        self.upscale_output_edit.textEdited.connect(lambda: self._set_upscale_output_auto(False))
        self.upscale_python_browse_btn.clicked.connect(self._on_browse_upscale_python)
        self.upscale_main_browse_btn.clicked.connect(self._on_browse_upscale_main)
        self.upscale_output_browse_btn.clicked.connect(self._on_browse_upscale_output)
        self.upscale_launch_btn.clicked.connect(self._on_launch_comfyui)
        self.upscale_stop_btn.clicked.connect(self._on_stop_comfyui)
        self.upscale_cancel_btn.clicked.connect(self._on_cancel_upscale)
        self.upscale_run_btn.clicked.connect(self._on_run_upscale)
        self.upscale_package_btn.clicked.connect(self._on_upscale_and_zip)

        self._on_upscale_resize_type_changed(self.upscale_rtx_resize_type_combo.currentText())
        self._on_upscale_engine_changed(self.upscale_engine_combo.currentText())
        self._apply_stored_upscale_settings()
        self._update_upscale_controls()

    def _apply_stored_upscale_settings(self) -> None:
        settings = self._settings

        def read_text(key: str, widget: QtWidgets.QLineEdit) -> None:
            value = settings.value(key, "")
            if value:
                widget.setText(str(value))

        read_text("upscale/comfyUrl", self.upscale_url_edit)
        read_text("upscale/comfyuiPython", self.upscale_python_edit)
        read_text("upscale/comfyuiMain", self.upscale_main_edit)
        read_text("upscale/comfyuiExtraArgs", self.upscale_extra_args_edit)

        engine = settings.value("upscale/engine", "")
        if engine:
            self.upscale_engine_combo.setCurrentText(str(engine))
        resize_type = settings.value("upscale/rtxResizeType", "")
        if resize_type:
            self.upscale_rtx_resize_type_combo.setCurrentText(str(resize_type))
        scale = settings.value("upscale/rtxScale", None)
        if scale is not None:
            self.upscale_rtx_scale_spin.setValue(float(str(scale)))
        width = settings.value("upscale/rtxWidth", None)
        if width is not None:
            self.upscale_rtx_width_spin.setValue(int(str(width)))
        height = settings.value("upscale/rtxHeight", None)
        if height is not None:
            self.upscale_rtx_height_spin.setValue(int(str(height)))
        quality = settings.value("upscale/rtxQuality", "")
        if quality:
            self.upscale_rtx_quality_combo.setCurrentText(str(quality))
        resolution = settings.value("upscale/seedvrResolution", None)
        if resolution is not None:
            self.upscale_seedvr_resolution_spin.setValue(int(str(resolution)))
        max_resolution = settings.value("upscale/seedvrMaxResolution", None)
        if max_resolution is not None:
            self.upscale_seedvr_max_resolution_spin.setValue(int(str(max_resolution)))
        seed = settings.value("upscale/seedvrSeed", None)
        if seed is not None:
            self.upscale_seedvr_seed_spin.setValue(int(str(seed)))
        output_path = settings.value("upscale/outputPath", "")
        if output_path:
            self.upscale_output_edit.setText(str(output_path))
            self._upscale_output_auto = False

    def _set_upscale_output_auto(self, enabled: bool) -> None:
        self._upscale_output_auto = bool(enabled)

    def _default_upscale_output_path(self) -> Path | None:
        if not self._loaded:
            return None
        return self._loaded.path.with_name(f"{self._loaded.path.stem}_upscaled.png")

    def _unique_upscale_output_path(self, path: Path) -> Path:
        if not path.exists():
            return path
        index = 1
        while True:
            candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
            if not candidate.exists():
                return candidate
            index += 1

    def _set_default_upscale_output_path(self) -> None:
        default_path = self._default_upscale_output_path()
        if default_path is None:
            return
        self.upscale_output_edit.setText(str(self._unique_upscale_output_path(default_path)))
        self._upscale_output_auto = True

    def _update_upscale_source_label(self) -> None:
        if self._loaded is None:
            self.upscale_source_label.setText("Source: No image loaded")
            return
        height, width = self._loaded.original_rgb8.shape[:2]
        self.upscale_source_label.setText(f"Source: {self._loaded.path.name} ({width} x {height})")

    def _on_upscale_resize_type_changed(self, text: str) -> None:
        is_scale = text == "scale by multiplier"
        self.upscale_rtx_scale_label.setVisible(is_scale)
        self.upscale_rtx_scale_spin.setVisible(is_scale)
        self.upscale_rtx_wh_label.setVisible(not is_scale)
        self.upscale_rtx_width_spin.setVisible(not is_scale)
        self.upscale_rtx_height_spin.setVisible(not is_scale)

    def _on_upscale_engine_changed(self, text: str) -> None:
        is_rtx = text == "NVIDIA RTX"
        self.upscale_rtx_group.setVisible(is_rtx)
        self.upscale_seedvr_group.setVisible(not is_rtx)

    def _update_upscale_controls(self) -> None:
        busy = self._upscale_worker is not None and self._upscale_worker.isRunning()
        has_image = self._loaded is not None
        watcher = self._comfyui_launch_watcher
        watcher_starting = watcher is not None and watcher.isRunning() and not watcher.ready
        comfyui_managed = self._comfyui_launch_is_active()
        self.upscale_run_btn.setEnabled(has_image and not busy and not watcher_starting)
        self.upscale_package_btn.setEnabled(has_image and not busy and not watcher_starting)
        self.upscale_cancel_btn.setEnabled(busy)
        self.upscale_launch_btn.setEnabled(not watcher_starting and not busy and not comfyui_managed)
        self.upscale_stop_btn.setEnabled(comfyui_managed)
        controls_enabled = not busy and not watcher_starting
        for group in (
            self.upscale_connection_group,
            self.upscale_rtx_group,
            self.upscale_seedvr_group,
            self.upscale_launch_group,
            self.upscale_output_group,
        ):
            group.setEnabled(controls_enabled)

    def _save_upscale_settings(self) -> None:
        settings = self._settings
        settings.setValue("upscale/comfyUrl", self._normalise_comfy_url())
        settings.setValue("upscale/engine", self.upscale_engine_combo.currentText())
        settings.setValue("upscale/rtxResizeType", self.upscale_rtx_resize_type_combo.currentText())
        settings.setValue("upscale/rtxScale", self.upscale_rtx_scale_spin.value())
        settings.setValue("upscale/rtxWidth", self.upscale_rtx_width_spin.value())
        settings.setValue("upscale/rtxHeight", self.upscale_rtx_height_spin.value())
        settings.setValue("upscale/rtxQuality", self.upscale_rtx_quality_combo.currentText())
        settings.setValue("upscale/seedvrResolution", self.upscale_seedvr_resolution_spin.value())
        settings.setValue("upscale/seedvrMaxResolution", self.upscale_seedvr_max_resolution_spin.value())
        settings.setValue("upscale/seedvrSeed", self.upscale_seedvr_seed_spin.value())
        settings.setValue("upscale/comfyuiPython", self.upscale_python_edit.text().strip())
        settings.setValue("upscale/comfyuiMain", self.upscale_main_edit.text().strip())
        settings.setValue("upscale/comfyuiExtraArgs", self.upscale_extra_args_edit.text().strip())
        settings.setValue(
            "upscale/outputPath",
            "" if self._upscale_output_auto else self.upscale_output_edit.text().strip(),
        )
        settings.sync()

    def _cleanup_upscale_source_temp(self) -> None:
        temp_path = self._upscale_source_temp
        self._upscale_source_temp = None
        if temp_path is None:
            return
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _append_upscale_log(self, message: str) -> None:
        self.upscale_log.appendPlainText(str(message))

    def _normalise_comfy_url(self) -> str:
        url = self.upscale_url_edit.text().strip() or "http://127.0.0.1:8000"
        if "://" not in url:
            url = f"http://{url}"
        return url.rstrip("/")

    def _parse_upscale_extra_args(self) -> tuple[str, ...]:
        text = self.upscale_extra_args_edit.text().strip()
        if not text:
            return ()
        try:
            values = shlex.split(text, posix=False)
        except ValueError:
            values = text.split()
        return tuple(value.strip('"') for value in values)

    def _comfyui_is_running(self, url: str) -> bool:
        try:
            with urllib.request.urlopen(f"{url}/system_stats", timeout=3):
                pass
            return True
        except Exception:
            return False

    def _comfyui_launch_is_active(self) -> bool:
        watcher = self._comfyui_launch_watcher
        if watcher is None:
            return False
        if watcher.isRunning():
            return True
        process = watcher.process
        return process is not None and process.poll() is None

    def _on_browse_upscale_python(self) -> None:
        current = self.upscale_python_edit.text().strip()
        start = str(Path(current).parent) if current else str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select ComfyUI Python",
            start,
            "Executables (*.exe);;All Files (*)",
        )
        if path:
            self.upscale_python_edit.setText(path)

    def _on_browse_upscale_main(self) -> None:
        current = self.upscale_main_edit.text().strip()
        start = str(Path(current).parent) if current else str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select ComfyUI main.py",
            start,
            "Python Files (*.py);;All Files (*)",
        )
        if path:
            self.upscale_main_edit.setText(path)

    def _on_browse_upscale_output(self) -> None:
        current = self.upscale_output_edit.text().strip()
        if not current:
            default_path = self._default_upscale_output_path()
            current = str(default_path) if default_path else str(Path.home() / "upscaled.png")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Choose Upscale Output",
            current,
            "PNG (*.png)",
        )
        if path:
            output_path = Path(path)
            if output_path.suffix.lower() != ".png":
                output_path = output_path.with_suffix(".png")
            self.upscale_output_edit.setText(str(output_path))
            self._upscale_output_auto = False

    def _on_launch_comfyui(self) -> None:
        if self._comfyui_launch_is_active():
            return

        python = self.upscale_python_edit.text().strip()
        main_py = self.upscale_main_edit.text().strip()
        if not python or not Path(python).is_file():
            QtWidgets.QMessageBox.warning(self, "ComfyUI", "Choose a valid ComfyUI Python executable first.")
            return
        if not main_py or not Path(main_py).is_file():
            QtWidgets.QMessageBox.warning(self, "ComfyUI", "Choose a valid ComfyUI main.py first.")
            return

        url = self._normalise_comfy_url()
        self.upscale_url_edit.setText(url)
        if self._comfyui_is_running(url):
            self._append_upscale_log(f"ComfyUI is already running at {url}")
            self.upscale_status_label.setText("ComfyUI ready")
            return

        self.upscale_launch_btn.setEnabled(False)
        self.upscale_status_label.setText("Launching ComfyUI...")
        self._upscale_comfyui_stopping = False
        watcher = ComfyUILaunchWatcher(
            python=python,
            main_py=main_py,
            comfy_url=url,
            extra_args=self._parse_upscale_extra_args(),
        )
        watcher.setParent(self)
        self._comfyui_launch_watcher = watcher
        watcher.log.connect(self._append_upscale_log)
        watcher.completed.connect(self._on_comfyui_launch_completed)
        watcher.finished.connect(self._on_comfyui_launch_finished)
        watcher.start()

    def _on_comfyui_launch_completed(self, success: bool) -> None:
        self.upscale_status_label.setText("ComfyUI ready" if success else "ComfyUI launch failed")
        self._update_upscale_controls()

    def _on_comfyui_launch_finished(self) -> None:
        watcher = self._comfyui_launch_watcher
        if self._upscale_comfyui_stopping:
            self.upscale_status_label.setText("ComfyUI stopped")
        if watcher is not None and watcher.ready and not self._upscale_comfyui_stopping:
            self._update_upscale_controls()
            return
        self._comfyui_launch_watcher = None
        if watcher is not None:
            watcher.deleteLater()
        self._upscale_comfyui_stopping = False
        self._update_upscale_controls()

    def _on_stop_comfyui(self) -> None:
        watcher = self._comfyui_launch_watcher
        if watcher is None:
            self._append_upscale_log("No ComfyUI instance is managed by this app")
            return
        self._upscale_comfyui_stopping = True
        watcher.shutdown()
        self.upscale_status_label.setText("Stopping ComfyUI...")
        if not watcher.isRunning():
            self._comfyui_launch_watcher = None
            watcher.deleteLater()
            self._upscale_comfyui_stopping = False
            self.upscale_status_label.setText("ComfyUI stopped")
            self._update_upscale_controls()

    def _render_color_corrected_rgb8(self) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Load an image first")
        return process_rgb8_stack(
            self._loaded.original_rgb8,
            [self._base_params, self._effective_adjust_params()],
            self._strength,
        )

    def _render_upscale_source_rgb8(self) -> np.ndarray:
        return self._render_color_corrected_rgb8()

    def _render_censored_output_rgb8(self) -> np.ndarray:
        return apply_censor_blur(
            self._render_color_corrected_rgb8(),
            self._censor_circles,
            self._censor_blur_radius,
        )

    def _package_output_exists(self, paths: UpscalePackagePaths) -> list[Path]:
        return [
            path
            for path in (paths.censored_path, paths.archive_path, paths.upscaled_path)
            if path.exists()
        ]

    def _confirm_upscale_package_overwrite(self, paths: UpscalePackagePaths) -> bool:
        existing = self._package_output_exists(paths)
        if not existing:
            return True
        names = "\n".join(path.name for path in existing)
        answer = QtWidgets.QMessageBox.question(
            self,
            "Overwrite Package",
            f"These generated files already exist in:\n{paths.directory}\n\n{names}\n\nOverwrite them?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def _cleanup_upscale_package_temp(self) -> None:
        temp_path = self._upscale_package_censored_temp
        self._upscale_package_censored_temp = None
        if temp_path is None:
            return
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _reset_upscale_package_state(self) -> None:
        self._cleanup_upscale_package_temp()
        self._upscale_package_mode = False
        self._upscale_package_paths = None

    def _on_run_upscale(self) -> None:
        self._start_upscale(package=False)

    def _on_upscale_and_zip(self) -> None:
        self._start_upscale(package=True)

    def _start_upscale(self, package: bool) -> None:
        if self._upscale_worker is not None and self._upscale_worker.isRunning():
            QtWidgets.QMessageBox.information(self, "Upscale", "An upscale is already running.")
            return
        if not self._loaded:
            QtWidgets.QMessageBox.information(self, "Upscale", "Load an image first.")
            return

        package_paths: UpscalePackagePaths | None = None
        if package:
            if not self._censor_circles:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Upscale and Zip",
                    "Add at least one censor circle before creating a preview package.",
                )
                return
            package_paths = build_upscale_package_paths(self._loaded.path)
            if not self._confirm_upscale_package_overwrite(package_paths):
                return
            output_path = package_paths.upscaled_path
        else:
            output_text = self.upscale_output_edit.text().strip()
            if self._upscale_output_auto or not output_text:
                default_path = self._default_upscale_output_path()
                if default_path is None:
                    return
                output_path = self._unique_upscale_output_path(default_path)
                self.upscale_output_edit.setText(str(output_path))
                self._upscale_output_auto = True
            else:
                output_path = Path(output_text)
                if output_path.suffix.lower() != ".png":
                    QtWidgets.QMessageBox.warning(self, "Upscale", "The output file must use the .png extension.")
                    return
                try:
                    same_as_source = output_path.resolve() == self._loaded.path.resolve()
                except OSError:
                    same_as_source = output_path.absolute() == self._loaded.path.absolute()
                if same_as_source:
                    QtWidgets.QMessageBox.warning(self, "Upscale", "Choose an output path different from the source image.")
                    return
                if output_path.exists():
                    answer = QtWidgets.QMessageBox.question(
                        self,
                        "Overwrite Output",
                        f"Overwrite this file?\n\n{output_path}",
                        QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                        QtWidgets.QMessageBox.StandardButton.No,
                    )
                    if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                        return

        try:
            rendered = self._render_upscale_source_rgb8()
            with tempfile.NamedTemporaryFile(
                prefix="cb_color_correct_upscale_",
                suffix=".png",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
            save_metadata_free_rgb8(rendered, temp_path)
            censored_temp_path: Path | None = None
            if package:
                with tempfile.NamedTemporaryFile(
                    prefix="cb_color_correct_censored_",
                    suffix=self._loaded.path.suffix or ".png",
                    delete=False,
                ) as temp_file:
                    censored_temp_path = Path(temp_file.name)
                save_metadata_free_rgb8(self._render_censored_output_rgb8(), censored_temp_path)
        except Exception as exc:
            self._cleanup_upscale_source_temp()
            if "censored_temp_path" in locals() and censored_temp_path is not None:
                try:
                    censored_temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            QtWidgets.QMessageBox.critical(self, "Upscale", f"Could not prepare the image:\n\n{exc}")
            return

        self._upscale_source_temp = temp_path
        self._upscale_package_mode = package
        self._upscale_package_paths = package_paths
        self._upscale_package_censored_temp = censored_temp_path
        self.upscale_url_edit.setText(self._normalise_comfy_url())
        settings = UpscaleSettings(
            source_path=str(temp_path),
            output_path=str(output_path),
            comfy_url=self._normalise_comfy_url(),
            engine=self.upscale_engine_combo.currentText(),
            resize_type=self.upscale_rtx_resize_type_combo.currentText(),
            scale=self.upscale_rtx_scale_spin.value(),
            width=self.upscale_rtx_width_spin.value(),
            height=self.upscale_rtx_height_spin.value(),
            quality=self.upscale_rtx_quality_combo.currentText(),
            seedvr_resolution=self.upscale_seedvr_resolution_spin.value(),
            seedvr_max_resolution=self.upscale_seedvr_max_resolution_spin.value(),
            seedvr_seed=self.upscale_seedvr_seed_spin.value(),
            comfyui_python=self.upscale_python_edit.text().strip(),
            comfyui_main=self.upscale_main_edit.text().strip(),
            comfyui_extra_args=self._parse_upscale_extra_args(),
        )
        self.upscale_progress.setValue(0)
        self.upscale_status_label.setText("Starting upscale...")
        self._append_upscale_log(f"Starting {settings.engine} upscale for {self._loaded.path.name}")
        self._upscale_cancel_requested = False
        worker = ComfyUpscaleWorker(settings)
        worker.setParent(self)
        self._upscale_worker = worker
        worker.progress.connect(self.upscale_progress.setValue)
        worker.status.connect(self.upscale_status_label.setText)
        worker.log.connect(self._append_upscale_log)
        worker.completed.connect(self._on_upscale_completed)
        worker.finished.connect(worker.deleteLater)
        self._update_upscale_controls()
        worker.start()

    def _on_cancel_upscale(self) -> None:
        worker = self._upscale_worker
        if worker is None or not worker.isRunning():
            return
        self._upscale_cancel_requested = True
        worker.requestInterruption()
        self.upscale_cancel_btn.setEnabled(False)
        self.upscale_status_label.setText("Cancelling upscale...")

    def _on_upscale_completed(self, success: bool, output_path: str) -> None:
        self._cleanup_upscale_source_temp()

        package_mode = self._upscale_package_mode
        package_paths = self._upscale_package_paths
        censored_temp_path = self._upscale_package_censored_temp
        self._upscale_worker = None
        self._update_upscale_controls()
        if success:
            if package_mode:
                try:
                    if package_paths is None or censored_temp_path is None:
                        raise RuntimeError("Upscale package state is incomplete")
                    package_paths.directory.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(censored_temp_path, package_paths.censored_path)
                    create_upscale_zip(Path(output_path), package_paths.archive_path)
                    Path(output_path).unlink(missing_ok=True)
                except Exception as exc:
                    self._append_upscale_log(f"Package creation failed: {exc}")
                    self._reset_upscale_package_state()
                    self.upscale_progress.setValue(0)
                    self.upscale_status_label.setText("Package creation failed")
                    QtWidgets.QMessageBox.critical(
                        self,
                        "Upscale and Zip",
                        f"The upscale completed, but the package could not be created:\n\n{exc}",
                    )
                    self._upscale_cancel_requested = False
                    return

                self._reset_upscale_package_state()
                self.upscale_progress.setValue(1)
                self.upscale_status_label.setText("Upscale and ZIP complete")
                self._append_upscale_log(f"Censored preview saved to {package_paths.censored_path}")
                self._append_upscale_log(f"Uncensored upscale ZIP saved to {package_paths.archive_path}")
                QtWidgets.QMessageBox.information(
                    self,
                    "Upscale and Zip Complete",
                    f"Package folder:\n{package_paths.directory}\n\n"
                    f"Preview:\n{package_paths.censored_path.name}\n"
                    f"ZIP:\n{package_paths.archive_path.name}",
                )
                self._upscale_cancel_requested = False
                return

            self.upscale_progress.setValue(1)
            self.upscale_status_label.setText("Upscale complete")
            QtWidgets.QMessageBox.information(self, "Upscale Complete", f"Saved to:\n{output_path}")
        elif self._upscale_cancel_requested:
            self.upscale_progress.setValue(0)
            self.upscale_status_label.setText("Upscale cancelled")
        else:
            self.upscale_progress.setValue(0)
            self.upscale_status_label.setText("Upscale failed")
        self._reset_upscale_package_state()
        self._upscale_cancel_requested = False

    def _preview_censor_blur_radius(self) -> float:
        if not self._loaded:
            return float(self._censor_blur_radius)
        preview_width = self._loaded.preview_rgb8.shape[1]
        original_width = self._loaded.original_rgb8.shape[1]
        return float(self._censor_blur_radius) * preview_width / max(1, original_width)

    def _update_censor_controls(self) -> None:
        has_image = self._loaded is not None
        self.censor_toolbar.setEnabled(has_image)
        self.censor_remove_btn.setEnabled(bool(self._censor_circles))
        self.censor_clear_btn.setEnabled(bool(self._censor_circles))
        self.image_label.set_censor_circles(self._censor_circles)
        self.image_label.set_censor_enabled(self._censor_enabled and has_image)

    def _build_adjustment_widgets(self) -> None:
        # Tone group
        self.tone_group = QtWidgets.QGroupBox("Tone")
        self.tone_group.setCheckable(True)
        self.tone_group.setChecked(True)
        tone_form = QtWidgets.QFormLayout(self.tone_group)
        tone_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        tone_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
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
        color_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
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
        wb_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
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
        sh_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
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
        split_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
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
        fx_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        fx_form.setContentsMargins(6, 6, 6, 6)
        fx_form.setHorizontalSpacing(8)
        fx_form.setVerticalSpacing(4)

        self.clarity_slider, self.clarity_value = self._make_slider(-100, 100, 0, suffix="")
        self.dehaze_slider, self.dehaze_value = self._make_slider(-100, 100, 0, suffix="")
        self.vignette_slider, self.vignette_value = self._make_slider(-100, 100, 0, suffix="")
        self.vignette_mid_slider, self.vignette_mid_value = self._make_slider(0, 100, 50, suffix="")
        self.bloom_slider, self.bloom_value = self._make_slider(0, 100, 0, suffix="")

        fx_form.addRow("Clarity", self._hbox(self.clarity_slider, self.clarity_value))
        fx_form.addRow("Dehaze", self._hbox(self.dehaze_slider, self.dehaze_value))
        fx_form.addRow("Vignette", self._hbox(self.vignette_slider, self.vignette_value))
        fx_form.addRow("Vig Mid", self._hbox(self.vignette_mid_slider, self.vignette_mid_value))
        fx_form.addRow("Bloom", self._hbox(self.bloom_slider, self.bloom_value))

        # Levels group
        self.levels_group = QtWidgets.QGroupBox("Levels")
        self.levels_group.setCheckable(True)
        self.levels_group.setChecked(True)
        levels_form = QtWidgets.QFormLayout(self.levels_group)
        levels_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        levels_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
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
        self.bloom_slider.valueChanged.connect(self._on_adjust_change)
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
        self._record_history_if_changed()
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
        bloom = self.bloom_slider.value() / 100.0

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
        self.bloom_value.setText(f"{bloom:.2f}")
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
            bloom = 0.0
        if not self.levels_group.isChecked():
            black = 0.0
            white = 1.0
            gamma = 1.0

        # Always preserve stored curves; runtime application is controlled by
        # _effective_adjust_params() based on curves_group checked state.
        curve_points = self._adjust_params.curve_points
        curve_r = self._adjust_params.curve_points_r
        curve_g = self._adjust_params.curve_points_g
        curve_b = self._adjust_params.curve_points_b

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
            bloom=bloom,
            levels_black=black,
            levels_white=white,
            levels_gamma=gamma,
            curve_points=curve_points,
            curve_points_r=curve_r,
            curve_points_g=curve_g,
            curve_points_b=curve_b,
        )
        self._record_history_if_changed()
        self._schedule_apply()

    def _effective_adjust_params(self) -> FilterParams:
        adjust = self._adjust_params
        if not getattr(self, "curves_group", None) or self.curves_group.isChecked():
            return adjust
        return replace(
            adjust,
            curve_points=None,
            curve_points_r=None,
            curve_points_g=None,
            curve_points_b=None,
        )

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
            self.bloom_slider,
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
        self.bloom_slider.setValue(int(round(self._adjust_params.bloom * 100.0)))

        self.black_slider.setValue(int(round(self._adjust_params.levels_black * 100.0)))
        self.white_slider.setValue(int(round(self._adjust_params.levels_white * 100.0)))
        self.gamma_slider.setValue(int(round(self._adjust_params.levels_gamma * 100.0)))

        # Enable/disable optional groups based on state (important for preset load).
        self.wb_group.setChecked(bool(self._adjust_params.temperature or self._adjust_params.tint))
        self.sh_group.setChecked(bool(self._adjust_params.shadows or self._adjust_params.highlights))
        self.split_group.setChecked(bool(self._adjust_params.split_amount))
        self.effects_group.setChecked(
            bool(self._adjust_params.clarity or self._adjust_params.dehaze or self._adjust_params.vignette or self._adjust_params.bloom)
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
        self._record_history_if_changed()
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
        self._record_history_if_changed()
        self._schedule_apply()

    def _make_slider(self, min_v: int, max_v: int, value: int, suffix: str = "") -> tuple[QtWidgets.QSlider, QtWidgets.QLabel]:
        s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        s.setRange(min_v, max_v)
        s.setValue(value)
        s.setSingleStep(1)
        s.setMinimumWidth(180)
        lab = QtWidgets.QLabel("0")
        lab.setMinimumWidth(52)
        lab.setFixedWidth(52)
        lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        return s, lab

    def _set_adjustments_visible(self, visible: bool) -> None:
        self.adjust_sidebar.setVisible(visible)
        self.sidebar_toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow if visible else QtCore.Qt.ArrowType.RightArrow)
        # Keep professional/compact: narrow when collapsed, comfortable when expanded
        if visible:
            # ~70% of the previous expanded width.
            self.sidebar_container.setMinimumWidth(225)
            self.sidebar_container.setMaximumWidth(420)
        else:
            self.sidebar_container.setMinimumWidth(34)
            self.sidebar_container.setMaximumWidth(34)

    def _hbox(self, slider: QtWidgets.QSlider, label: QtWidgets.QLabel) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addWidget(slider, 1)  # stretch factor 1 allows slider to expand
        lay.addWidget(label, 0)   # stretch factor 0 keeps label fixed
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
            "pilgram_filter": p.pilgram_filter,
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
            pilgram_filter=(str(d.get("pilgram_filter")) if d.get("pilgram_filter") else None),
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
            pilgram_filter=adjust.pilgram_filter if adjust.pilgram_filter else base.pilgram_filter,
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

    def _display_source_pixmap(self) -> QtGui.QPixmap | None:
        if not self._base_pixmap or self._base_pixmap.isNull():
            return None
        if self._split_preview_enabled:
            split = self._make_split_preview_pixmap()
            if split is not None and not split.isNull():
                return split
        return self._base_pixmap

    def _make_split_preview_pixmap(self) -> QtGui.QPixmap | None:
        if not self._base_pixmap or self._base_pixmap.isNull():
            return None
        if not self._original_preview_pixmap or self._original_preview_pixmap.isNull():
            return None

        processed = self._base_pixmap
        original = self._original_preview_pixmap
        if original.size() != processed.size():
            original = original.scaled(
                processed.size(),
                QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )

        out = QtGui.QPixmap(processed.size())
        out.fill(QtCore.Qt.GlobalColor.black)
        split_x = int(round(processed.width() * self._split_divider_ratio))

        painter = QtGui.QPainter(out)
        painter.drawPixmap(0, 0, original)
        painter.setClipRect(split_x, 0, processed.width() - split_x, processed.height())
        painter.drawPixmap(0, 0, processed)
        painter.setClipping(False)
        painter.setPen(QtGui.QPen(QtGui.QColor(240, 240, 240, 220), 1))
        painter.drawLine(split_x, 0, split_x, processed.height())
        painter.end()

        return out

    def _refit_pixmap(self) -> None:
        src = self._display_source_pixmap()
        if not src or src.isNull():
            return
        viewport = self.scroll.viewport().size()
        if viewport.width() <= 10 or viewport.height() <= 10:
            return

        if self._zoom_mode == "fit":
            factor = self._fit_zoom_factor()
        else:
            factor = max(0.05, min(16.0, float(self._zoom_factor)))

        target_w = max(1, int(round(src.width() * factor)))
        target_h = max(1, int(round(src.height() * factor)))
        scaled = src.scaled(
            QtCore.QSize(target_w, target_h),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setText("")
        self.image_label.setPixmap(scaled)
        self.image_label.setFixedSize(scaled.size())
        self._set_zoom_label()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._save_upscale_settings()

        worker = self._upscale_worker
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait()

        watcher = self._comfyui_launch_watcher
        if watcher is not None and self._comfyui_launch_is_active():
            watcher.shutdown()
            if watcher.isRunning():
                watcher.wait()

        self._cleanup_upscale_source_temp()
        self._reset_upscale_package_state()
        super().closeEvent(event)


def main() -> int:
    validate_packages()
    
    app = QtWidgets.QApplication(sys.argv)

    apply_ableton_theme(app)

    w = MainWindow()
    w.resize(1200, 800)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
