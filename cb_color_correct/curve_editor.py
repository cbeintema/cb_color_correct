from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

from PySide6 import QtCore, QtGui, QtWidgets


Point = tuple[float, float]


_SCENE_SIZE = 1000.0
_POINT_RADIUS = 10.0
_EPS = 1e-4


class _ControlPointItem(QtWidgets.QGraphicsEllipseItem):
    def __init__(
        self,
        editor: "CurveEditor",
        index: int,
        movable: bool,
        rect: QtCore.QRectF,
    ) -> None:
        super().__init__(rect)
        self._editor = editor
        self._index = index
        self.setBrush(QtGui.QBrush(QtGui.QColor(210, 210, 210)))
        self.setPen(QtGui.QPen(QtGui.QColor(20, 20, 20), 1))
        self.setZValue(10)

        flags = QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        if movable:
            flags |= QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            flags |= QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        self.setFlags(flags)

    def itemChange(self, change: QtWidgets.QGraphicsItem.GraphicsItemChange, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            p: QtCore.QPointF = value
            return self._editor._clamp_item_pos(self._index, p)
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._editor._on_item_moved(self._index)
        return super().itemChange(change, value)


class CurveEditor(QtWidgets.QWidget):
    pointsChanged = QtCore.Signal(list)  # list[(x,y)] normalized

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self._points: list[Point] = [(0.0, 0.0), (1.0, 1.0)]
        self._suppress_emit = False

        self._emit_timer = QtCore.QTimer(self)
        self._emit_timer.setSingleShot(True)
        self._emit_timer.timeout.connect(self._emit_points)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.view = QtWidgets.QGraphicsView()
        self.view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setMinimumHeight(170)

        self.scene = QtWidgets.QGraphicsScene(self)
        self.scene.setSceneRect(0.0, 0.0, _SCENE_SIZE, _SCENE_SIZE)
        self.view.setScene(self.scene)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(8)
        self.reset_btn = QtWidgets.QPushButton("Reset")
        self.scurve_btn = QtWidgets.QPushButton("S-Curve")
        btn_row.addWidget(self.reset_btn)
        btn_row.addWidget(self.scurve_btn)
        btn_row.addStretch(1)

        layout.addWidget(self.view)
        layout.addLayout(btn_row)

        self.reset_btn.clicked.connect(self.set_identity)
        self.scurve_btn.clicked.connect(self.set_scurve)

        self._grid_items: list[QtWidgets.QGraphicsItem] = []
        self._curve_path_item: QtWidgets.QGraphicsPathItem | None = None
        self._point_items: list[_ControlPointItem] = []

        self.set_identity()

    def points(self) -> list[Point]:
        return list(self._points)

    def set_points(self, points: Sequence[Point], *, emit: bool = True) -> None:
        pts = [(float(x), float(y)) for x, y in points]
        pts = [(max(0.0, min(1.0, x)), max(0.0, min(1.0, y))) for x, y in pts]
        pts.sort(key=lambda p: p[0])

        if not pts:
            pts = [(0.0, 0.0), (1.0, 1.0)]

        # Ensure endpoints
        if pts[0][0] != 0.0:
            pts.insert(0, (0.0, pts[0][1]))
        if pts[-1][0] != 1.0:
            pts.append((1.0, pts[-1][1]))
        pts[0] = (0.0, 0.0)
        pts[-1] = (1.0, 1.0)

        # Remove near-duplicates in x
        cleaned: list[Point] = [pts[0]]
        for x, y in pts[1:]:
            if x - cleaned[-1][0] < 0.001:
                continue
            cleaned.append((x, y))

        # Cap count (keep endpoints + up to 4 internal)
        if len(cleaned) > 6:
            internal = cleaned[1:-1]
            # keep evenly spaced
            keep = [internal[int(i * (len(internal) - 1) / 3)] for i in range(4)]
            cleaned = [cleaned[0], *keep, cleaned[-1]]

        self._points = cleaned
        self._rebuild_scene()
        if emit and not self._suppress_emit:
            self._schedule_emit()

    def set_identity(self, *, emit: bool = True) -> None:
        self.set_points([(0.0, 0.0), (0.33, 0.33), (0.66, 0.66), (1.0, 1.0)], emit=emit)

    def set_scurve(self, *, emit: bool = True) -> None:
        self.set_points([(0.0, 0.0), (0.25, 0.18), (0.5, 0.5), (0.75, 0.82), (1.0, 1.0)], emit=emit)

    def _schedule_emit(self) -> None:
        self._emit_timer.start(10)

    def _emit_points(self) -> None:
        self.pointsChanged.emit(self.points())

    def _rebuild_scene(self) -> None:
        self.scene.clear()
        self._grid_items.clear()
        self._point_items.clear()
        self._curve_path_item = None

        self._draw_grid()
        self._draw_curve()
        self._draw_points()

        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def _draw_grid(self) -> None:
        pen_minor = QtGui.QPen(QtGui.QColor(55, 55, 55), 1)
        pen_major = QtGui.QPen(QtGui.QColor(75, 75, 75), 1)

        steps = 10
        for i in range(steps + 1):
            x = _SCENE_SIZE * (i / steps)
            pen = pen_major if i in (0, steps, steps // 2) else pen_minor
            self.scene.addLine(x, 0.0, x, _SCENE_SIZE, pen)
            self.scene.addLine(0.0, x, _SCENE_SIZE, x, pen)

        axis_pen = QtGui.QPen(QtGui.QColor(110, 110, 110), 2)
        self.scene.addRect(0.0, 0.0, _SCENE_SIZE, _SCENE_SIZE, axis_pen)

    def _draw_curve(self) -> None:
        path = QtGui.QPainterPath()
        pts_scene = [self._to_scene(p) for p in self._points]
        if pts_scene:
            path.moveTo(pts_scene[0])
            for p in pts_scene[1:]:
                path.lineTo(p)
        item = QtWidgets.QGraphicsPathItem(path)
        item.setPen(QtGui.QPen(QtGui.QColor(60, 120, 200), 3))
        item.setZValue(5)
        self.scene.addItem(item)
        self._curve_path_item = item

    def _draw_points(self) -> None:
        for i, p in enumerate(self._points):
            movable = i not in (0, len(self._points) - 1)
            rect = QtCore.QRectF(-_POINT_RADIUS, -_POINT_RADIUS, _POINT_RADIUS * 2, _POINT_RADIUS * 2)
            item = _ControlPointItem(self, i, movable=movable, rect=rect)
            item.setPos(self._to_scene(p))
            if not movable:
                item.setBrush(QtGui.QBrush(QtGui.QColor(140, 140, 140)))
            self.scene.addItem(item)
            self._point_items.append(item)

    def _on_item_moved(self, index: int) -> None:
        self._sync_points_from_items()
        self._update_curve_path()
        self._schedule_emit()

    def _sync_points_from_items(self) -> None:
        pts: list[Point] = []
        for item in self._point_items:
            pts.append(self._from_scene(item.pos()))
        pts.sort(key=lambda p: p[0])
        if pts:
            pts[0] = (0.0, 0.0)
            pts[-1] = (1.0, 1.0)
        self._points = pts

    def _update_curve_path(self) -> None:
        if not self._curve_path_item:
            return
        path = QtGui.QPainterPath()
        pts_scene = [self._to_scene(p) for p in self._points]
        if pts_scene:
            path.moveTo(pts_scene[0])
            for p in pts_scene[1:]:
                path.lineTo(p)
        self._curve_path_item.setPath(path)

    def _to_scene(self, p: Point) -> QtCore.QPointF:
        x, y = p
        sx = x * _SCENE_SIZE
        sy = (1.0 - y) * _SCENE_SIZE
        return QtCore.QPointF(sx, sy)

    def _from_scene(self, p: QtCore.QPointF) -> Point:
        x = float(p.x()) / _SCENE_SIZE
        y = 1.0 - (float(p.y()) / _SCENE_SIZE)
        return (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))

    def _clamp_item_pos(self, index: int, pos: QtCore.QPointF) -> QtCore.QPointF:
        # Clamp within bounds.
        x = float(pos.x())
        y = float(pos.y())
        x = max(0.0, min(_SCENE_SIZE, x))
        y = max(0.0, min(_SCENE_SIZE, y))

        # Endpoints are fixed.
        if index == 0:
            return self._to_scene((0.0, 0.0))
        if index == len(self._points) - 1:
            return self._to_scene((1.0, 1.0))

        # Prevent crossing neighbors in X.
        # During scene construction, the item list may not yet contain the neighbors.
        if 0 <= index - 1 < len(self._point_items):
            left = self._point_items[index - 1].pos().x() + 1.0
        else:
            left = 0.0

        if 0 <= index + 1 < len(self._point_items):
            right = self._point_items[index + 1].pos().x() - 1.0
        else:
            right = _SCENE_SIZE

        x = max(left, min(right, x))
        return QtCore.QPointF(x, y)
