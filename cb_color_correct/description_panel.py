"""Editor panel for local descriptions, separate from the main image editor."""
from __future__ import annotations

from typing import Callable

from PySide6 import QtCore, QtWidgets

from .description import (DEFAULT_MODEL, DEFAULT_PROMPT, DescriptionSettings,
                          DescriptionWorker, ImageSnapshot, local_url)


class DescriptionPanel(QtWidgets.QWidget):
    busy_changed = QtCore.Signal()
    idle = QtCore.Signal()
    layout_changed = QtCore.Signal()

    def __init__(self, settings: QtCore.QSettings, snapshot: Callable[[bool], ImageSnapshot]) -> None:
        super().__init__()
        self.settings = settings
        self.snapshot = snapshot
        self.worker: DescriptionWorker | None = None
        self.revision = 0
        self.request_revision = 0
        self.has_image = False
        self.upscale_busy = False
        self.action = ""
        self.owned_instances: dict[tuple[str, str], str] = {}
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        row = QtWidgets.QHBoxLayout()
        self.write_btn = QtWidgets.QPushButton("Write description")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.copy_btn = QtWidgets.QPushButton("Copy")
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.setToolTip("Clear the description text")
        self.source = QtWidgets.QComboBox()
        self.source.addItems(["Edited artwork (without censor blur)", "Censored preview"])
        for widget in (self.write_btn, self.cancel_btn, self.copy_btn, self.clear_btn):
            row.addWidget(widget)
        layout.addLayout(row)
        layout.addWidget(self.source)
        self.output = QtWidgets.QPlainTextEdit()
        self.output.setPlaceholderText("Write a description, edit it here, then copy it into your DeviantArt post.")
        self.output.setMinimumHeight(180)
        layout.addWidget(self.output, 1)
        self.status = QtWidgets.QLabel("Uses LM Studio on this computer. Load an image to begin.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        toggle = QtWidgets.QToolButton()
        toggle.setText("Description settings")
        toggle.setCheckable(True)
        layout.addWidget(toggle)
        self.options = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.options)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setContentsMargins(0, 0, 0, 0)
        self.url = QtWidgets.QLineEdit(str(settings.value("description/url", "http://127.0.0.1:1234")))
        self.model = QtWidgets.QComboBox()
        self.model.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.model.setMinimumContentsLength(18)
        selected = str(settings.value("description/model", DEFAULT_MODEL))
        self.model.addItem("Qwen3.5 9B Defiant Fable (Q4_K_S)" if selected == DEFAULT_MODEL else selected, selected)
        self.refresh_btn = QtWidgets.QPushButton("Refresh models")
        self.load_btn = QtWidgets.QPushButton("Load model")
        self.unload_btn = QtWidgets.QPushButton("Unload app's model")
        self.unload_btn.setToolTip("Free the selected model loaded by this app before upscaling. Other model sessions are left alone.")
        buttons = QtWidgets.QGridLayout()
        buttons.addWidget(self.refresh_btn, 0, 0)
        buttons.addWidget(self.load_btn, 0, 1)
        buttons.addWidget(self.unload_btn, 1, 0, 1, 2)
        self.token = QtWidgets.QLineEdit()
        self.token.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.token.setPlaceholderText("Optional LM Studio API token (not saved)")
        self.prompt = QtWidgets.QPlainTextEdit(str(settings.value("description/prompt", DEFAULT_PROMPT)))
        self.prompt.setMaximumHeight(85)
        self.context = QtWidgets.QSpinBox()
        self.context.setRange(2048, 32768)
        self.context.setSingleStep(1024)
        try:
            self.context.setValue(int(str(settings.value("description/context", 8192))))
        except ValueError:
            self.context.setValue(8192)
        self.context.setToolTip("Used when loading a model. Existing instances retain their LM Studio settings.")
        form.addRow("Local server", self.url)
        form.addRow("Vision model", self.model)
        form.addRow(buttons)
        form.addRow("API token", self.token)
        form.addRow("Context tokens", self.context)
        form.addRow("Writing instructions", self.prompt)
        layout.addWidget(self.options)
        self.options.hide()
        toggle.toggled.connect(self.options.setVisible)
        toggle.toggled.connect(lambda _checked: self.layout_changed.emit())
        self.write_btn.clicked.connect(lambda: self.start("describe"))
        self.refresh_btn.clicked.connect(lambda: self.start("refresh"))
        self.load_btn.clicked.connect(lambda: self.start("load"))
        self.unload_btn.clicked.connect(lambda: self.start("unload"))
        self.cancel_btn.clicked.connect(self.cancel)
        self.copy_btn.clicked.connect(self.copy)
        self.clear_btn.clicked.connect(self.clear)
        self.output.textChanged.connect(self.update_controls)
        self.source.currentIndexChanged.connect(self.invalidate)
        self.model.currentIndexChanged.connect(self.update_controls)
        self.url.textChanged.connect(self.update_controls)
        self.update_controls()

    @property
    def busy(self) -> bool:
        # Retain ownership until the thread's finished signal has been processed.
        return self.worker is not None

    def save_settings(self) -> None:
        for name, value in (("url", self.url.text().strip()), ("model", self.model.currentData()),
                            ("prompt", self.prompt.toPlainText()), ("context", self.context.value())):
            self.settings.setValue("description/" + name, value)

    def connection_key(self) -> tuple[str, str]:
        return local_url(self.url.text()), self.model.currentData() or ""

    def update_controls(self, *_args) -> None:
        available = not self.busy and not self.upscale_busy
        self.write_btn.setEnabled(self.has_image and available)
        self.cancel_btn.setEnabled(self.busy and not self.worker.isInterruptionRequested())
        self.copy_btn.setEnabled(bool(self.output.toPlainText().strip()))
        self.clear_btn.setEnabled(not self.busy and not self.output.document().isEmpty())
        self.options.setEnabled(available)
        self.source.setEnabled(not self.busy)
        try:
            owned = self.connection_key() in self.owned_instances
        except ValueError:
            owned = False
        except RuntimeError:
            owned = False
        self.unload_btn.setEnabled(available and owned)

    def invalidate(self, *_args, new_image: bool = False) -> None:
        self.revision += 1
        if new_image:
            self.output.clear()
        if self.busy and self.action == "describe":
            self.cancel()
            self.status.setText("Image changed; previous request cancelled. Write a new description when ready.")
        elif self.output.toPlainText().strip():
            self.status.setText("Image or source changed. This description may be out of date.")
        elif self.has_image:
            self.status.setText("Ready to describe the selected image source.")
        self.update_controls()

    def copy(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self.output.toPlainText())

    def clear(self) -> None:
        self.output.clear()
        self.status.setText("Description cleared.")

    def start(self, action: str) -> None:
        if self.busy or self.upscale_busy or (action == "describe" and not self.has_image):
            return
        try:
            key = self.connection_key()
            prompt = self.prompt.toPlainText().strip()
            if action == "describe" and not prompt:
                self.status.setText("Enter writing instructions in Description settings.")
                return
            settings = DescriptionSettings(key[0], key[1], self.token.text().strip(), prompt, self.context.value())
            snapshot = self.snapshot(self.source.currentIndex() == 1) if action == "describe" else None
            worker = DescriptionWorker(settings, action, snapshot, self.owned_instances.get(key, ""))
        except Exception as exc:
            self.status.setText(str(exc))
            return
        self.save_settings()
        self.worker = worker
        self.action = action
        self.request_revision = self.revision
        self.status.setText("Connecting to LM Studio…")
        # Preserve the previous text until a complete, current result arrives.
        self.output.setReadOnly(action == "describe")
        worker.status.connect(self.on_status)
        worker.loaded.connect(self.on_loaded)
        worker.result.connect(self.on_result)
        worker.failed.connect(self.on_failed)
        worker.finished.connect(self.on_finished)
        self.update_controls()
        self.busy_changed.emit()
        worker.start()

    def on_loaded(self, instance: str) -> None:
        if self.worker is not None:
            settings = self.worker.settings
            self.owned_instances[(settings.url, settings.model)] = instance

    def on_status(self, status: str) -> None:
        if self.worker is not None and not self.worker.isInterruptionRequested():
            self.status.setText(status)

    def on_failed(self, error: str) -> None:
        if self.action != "describe" or self.request_revision == self.revision:
            self.status.setText(error)

    def on_result(self, result: object) -> None:
        if self.worker is None or self.worker.isInterruptionRequested():
            return
        if self.action == "describe":
            if self.request_revision != self.revision:
                return
            self.output.setPlainText(str(result))
            self.status.setText("Description ready. Edit it if needed, then Copy to your DeviantArt post.")
        elif self.action == "refresh":
            selected = self.model.currentData()
            self.model.clear()
            for model in result:
                quant = (model.get("quantization") or {}).get("name", "")
                self.model.addItem(f"{model.get('display_name', model['key'])} ({quant})", model["key"])
            index = self.model.findData(selected)
            if index < 0:
                index = self.model.findData(DEFAULT_MODEL)
            if index >= 0:
                self.model.setCurrentIndex(index)
            self.status.setText("Vision models refreshed." if result else "No vision models found. Check the model and mmproj in LM Studio.")
        elif self.action == "unload":
            self.owned_instances.pop(self.connection_key(), None)
            self.status.setText("Model unloaded. GPU memory is available for other work.")
        else:
            self.status.setText("Model ready. Write description when an image is loaded.")

    def cancel(self) -> None:
        if self.worker is not None:
            self.worker.requestInterruption()
            self.status.setText("Cancelling…")
            self.update_controls()

    def on_finished(self) -> None:
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.deleteLater()
        self.output.setReadOnly(False)
        self.update_controls()
        self.busy_changed.emit()
        self.idle.emit()
