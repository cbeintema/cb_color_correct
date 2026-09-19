"""Small standalone result window for Explorer's metadata check."""
import sys
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from cb_color_correct.metadata_check import SCOPE, check_file


class CheckWorker(QtCore.QThread):
    result_ready = QtCore.Signal(object)

    def __init__(self, path, parent):
        super().__init__(parent)
        self.path = path

    def run(self):
        self.result_ready.emit(check_file(self.path))


class ResultWindow(QtWidgets.QDialog):
    def __init__(self, path):
        super().__init__()
        self.setWindowTitle("CB Color Correct — Metadata Check")
        self.resize(640, 420)
        layout = QtWidgets.QVBoxLayout(self)
        self.heading = QtWidgets.QLabel("Checking metadata…")
        font = self.heading.font()
        font.setPointSize(15)
        font.setBold(True)
        self.heading.setFont(font)
        layout.addWidget(self.heading)
        filename = QtWidgets.QLabel(str(path))
        filename.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        filename.setWordWrap(True)
        layout.addWidget(filename)
        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details)
        scope = QtWidgets.QLabel(SCOPE)
        scope.setWordWrap(True)
        layout.addWidget(scope)
        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        layout.addWidget(self.close_button)
        self.worker = CheckWorker(path, self)
        self.worker.result_ready.connect(self.show_result)
        self.worker.start()

    def show_result(self, result):
        self.heading.setText(result.title)
        lines = result.entries + [f"Check issue: {issue}" for issue in result.issues]
        self.details.setPlainText("\n\n".join(lines) or "No embedded metadata was detected in the checked fields.")

    def reject(self):
        if not self.worker.isRunning():
            super().reject()

    def closeEvent(self, event):
        if self.worker.isRunning():
            event.ignore()
        else:
            super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    if len(sys.argv) != 2:
        QtWidgets.QMessageBox.information(None, "Metadata Check", "Right-click an image or video and choose Check metadata.")
        raise SystemExit(1)
    window = ResultWindow(Path(sys.argv[1]))
    window.show()
    raise SystemExit(app.exec())
