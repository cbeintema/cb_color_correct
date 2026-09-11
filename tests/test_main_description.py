import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

import main
from cb_color_correct.censor import CensorCircle


class MemorySettings:
    def __init__(self):
        self.values = {}
    def value(self, key, default=None):
        return self.values.get(key, default)
    def setValue(self, key, value):
        self.values[key] = value
    def sync(self):
        pass


class FakeWorker(QtCore.QObject):
    status = QtCore.Signal(str)
    result = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    loaded = QtCore.Signal(str)
    finished = QtCore.Signal()
    def __init__(self, settings, action, snapshot, instance):
        super().__init__()
        self.settings, self.action, self.snapshot = settings, action, snapshot
        self.interrupted = False
    def start(self):
        pass
    def requestInterruption(self):
        self.interrupted = True
    def isInterruptionRequested(self):
        return self.interrupted


class MainDescriptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.settings = MemorySettings()
        with patch.object(main.QtCore, "QSettings", return_value=self.settings):
            self.window = main.MainWindow()
        self.panel = self.window.description_panel
        self.worker_patch = patch("cb_color_correct.description_panel.DescriptionWorker", FakeWorker)
        self.worker_patch.start()

    def tearDown(self):
        if self.panel.worker:
            self.panel.worker.finished.emit()
        self.window.close()
        self.app.processEvents()
        self.window.deleteLater()
        self.worker_patch.stop()

    def load_image(self):
        rgb = np.full((40, 60, 3), 120, dtype=np.uint8)
        self.window._loaded = main.LoadedImage(Path("test.png"), rgb, rgb)
        self.panel.has_image = True
        self.panel.invalidate(new_image=True)

    def test_disabled_without_image_and_copy_finished_text(self):
        self.assertFalse(self.panel.write_btn.isEnabled())
        self.load_image()
        self.panel.start("describe")
        self.assertFalse(self.panel.write_btn.isEnabled())
        self.assertFalse(self.window.upscale_run_btn.isEnabled())
        self.panel.worker.result.emit("A quiet landscape.")
        self.panel.worker.finished.emit()
        self.panel.output.insertPlainText("Edited: ")
        self.panel.copy()
        self.assertEqual(self.app.clipboard().text(), self.panel.output.toPlainText())
        self.assertTrue(self.panel.write_btn.isEnabled())

    def test_snapshot_independent_with_selected_censor(self):
        self.load_image()
        self.window._censor_circles = (CensorCircle(0.5, 0.5, 0.2),)
        self.panel.start("describe")
        snapshot = self.panel.worker.snapshot
        self.assertEqual(snapshot.circles, ())
        self.window._loaded.original_rgb8[:] = 0
        self.assertTrue(np.all(snapshot.rgb8 == 120))
        self.panel.worker.finished.emit()
        self.panel.source.setCurrentIndex(1)
        self.panel.start("describe")
        self.assertEqual(self.panel.worker.snapshot.circles, self.window._censor_circles)

    def test_pending_preview_does_not_cancel_new_description(self):
        self.load_image()
        self.window._schedule_apply()
        self.panel.start("describe")
        QtTest.QTest.qWait(40)
        self.assertFalse(self.panel.worker.interrupted)
        self.assertEqual(self.panel.request_revision, self.panel.revision)

    def test_edits_and_new_images_discard_old_results(self):
        self.load_image()
        self.panel.output.setPlainText("Previous text")
        self.panel.start("describe")
        worker = self.panel.worker
        self.window._schedule_apply()
        self.assertTrue(worker.interrupted)
        worker.result.emit("Wrong old result")
        self.assertEqual(self.panel.output.toPlainText(), "Previous text")
        self.panel.invalidate(new_image=True)
        worker.result.emit("Wrong image")
        self.assertEqual(self.panel.output.toPlainText(), "")

    def test_failure_preserves_previous_text(self):
        self.load_image()
        self.panel.output.setPlainText("Keep this draft")
        self.panel.start("describe")
        self.panel.worker.failed.emit("Server offline")
        self.panel.worker.finished.emit()
        self.assertEqual(self.panel.output.toPlainText(), "Keep this draft")
        self.assertIn("offline", self.panel.status.text())

    def test_close_waits_asynchronously_for_worker(self):
        self.load_image()
        self.panel.start("describe")
        event = QtGui.QCloseEvent()
        self.window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        self.assertTrue(self.panel.worker.interrupted)
        self.panel.worker.finished.emit()
        self.app.processEvents()
        self.assertFalse(self.panel.busy)

    def test_settings_exclude_token_and_output(self):
        self.panel.token.setText("secret")
        self.panel.output.setPlainText("private draft")
        self.panel.prompt.setPlainText("A short description")
        self.panel.save_settings()
        values = str(self.settings.values)
        self.assertNotIn("secret", values)
        self.assertNotIn("private draft", values)
        self.assertEqual(self.settings.values["description/prompt"], "A short description")

    def test_text_shortcuts_select_all_and_undo(self):
        self.window.show()
        self.window.sidebar_toggle.setChecked(True)
        self.window.sidebar_tabs.setCurrentWidget(self.window.description_scroll)
        self.panel.output.setFocus()
        self.app.processEvents()
        self.panel.output.setPlainText("A sample description")
        QtTest.QTest.keyClick(self.panel.output, QtCore.Qt.Key.Key_A, QtCore.Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(self.panel.output.textCursor().selectedText(), "A sample description")
        QtTest.QTest.keyClicks(self.panel.output, "replacement")
        QtTest.QTest.keyClick(self.panel.output, QtCore.Qt.Key.Key_Z, QtCore.Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(self.panel.output.toPlainText(), "A sample description")


if __name__ == "__main__":
    unittest.main()
