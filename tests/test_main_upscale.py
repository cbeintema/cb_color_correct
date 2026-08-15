import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PySide6 import QtWidgets

import main
from cb_color_correct.censor import CensorCircle
from cb_color_correct.image_ops import process_rgb8_stack
from cb_color_correct.upscale import build_upscale_package_paths


class MainUpscaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.window = main.MainWindow()
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[8:16, 8:16] = 255
        self.window._loaded = main.LoadedImage(
            path=Path("source.jpg"),
            original_rgb8=image,
            preview_rgb8=image,
        )

    def tearDown(self) -> None:
        self.window.close()
        self.app.processEvents()

    def test_upscale_source_ignores_censor_regions(self) -> None:
        self.window._censor_circles = (CensorCircle(0.5, 0.5, 0.25),)

        expected = process_rgb8_stack(
            self.window._loaded.original_rgb8,
            [self.window._base_params, self.window._effective_adjust_params()],
            self.window._strength,
        )

        np.testing.assert_array_equal(self.window._render_upscale_source_rgb8(), expected)

    def test_session_settings_restore_censor_and_upscale_values(self) -> None:
        class MemorySettings:
            def __init__(self) -> None:
                self.values: dict[str, object] = {}

            def value(self, key: str, default: object = None) -> object:
                return self.values.get(key, default)

            def setValue(self, key: str, value: object) -> None:
                self.values[key] = value

            def sync(self) -> None:
                pass

        settings = MemorySettings()
        self.window._settings = settings
        self.window._on_censor_blur_changed(37)
        self.window.upscale_engine_combo.setCurrentText("SeedVR")
        self.window.upscale_seedvr_resolution_spin.setValue(6144)
        self.window.upscale_seedvr_max_resolution_spin.setValue(8192)
        self.window.upscale_seedvr_seed_spin.setValue(123)
        self.window.upscale_rtx_scale_spin.setValue(3.5)
        self.window._save_upscale_settings()

        with patch.object(main.QtCore, "QSettings", return_value=settings):
            restored = main.MainWindow()
        try:
            self.assertEqual(restored.censor_blur_spin.value(), 37)
            self.assertEqual(restored.upscale_engine_combo.currentText(), "SeedVR")
            self.assertEqual(restored.upscale_seedvr_resolution_spin.value(), 6144)
            self.assertEqual(restored.upscale_seedvr_max_resolution_spin.value(), 8192)
            self.assertEqual(restored.upscale_seedvr_seed_spin.value(), 123)
            self.assertEqual(restored.upscale_rtx_scale_spin.value(), 3.5)
        finally:
            restored.close()

    def test_save_suggests_censored_suffix_when_regions_exist(self) -> None:
        self.window._censor_circles = (CensorCircle(0.5, 0.5, 0.25),)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "saved.png"
            with patch.object(
                QtWidgets.QFileDialog,
                "getSaveFileName",
                return_value=(str(output_path), "PNG (*.png)"),
            ) as chooser:
                self.window._on_save()

            self.assertEqual(Path(chooser.call_args.args[2]).name, "source_censored.jpg")

    def test_save_keeps_filtered_suffix_without_regions(self) -> None:
        with patch.object(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            return_value=("", ""),
        ) as chooser:
            self.window._on_save()

        self.assertEqual(Path(chooser.call_args.args[2]).name, "source_filtered.jpg")

    def test_package_completion_saves_preview_and_archives_upscale(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "portrait.jpg"
            paths = build_upscale_package_paths(source_path, "delivery")
            paths.upscaled_path.parent.mkdir(parents=True)
            paths.upscaled_path.write_bytes(b"uncensored upscale")
            censored_temp_path = root / "censored-temp.jpg"
            censored_temp_path.write_bytes(b"censored preview")

            self.window._upscale_package_mode = True
            self.window._upscale_package_paths = paths
            self.window._upscale_package_censored_temp = censored_temp_path

            with patch.object(QtWidgets.QMessageBox, "information"):
                self.window._on_upscale_completed(True, str(paths.upscaled_path))

            self.assertEqual(paths.censored_path.read_bytes(), b"censored preview")
            self.assertFalse(paths.upscaled_path.exists())
            with zipfile.ZipFile(paths.archive_path) as archive:
                self.assertEqual(archive.namelist(), ["delivery_upscaled.png"])
                self.assertEqual(archive.read("delivery_upscaled.png"), b"uncensored upscale")
            self.assertFalse(censored_temp_path.exists())


if __name__ == "__main__":
    unittest.main()