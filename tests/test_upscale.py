import unittest
from pathlib import Path
import tempfile
import zipfile

from cb_color_correct.upscale import (
    build_rtx_workflow,
    build_upscale_package_paths,
    build_seedvr_workflow,
    comfy_ws_url,
    create_upscale_zip,
)


class UpscaleWorkflowTests(unittest.TestCase):
    def test_seedvr_workflow_contains_expected_nodes_and_values(self) -> None:
        workflow = build_seedvr_workflow("input.png", 4096, 6144, 123)

        self.assertEqual(workflow["10"]["class_type"], "SeedVR2VideoUpscaler")
        self.assertEqual(workflow["10"]["inputs"]["resolution"], 4096)
        self.assertEqual(workflow["10"]["inputs"]["max_resolution"], 6144)
        self.assertEqual(workflow["10"]["inputs"]["seed"], 123)
        self.assertEqual(workflow["16"]["inputs"]["image"], "input.png")
        self.assertEqual(workflow["17"]["inputs"]["image"], ["16", 0])

    def test_rtx_scale_workflow_uses_multiplier(self) -> None:
        workflow = build_rtx_workflow("input.png", "scale by multiplier", 2.5, 3840, 2160, "HIGH")
        inputs = workflow["1"]["inputs"]

        self.assertEqual(workflow["1"]["class_type"], "RTXVideoSuperResolution")
        self.assertEqual(inputs["resize_type"], "scale by multiplier")
        self.assertEqual(inputs["resize_type.scale"], 2.5)
        self.assertNotIn("resize_type.width", inputs)
        self.assertEqual(inputs["quality"], "HIGH")

    def test_rtx_dimensions_workflow_uses_width_and_height(self) -> None:
        workflow = build_rtx_workflow(
            "input.png",
            "resize to width and height",
            2.0,
            3840,
            2160,
            "ULTRA",
        )
        inputs = workflow["1"]["inputs"]

        self.assertEqual(inputs["resize_type.width"], 3840)
        self.assertEqual(inputs["resize_type.height"], 2160)
        self.assertNotIn("resize_type.scale", inputs)

    def test_comfy_ws_url_normalizes_http_and_https(self) -> None:
        self.assertEqual(comfy_ws_url("127.0.0.1:8000"), "ws://127.0.0.1:8000")
        self.assertEqual(comfy_ws_url("http://localhost:8188/"), "ws://localhost:8188")
        self.assertEqual(comfy_ws_url("https://example.test:443"), "wss://example.test:443")

    def test_package_paths_use_source_stem_folder_and_names(self) -> None:
        paths = build_upscale_package_paths(r"G:\images\portrait.jpg")

        self.assertEqual(paths.directory, Path(r"G:\images\portrait"))
        self.assertEqual(paths.censored_path, Path(r"G:\images\portrait\portrait_censored.jpg"))
        self.assertEqual(paths.archive_path, Path(r"G:\images\portrait\portrait.zip"))
        self.assertEqual(paths.upscaled_path, Path(r"G:\images\portrait\portrait_upscaled.png"))

    def test_package_paths_use_optional_output_name(self) -> None:
        paths = build_upscale_package_paths(r"G:\images\portrait.jpg", "delivery")

        self.assertEqual(paths.directory, Path(r"G:\images\delivery"))
        self.assertEqual(paths.censored_path, Path(r"G:\images\delivery\delivery_censored.jpg"))
        self.assertEqual(paths.archive_path, Path(r"G:\images\delivery\delivery.zip"))
        self.assertEqual(paths.upscaled_path, Path(r"G:\images\delivery\delivery_upscaled.png"))

    def test_package_paths_reject_output_name_paths(self) -> None:
        with self.assertRaises(ValueError):
            build_upscale_package_paths(r"G:\images\portrait.jpg", "nested/delivery")

    def test_create_upscale_zip_contains_only_the_upscaled_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            upscaled_path = root / "portrait_upscaled.png"
            archive_path = root / "portrait.zip"
            upscaled_path.write_bytes(b"uncensored upscale")

            create_upscale_zip(upscaled_path, archive_path)

            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(archive.namelist(), ["portrait_upscaled.png"])
                self.assertEqual(archive.read("portrait_upscaled.png"), b"uncensored upscale")
                self.assertEqual(archive.comment, b"")
                entry = archive.infolist()[0]
                self.assertEqual(entry.date_time, (1980, 1, 1, 0, 0, 0))
                self.assertEqual(entry.extra, b"")
                self.assertEqual(entry.comment, b"")
                self.assertEqual(entry.external_attr, 0)


if __name__ == "__main__":
    unittest.main()