import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, PngImagePlugin, TiffImagePlugin

from cb_color_correct.metadata_check import check_file, parse_video


class MetadataCheckTests(unittest.TestCase):
    def test_clean_images_and_tagged_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for suffix in ("png", "jpg", "tiff", "webp", "bmp"):
                path = root / f"clean.{suffix}"
                Image.new("RGB", (8, 8)).save(path)
                result = check_file(path)
                self.assertEqual(result.issues, [], suffix)
                self.assertEqual(result.entries, [], suffix)
            text = PngImagePlugin.PngInfo()
            text.add_text("Author", "Test author")
            path = root / "tagged.png"
            Image.new("RGB", (8, 8)).save(path, pnginfo=text)
            before = path.read_bytes()
            self.assertIn("Test author", " ".join(check_file(path).entries))
            self.assertEqual(path.read_bytes(), before)
            exif = Image.Exif()
            exif[315] = "Test artist"
            path = root / "tagged.jpg"
            Image.new("RGB", (8, 8)).save(path, exif=exif)
            self.assertIn("Test artist", " ".join(check_file(path).entries))

    def test_later_tiff_page_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pages.tiff"
            first = Image.new("RGB", (8, 8))
            second = Image.new("RGB", (8, 8))
            with TiffImagePlugin.AppendingTiffWriter(str(path), True) as writer:
                first.save(writer, format="TIFF")
                writer.newFrame()
                second.save(writer, format="TIFF", tiffinfo={315: "Second page artist"})
                writer.newFrame()
            self.assertIn("Second page artist", " ".join(check_file(path).entries))

    def test_video_tags_and_structural_fields(self):
        data = {"format": {"tags": {"major_brand": "isom"}},
                "streams": [{"index": 0, "codec_type": "video", "width": 100}]}
        self.assertEqual(parse_video(data).entries, [])
        data["format"]["tags"]["location"] = "+12.34+56.78/"
        data["streams"][0]["tags"] = {"encoder": "Test encoder"}
        data["chapters"] = [{"tags": {"title": "Private title"}}]
        entries = " ".join(parse_video(data).entries)
        for value in ("location", "Test encoder", "Private title"):
            self.assertIn(value, entries)

    def test_invalid_file_is_not_clean(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "broken.png"
            path.write_bytes(b"not an image")
            self.assertTrue(check_file(path).issues)
            self.assertNotIn("Clean", check_file(path).title)

    def test_mp4_default_labels_are_structural_but_custom_labels_are_metadata(self):
        data = {"format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
                "streams": [{"codec_type": "video", "tags": {
                    "language": "und", "handler_name": "VideoHandler", "vendor_id": "[0][0][0][0]"}}]}
        self.assertEqual(parse_video(data).entries, [])
        for key, value in (("handler_name", "Private editor"), ("language", "eng"), ("vendor_id", "APPL")):
            data["streams"][0]["tags"][key] = value
            self.assertIn(value, " ".join(parse_video(data).entries))

    def test_probe_failures_are_not_clean(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "video & spaces.mp4"
            path.touch()
            with patch("cb_color_correct.metadata_check.shutil.which", return_value=None):
                self.assertTrue(check_file(path).issues)
            with patch("cb_color_correct.metadata_check.shutil.which", return_value="ffprobe"), patch(
                "cb_color_correct.metadata_check.subprocess.run"
            ) as run:
                run.side_effect = subprocess.TimeoutExpired("ffprobe", 15)
                self.assertTrue(check_file(path).issues)
                run.side_effect = None
                run.return_value = subprocess.CompletedProcess([], 0, json.dumps({}), "")
                self.assertTrue(check_file(path).issues)
                run.return_value = subprocess.CompletedProcess([], 0, json.dumps({
                    "format": {"format_name": "mp4"}, "streams": [{"codec_type": "video"}]
                }), "")
                self.assertFalse(check_file(path).issues)
                self.assertEqual(run.call_args.args[0][-1], str(path.resolve()))


if __name__ == "__main__":
    unittest.main()
