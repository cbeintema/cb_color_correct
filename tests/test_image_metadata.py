import unittest
from pathlib import Path
import tempfile

from PIL import Image, PngImagePlugin

from cb_color_correct.image_metadata import save_metadata_free_bytes, save_metadata_free_rgb8
import numpy as np


class ImageMetadataTests(unittest.TestCase):
    def test_png_bytes_are_reencoded_without_text_or_exif_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_path = root / "source.png"
            clean_path = root / "clean.png"
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("Author", "private@example.test")
            metadata.add_text("Comment", "private metadata")
            Image.new("RGB", (4, 4), (20, 40, 60)).save(source_path, pnginfo=metadata)

            save_metadata_free_bytes(source_path.read_bytes(), clean_path)

            with Image.open(clean_path) as clean:
                self.assertNotIn("Author", clean.info)
                self.assertNotIn("Comment", clean.info)
                self.assertEqual(len(clean.getexif()), 0)

    def test_rgb8_save_has_no_exif_or_icc_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "clean.jpg"
            save_metadata_free_rgb8(np.zeros((4, 4, 3), dtype=np.uint8), output_path)

            with Image.open(output_path) as clean:
                self.assertNotIn("exif", clean.info)
                self.assertNotIn("icc_profile", clean.info)
                self.assertNotIn("photoshop", clean.info)
                self.assertNotIn("xmp", clean.info)
                self.assertEqual(len(clean.getexif()), 0)


if __name__ == "__main__":
    unittest.main()
