import unittest

import numpy as np

from cb_color_correct.censor import CensorCircle, apply_censor_blur


class CensorBlurTests(unittest.TestCase):
    def test_no_circles_returns_identical_pixels(self) -> None:
        image = np.arange(32 * 24 * 3, dtype=np.uint8).reshape((24, 32, 3))

        result = apply_censor_blur(image, (), 24)

        self.assertTrue(np.array_equal(result, image))
        self.assertEqual(result.dtype, np.uint8)

    def test_pixels_outside_circle_are_unchanged(self) -> None:
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        image[30:50, 30:50] = 255

        result = apply_censor_blur(image, (CensorCircle(0.5, 0.5, 0.25),), 5)

        self.assertTrue(np.array_equal(result[0, 0], image[0, 0]))
        self.assertTrue(np.array_equal(result[79, 79], image[79, 79]))

    def test_pixels_inside_patterned_circle_are_blurred(self) -> None:
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        image[:, 36:44] = 255

        result = apply_censor_blur(image, (CensorCircle(0.5, 0.5, 0.25),), 5)

        self.assertLess(int(result[40, 40, 0]), int(image[40, 40, 0]))
        self.assertGreater(int(result[40, 34, 0]), int(image[40, 34, 0]))

    def test_circle_partly_outside_image_is_clipped(self) -> None:
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        image[:12, :12] = 255

        result = apply_censor_blur(image, (CensorCircle(0.0, 0.0, 0.3),), 8)

        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertLess(int(result[5, 5, 0]), 255)

    def test_normalized_coordinates_target_full_resolution_location(self) -> None:
        image = np.zeros((40, 100, 3), dtype=np.uint8)
        image[18:23, 72:78] = 255

        result = apply_censor_blur(image, (CensorCircle(0.75, 0.5, 0.12),), 4)

        self.assertLess(int(result[20, 75, 0]), int(image[20, 75, 0]))
        self.assertTrue(np.array_equal(result[20, 15], image[20, 15]))

    def test_multiple_circles_are_applied(self) -> None:
        image = np.zeros((60, 120, 3), dtype=np.uint8)
        image[25:35, 25:35] = 255
        image[25:35, 85:95] = 255

        result = apply_censor_blur(
            image,
            (CensorCircle(0.25, 0.5, 0.12), CensorCircle(0.75, 0.5, 0.12)),
            4,
        )

        self.assertLess(int(result[30, 30, 0]), 255)
        self.assertLess(int(result[30, 90, 0]), 255)


if __name__ == "__main__":
    unittest.main()
