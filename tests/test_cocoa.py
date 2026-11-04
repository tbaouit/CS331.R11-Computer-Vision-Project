import unittest
import numpy as np
from pycocotools import mask as coco_mask
from src.data.cocoa import decode_mask, Target
from pathlib import Path


class MaskTests(unittest.TestCase):
    def test_compressed_rle_roundtrip(self):
        expected = np.zeros((12, 17), dtype=np.uint8)
        expected[2:9, 3:14] = 1
        expected[4:6, 6:9] = 0
        rle = coco_mask.encode(np.asfortranarray(expected))
        rle["counts"] = rle["counts"].decode("ascii")
        np.testing.assert_array_equal(decode_mask(rle, 12, 17), expected.astype(bool))

    def test_uncompressed_rle_fortran_order(self):
        actual = decode_mask({"size": [2, 3], "counts": [1, 2, 3]}, 2, 3)
        np.testing.assert_array_equal(actual, [[False, True, False], [True, False, False]])

    def test_flat_polygon_and_polygon_list_agree(self):
        polygon = [2, 2, 8, 2, 8, 8, 2, 8]
        np.testing.assert_array_equal(decode_mask(polygon, 12, 17), decode_mask([polygon], 12, 17))
        self.assertEqual(int(decode_mask(polygon, 12, 17).sum()), 36)

    def test_wrong_dimensions_rejected(self):
        rle = coco_mask.encode(np.asfortranarray(np.ones((3, 4), dtype=np.uint8)))
        with self.assertRaises(ValueError):
            decode_mask(rle, 4, 3)

    def test_unoccluded_target_missing_visible(self):
        target = Target("train", 1, 1, 0, Path("dummy.jpg"), 10, 10,
                        {"segmentation": [1, 1, 8, 1, 8, 8, 1, 8], "occlude_rate": 0})
        visible, amodal, hidden = target.masks()
        np.testing.assert_array_equal(visible, amodal)
        self.assertFalse(hidden.any())

    def test_missing_visible_does_not_leak_amodal(self):
        target = Target("train", 1, 1, 0, Path("dummy.jpg"), 10, 10,
                        {"segmentation": [1, 1, 8, 1, 8, 8, 1, 8], "occlude_rate": 0.5})
        with self.assertRaisesRegex(ValueError, "no visible mask"):
            target.masks()


if __name__ == "__main__":
    unittest.main()
