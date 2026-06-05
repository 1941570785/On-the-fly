from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils import get_image_names


class FrameOrderingTests(unittest.TestCase):
    def test_get_image_names_returns_numeric_frame_order_for_streaming(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["1.jpg", "10.jpg", "100.jpg", "2.jpg", "20.jpg", "3.png", "notes.txt"]:
                (root / name).write_text("x", encoding="utf-8")

            self.assertEqual(
                get_image_names(str(root)),
                ["1.jpg", "2.jpg", "3.png", "10.jpg", "20.jpg", "100.jpg"],
            )


if __name__ == "__main__":
    unittest.main()
