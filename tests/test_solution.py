import unittest

import cv2
import numpy as np

from solution import Solution


class SolutionSmokeTest(unittest.TestCase):
    def setUp(self):
        self.camera_info = {
            "width": 640,
            "height": 480,
            "K": [[520.0, 0.0, 319.5], [0.0, 520.0, 239.5], [0.0, 0.0, 1.0]],
            "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
            "R_body_cam": [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            "fps": 10.0,
        }
        self.rules = {
            "dictionary": "DICT_4X4_50",
            "valid_ids": list(range(35)),
            "marker_size_m": 0.5,
            "field_half_size_m": 15.0,
        }
        self.info = {"part": "A", "frame_rate": 10.0}

    def test_blank_and_dropped_frames_are_safe(self):
        solution = Solution(self.camera_info, self.rules, self.info)
        telemetry = {
            "t": 0.0, "x": 0.0, "y": 0.0, "z": 8.0,
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            "vx": 0.0, "vy": 0.0, "vz": 0.0,
        }
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        solution.on_frame(blank, telemetry)
        telemetry["t"] = 0.1
        solution.on_frame(None, telemetry)
        self.assertEqual(solution.finalize(), [])

    def test_detector_module_is_available(self):
        self.assertTrue(hasattr(cv2, "aruco"))


if __name__ == "__main__":
    unittest.main()
