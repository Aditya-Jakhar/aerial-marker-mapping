"""Create a small reproducible dataset for local experiments."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def create_dataset(destination):
    destination = Path(destination)
    frames_dir = destination / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    camera = {
        "width": 640,
        "height": 480,
        "K": [[520.0, 0.0, 319.5], [0.0, 520.0, 239.5], [0.0, 0.0, 1.0]],
        "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
        "R_body_cam": [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        "fps": 10.0,
    }
    rules = {
        "dictionary": "DICT_4X4_50",
        "valid_ids": list(range(35)),
        "marker_size_m": 0.5,
        "field_half_size_m": 15.0,
    }
    info = {"part": "A", "frame_rate": 10.0, "altitude_m": 8.0, "speed_mps": 0.0}
    write_json(destination / "camera.json", camera)
    write_json(destination / "rules.json", rules)
    write_json(destination / "info.json", info)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 7, 33, borderBits=1)
    marker = cv2.copyMakeBorder(marker, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=238)

    rows = []
    for index in range(12):
        image = np.full((camera["height"], camera["width"], 3), 90, dtype=np.uint8)
        h, w = marker.shape
        x0 = int(round(camera["K"][0][2] - w / 2))
        y0 = int(round(camera["K"][1][2] - h / 2))
        image[y0:y0 + h, x0:x0 + w] = marker[:, :, None]
        filename = f"{index:06d}.png"
        cv2.imwrite(str(frames_dir / filename), image)
        rows.append({
            "frame": f"frames/{filename}", "t": f"{index / 10:.3f}",
            "x": 0.0, "y": 0.0, "z": 8.0,
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            "vx": 0.0, "vy": 0.0, "vz": 0.0,
        })

    with (destination / "telemetry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (destination / "ground_truth.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["marker_id", "x", "y"])
        writer.writerow([7, 0.0, 0.0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="examples/sample_dataset")
    args = parser.parse_args()
    create_dataset(args.output)
    print(f"Created sample dataset at {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
