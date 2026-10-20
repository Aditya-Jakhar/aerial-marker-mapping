# Aerial Marker Mapping

A compact computer-vision pipeline for locating fixed square fiducial markers
on a ground plane from downward-facing camera frames and platform telemetry.
It is designed as a practical reference implementation for aerial mapping
experiments where images and navigation measurements are imperfect.

## Capabilities

- Detects `DICT_4X4_50` ArUco markers with OpenCV.
- Converts image corners into world-frame ground coordinates using supplied
  camera intrinsics, lens distortion, mounting rotation, and pose telemetry.
- Combines repeated observations into one position per marker.
- Rejects large horizontal telemetry innovations using a constant-velocity gate.
- Handles missing frames while continuing to update the pose estimate.
- Uses a contrast-normalized recovery pass when normal detection fails.
- Can reject implausible physical sizes and multiple/moving position clusters.

## Requirements

- Python 3.9+
- NumPy
- OpenCV with the ArUco module

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## API

The main module exposes:

```python
from solution import Solution

mapper = Solution(camera_info, rules, info)
mapper.on_frame(bgr_frame, telemetry)
locations = mapper.finalize()
```

`bgr_frame` is a NumPy image or `None` when a frame is unavailable.
`telemetry` contains position, attitude, velocity, and timestamp fields. The
result is a list of `(marker_id, x, y)` tuples in metres.

The module does not read files, access the network, launch processes, or depend
on a particular flight simulator. A host application supplies the camera
frames, calibration, marker rules, and telemetry.

## Running the checks

Run the standard-library smoke test from the repository root:

```bash
python -m unittest discover -s tests -v
```

The test verifies initialization, dropped-frame handling, and safe finalization
without requiring a camera or external dataset. For a real dataset, pass each
decoded BGR frame and its synchronized telemetry record to `on_frame`, then
write the returned coordinates from `finalize()` in the host application.

## Try it with sample data

The repository includes a generator for a small, reproducible sample dataset.
It creates ArUco frames, camera calibration, telemetry, and a ground-truth file
without requiring any downloaded images:

```bash
python tools/create_sample_dataset.py --output examples/sample_dataset
python tools/run_dataset.py --dataset examples/sample_dataset \
  --output examples/sample_predictions.csv
```

The runner prints the number of predictions, processing time, found and missed
markers, false positives, mean position error, and a 0–100 localization score.
Generated sample data is ignored by Git so it can be recreated or removed
without affecting the source repository.

## Use your own dataset

Create a directory with this layout:

```text
my_dataset/
├── camera.json
├── rules.json
├── info.json
├── telemetry.csv
├── frames/
│   ├── 000001.jpg
│   └── 000002.jpg
└── ground_truth.csv       # optional, needed for scoring
```

`camera.json` must contain `width`, `height`, `K`, `dist`, `R_body_cam`, and
`fps`. `rules.json` contains `dictionary`, `valid_ids`, `marker_size_m`, and
`field_half_size_m`. `info.json` contains at least `part` and `frame_rate`.

Each row in `telemetry.csv` names a frame relative to the dataset directory and
contains these synchronized fields:

```text
frame,t,x,y,z,roll,pitch,yaw,vx,vy,vz
frames/000001.jpg,0.0,0.0,0.0,8.0,0.0,0.0,0.0,0.0,0.0,0.0
```

Use an empty `frame` value for a dropped image while retaining its telemetry
row. `ground_truth.csv`, when available, uses `marker_id,x,y` and one row per
known fixed marker. Run the dataset with:

```bash
python tools/run_dataset.py --dataset path/to/my_dataset \
  --output path/to/my_predictions.csv
```

Without ground truth, the tool still produces predictions and processing time.
With ground truth, it provides a repeatable accuracy report. The score gives
full position credit within 0.3 m, fades to zero at 1.0 m, and applies a small
penalty for predictions that do not match a known marker.

## What the project means in practice

This is a ground-plane localization component, not a complete drone autopilot
or mapping application. It assumes the host system already provides:

1. A calibrated camera and its mounting rotation.
2. A synchronized pose/velocity record for each image.
3. Square fiducial markers lying on a mostly flat ground plane.
4. Frames decoded as BGR NumPy arrays.

For a field deployment, connect the camera and navigation middleware to the
same `on_frame` loop used by `tools/run_dataset.py`, then publish the returned
marker coordinates to your map or planning system. Calibration quality,
timestamp synchronization, camera mounting, altitude, and ground flatness are
the main real-world accuracy factors. The supplied implementation does not
calibrate a camera, recover a pose from images alone, or replace a full SLAM
system for arbitrary unmarked environments.

## Coordinate model

The mapping uses a world frame with x east, y north, and z up; the ground plane
is z=0. The body frame uses x forward, y left, and z up; the camera frame uses
x right, y down, and z forward. Telemetry roll, pitch, and yaw are interpreted
as `R_WB = Rz(yaw) @ Ry(pitch) @ Rx(roll)`, and the supplied
`R_body_cam` gives `R_WC = R_WB @ R_body_cam`. Each detected corner is
undistorted, transformed into a world ray, intersected with z=0, and combined
with the other three corners to estimate the marker centre.

## Project structure

```text
solution.py              Core marker detector and mapper
requirements.txt         Runtime dependencies
tests/test_solution.py   Dependency-light API smoke test
tools/create_sample_dataset.py  Reproducible sample-data generator
tools/run_dataset.py     Dataset runner and accuracy report
LICENSE                  MIT license
```
