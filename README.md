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

## Coordinate model

The mapping uses a world frame with x east, y north, and z up; the ground plane
is z=0. The body-to-camera rotation is supplied by the caller. Each detected
corner is undistorted, transformed into a world ray, intersected with z=0, and
combined with the other three corners to estimate the marker centre.

## Project structure

```text
solution.py              Core marker detector and mapper
requirements.txt         Runtime dependencies
tests/test_solution.py   Dependency-light API smoke test
LICENSE                  MIT license
```
