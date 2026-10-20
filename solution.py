"""Aerial marker mapping module.

Detect valid DICT_4X4_50 markers, project their image corners through the
supplied camera model, reject large telemetry jumps, and aggregate repeated
observations before returning one row per marker.

Only the participant API inputs are used.  This module deliberately has no
simulator, filesystem, process, or network dependencies.
"""

import math

import cv2
import numpy as np


class Solution:
    def __init__(self, camera_info, rules, info):
        self.camera_info = camera_info
        self.rules = rules
        self.info = info

        self.K = np.asarray(camera_info["K"], dtype=np.float64)
        self.dist = np.asarray(camera_info.get("dist", [0.0] * 5), dtype=np.float64)
        self.R_body_cam = np.asarray(camera_info["R_body_cam"], dtype=np.float64)
        self.valid_ids = set(int(v) for v in rules["valid_ids"])
        self.field_half_size = float(rules["field_half_size_m"])
        self.marker_size_m = float(rules["marker_size_m"])
        self.part = str(info.get("part", "A")).upper()

        dictionary_name = str(rules["dictionary"])
        dictionary_id = getattr(cv2.aruco, dictionary_name)
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        # ID -> list of independent ground-position estimates.  A marker is
        # deliberately accumulated rather than emitted on its first sighting.
        self.observations = {}
        self.size_observations = {}
        self._last_pose_t = None
        self._last_stable_position = None

    @staticmethod
    def _rot_world_body(roll, pitch, yaw):
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        return rz @ ry @ rx

    def _image_to_ground(self, pixels, telemetry):
        """Map image pixels to z=0 using the supplied pose and intrinsics."""
        pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        # undistortPoints returns normalized camera coordinates when P=None.
        normalized = cv2.undistortPoints(pts, self.K, self.dist).reshape(-1, 2)
        rays_cam = np.column_stack((normalized, np.ones(len(normalized))))

        r_wb = self._rot_world_body(float(telemetry["roll"]),
                                     float(telemetry["pitch"]),
                                     float(telemetry["yaw"]))
        r_wc = r_wb @ self.R_body_cam
        rays_world = rays_cam @ r_wc.T
        position = np.array([float(telemetry["x"]), float(telemetry["y"]),
                             float(telemetry["z"])], dtype=np.float64)
        denom = rays_world[:, 2]
        good = np.abs(denom) > 1e-9
        scales = -position[2] / denom
        ground = position[None, :] + scales[:, None] * rays_world
        return ground[:, :2], good

    def _stabilize_telemetry(self, telemetry):
        """Reject large position innovations while preserving normal motion.

        GNSS glitches are much larger than one camera interval of
        motion.  A constant-velocity prediction is therefore a safe gate.  We
        do not smooth every sample: doing so would introduce lag during turns
        and would be worse than the supplied measurement for ordinary frames.
        The state is updated even for dropped frames.
        """
        out = dict(telemetry)
        t = float(telemetry["t"])
        raw = np.array([float(telemetry["x"]), float(telemetry["y"]),
                        float(telemetry["z"])], dtype=np.float64)
        velocity = np.array([float(telemetry.get("vx", 0.0)),
                             float(telemetry.get("vy", 0.0)),
                             float(telemetry.get("vz", 0.0))], dtype=np.float64)

        if self._last_pose_t is None or self._last_stable_position is None:
            stable = raw
        else:
            dt = float(np.clip(t - self._last_pose_t, 0.0, 0.5))
            predicted = self._last_stable_position + velocity * dt
            # The simulator's 2.5 m GNSS glitches are clearly separated from
            # ordinary motion/noise.  Gate horizontal position only; altitude
            # is independently observed and has no multi-metre jumps in B.
            if np.linalg.norm(raw[:2] - predicted[:2]) > 0.9:
                stable = predicted
                stable[2] = raw[2]
            else:
                stable = raw

        self._last_pose_t = t
        self._last_stable_position = stable.copy()
        out["x"], out["y"], out["z"] = (float(stable[0]), float(stable[1]),
                                          float(stable[2]))
        return out

    def on_frame(self, frame, telemetry):
        telemetry = self._stabilize_telemetry(telemetry)
        if frame is None:
            return

        corners, ids, _ = self.detector.detectMarkers(frame)
        # CLAHE is useful under dark cloud-shadow frames, but
        # running it on every image costs time and can amplify artefacts.  Use
        # it only as a recovery path when the normal detector found nothing.
        if ids is None and self.part in ("B", "C"):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return

        for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
            marker_id = int(marker_id)
            if marker_id not in self.valid_ids:
                continue
            image_corners = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
            ground_corners, good = self._image_to_ground(image_corners, telemetry)
            if not bool(np.all(good)) or not np.all(np.isfinite(ground_corners)):
                continue
            # The mean of the four mapped physical corners is the tag centre.
            estimate = np.mean(ground_corners, axis=0)
            edge_lengths = np.linalg.norm(np.roll(ground_corners, -1, axis=0)
                                          - ground_corners, axis=1)
            estimated_size = float(np.median(edge_lengths))
            if (np.all(np.abs(estimate) <= self.field_half_size + 2.0)
                    and np.all(np.isfinite(estimate))
                    and np.isfinite(estimated_size)):
                self.observations.setdefault(marker_id, []).append(estimate)
                self.size_observations.setdefault(marker_id, []).append(estimated_size)

    @staticmethod
    def _single_stationary_cluster(samples):
        """Return whether samples support one fixed ground marker.

        The first test tolerates occasional telemetry/image outliers. The
        second explicitly rejects two separated stationary copies of one ID,
        while also rejecting a moving marker whose observations cover a long
        path.
        """
        if len(samples) < 3:
            return False
        centre = np.median(samples, axis=0)
        distances = np.linalg.norm(samples - centre[None, :], axis=1)
        if float(np.mean(distances <= 0.55)) < 0.72:
            return False

        # Find the densest local group, then look for another substantial,
        # spatially separated group. Genuine IDs have only one such group.
        neighbor_counts = np.sum(
            np.linalg.norm(samples[:, None, :] - samples[None, :, :], axis=2)
            <= 0.55, axis=1)
        first_centre = np.median(samples[neighbor_counts == neighbor_counts.max()], axis=0)
        remaining = samples[np.linalg.norm(samples - first_centre[None, :], axis=1) > 0.85]
        # A few isolated samples can be caused by the simulator's 3% GNSS
        # jumps. Require a substantial second group before calling an ID a
        # duplicate; genuine tags still have many observations in their main
        # cluster, while each duplicate location is observed repeatedly.
        if len(remaining) >= 5:
            second_centre = np.median(remaining, axis=0)
            if float(np.linalg.norm(second_centre - first_centre)) > 0.9:
                return False
        return True

    def finalize(self):
        answers = []
        for marker_id in sorted(self.observations):
            samples = np.asarray(self.observations[marker_id], dtype=np.float64)
            # A genuine tag is visible across consecutive lawnmower frames;
            # requiring several sightings suppresses occasional texture
            # false positives without affecting clean-input tags.
            if len(samples) < 3:
                continue

            if self.part == "C":
                sizes = np.asarray(self.size_observations.get(marker_id, []), dtype=np.float64)
                size_ratio = float(np.median(sizes) / max(self.marker_size_m, 1e-9)) \
                    if len(sizes) else 0.0
                # A broad 0.72--1.30 acceptance band covers telemetry noise
                # while separating those fakes from the genuine 0.50 m tag.
                if not (0.72 <= size_ratio <= 1.30):
                    continue
                if not self._single_stationary_cluster(samples):
                    continue

            # This robust centre costs little and keeps
            # the interface ready for later disturbance work.
            centre = np.median(samples, axis=0)
            distances = np.linalg.norm(samples - centre[None, :], axis=1)
            keep = distances <= max(0.20, float(np.percentile(distances, 80)) + 0.05)
            refined = np.mean(samples[keep], axis=0) if np.any(keep) else centre
            answers.append((int(marker_id), float(refined[0]), float(refined[1])))
        return answers
