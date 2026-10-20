"""Run the mapper on a frame/telemetry dataset and optionally score it."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2


def load_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_truth(path):
    if not path.exists():
        return None
    return [(int(row["marker_id"]), float(row["x"]), float(row["y"]))
            for row in load_rows(path)]


def score_predictions(predictions, truth, radius=1.0, full_radius=0.3, false_penalty=0.5):
    truth_by_id = {int(marker_id): (float(x), float(y)) for marker_id, x, y in truth}
    best = {}
    for index, (marker_id, x, y) in enumerate(predictions):
        if marker_id not in truth_by_id:
            continue
        tx, ty = truth_by_id[marker_id]
        distance = ((x - tx) ** 2 + (y - ty) ** 2) ** 0.5
        if distance <= radius and (marker_id not in best or distance < best[marker_id][0]):
            best[marker_id] = (distance, index)
    matched = {index for _, index in best.values()}
    false_positives = sum(index not in matched for index in range(len(predictions)))
    errors = [distance for distance, _ in best.values()]
    credit = sum(1.0 if d <= full_radius else (radius - d) / (radius - full_radius)
                 for d in errors)
    score = 100.0 * (credit - false_penalty * false_positives) / max(len(truth_by_id), 1)
    return {
        "score": max(0.0, min(100.0, score)),
        "found": len(best),
        "missed": sorted(set(truth_by_id) - set(best)),
        "false_positives": false_positives,
        "mean_error_m": sum(errors) / len(errors) if errors else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--solution", default=None)
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from solution import Solution

    dataset = Path(args.dataset)
    with (dataset / "camera.json").open(encoding="utf-8") as handle:
        camera_info = json.load(handle)
    with (dataset / "rules.json").open(encoding="utf-8") as handle:
        rules = json.load(handle)
    with (dataset / "info.json").open(encoding="utf-8") as handle:
        info = json.load(handle)

    solution = Solution(camera_info, rules, info)
    rows = load_rows(dataset / "telemetry.csv")
    start = time.perf_counter()
    for row in rows:
        filename = (row.get("frame") or "").strip()
        frame = cv2.imread(str(dataset / filename), cv2.IMREAD_COLOR) if filename else None
        telemetry = {key: float(row[key]) for key in
                     ("t", "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz")}
        solution.on_frame(frame, telemetry)
    predictions = solution.finalize()
    elapsed = time.perf_counter() - start

    with Path(args.output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["marker_id", "x", "y"])
        writer.writerows(predictions)
    print(f"predictions: {len(predictions)} rows")
    print(f"processing time: {elapsed:.3f} s")

    truth = load_truth(dataset / "ground_truth.csv")
    if truth is not None:
        result = score_predictions(predictions, truth)
        mean_error = "n/a" if result["mean_error_m"] is None else f"{result['mean_error_m']:.4f} m"
        print(f"score: {result['score']:.2f}/100")
        print(f"found: {result['found']} | missed: {result['missed']} | "
              f"false positives: {result['false_positives']} | mean error: {mean_error}")
    else:
        print("ground_truth.csv not found; predictions were written without a score")


if __name__ == "__main__":
    main()
