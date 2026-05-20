from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py


EXPECTED = {
    1: (200, 256),
    2: (200, 256),
    3: (400, 256),
}


def validate_jsonl(path: Path) -> None:
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            row = json.loads(line)
            if "timestamp" not in row or "elapsed_seconds" not in row:
                raise ValueError(f"{path}:{idx} missing timestamp or elapsed_seconds")
            if "response" not in row and "tool_calls" not in row:
                raise ValueError(f"{path}:{idx} missing response/tool_calls")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", default="submission")
    parser.add_argument("--tasks", nargs="+", type=int, required=True)
    args = parser.parse_args()

    root = Path(args.submission_dir)
    for task_id in args.tasks:
        pred_path = root / f"task{task_id}_pred.hdf5"
        time_path = root / f"task{task_id}_time.csv"
        log_path = root / f"task{task_id}_logs.log"
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        with h5py.File(pred_path, "r") as h5:
            key = "tensor" if "tensor" in h5 else list(h5.keys())[0]
            shape = h5[key].shape
        if shape[1:] != EXPECTED[task_id]:
            raise ValueError(f"task{task_id} expected (*,{EXPECTED[task_id][0]},{EXPECTED[task_id][1]}), got {shape}")
        with open(time_path, newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f))
            float(row["train_time"])
            float(row["inference_time"])
        validate_jsonl(log_path)
        print(f"task{task_id}: ok")


if __name__ == "__main__":
    main()

