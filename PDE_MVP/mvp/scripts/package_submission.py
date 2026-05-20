from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-id", required=True)
    parser.add_argument("--tasks", nargs="+", type=int, required=True)
    parser.add_argument("--source-dir", default=".")
    parser.add_argument("--work-dir", default="submission")
    parser.add_argument("--zip-path", default="submission.zip")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    metadata = {
        "submission_id": args.submission_id,
        "problem_id": "PDE_Burgers",
        "code_path": "code",
    }
    (work_dir / "submission.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    for task_id in args.tasks:
        copy_if_exists(source_dir / "submission" / f"task{task_id}_pred.hdf5", work_dir / f"task{task_id}_pred.hdf5")
        copy_if_exists(source_dir / "submission" / f"task{task_id}_time.csv", work_dir / f"task{task_id}_time.csv")
        copy_if_exists(source_dir / "submission" / f"task{task_id}_logs.log", work_dir / f"task{task_id}_logs.log")

    methodology = source_dir / "methodology.pdf"
    if methodology.exists():
        shutil.copy2(methodology, work_dir / "methodology.pdf")
    else:
        (work_dir / "methodology.md").write_text(
            "# PDE Agent Methodology\n\n"
            "This MVP uses a compact Fourier-style 1D neural operator, segmented losses, "
            "and JSONL experiment logging. Replace this placeholder with a rendered PDF before final submission.\n",
            encoding="utf-8",
        )

    code_dst = work_dir / "code"
    ignore = shutil.ignore_patterns("runs", "submission", "__pycache__", "*.pt", "*.hdf5", "*.zip")
    shutil.copytree(source_dir, code_dst, ignore=ignore)

    zip_path = Path(args.zip_path).resolve()
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in work_dir.rglob("*"):
            zf.write(path, path.relative_to(work_dir.parent))
    print(f"Wrote {zip_path}")


if __name__ == "__main__":
    main()

