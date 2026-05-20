from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pde_agent.agent_logger import AgentLogger


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal experiment loop for the PDE Agent MVP.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", default="agent_runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    logger = AgentLogger(work_dir / "agent_logs.jsonl")

    logger.record(
        response=(
            "Agent loop started. Current MVP policy: run one configured training job, "
            "read its metrics, and write an experiment reflection. Future versions should "
            "generate code patches in a sandbox before launching longer experiments."
        )
    )

    if args.dry_run:
        logger.record(response="Dry run selected; no subprocess training was launched.")
        return

    cmd = [sys.executable, "train.py", "--config", args.config, "--run-dir", str(work_dir)]
    logger.record(tool_calls=[{"name": "subprocess.run", "arguments": cmd}])
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    logger.record(
        response="Training subprocess completed.",
        tool_calls=[
            {
                "name": "subprocess.run.result",
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-2000:],
                "stderr_tail": result.stderr[-2000:],
            }
        ],
    )

    metrics_files = sorted(work_dir.glob("*/metrics.json"))
    if metrics_files:
        latest = metrics_files[-1]
        metrics = json.loads(latest.read_text(encoding="utf-8"))
        logger.record(
            response=(
                f"Experiment reflection: latest metrics from {latest} show "
                f"best_rel_mse={metrics.get('best_rel_mse')}. "
                "If segment 3 remains weak, the next hypothesis should add spectral/statistical loss."
            ),
            extra={"metrics": metrics},
        )


if __name__ == "__main__":
    main()

