from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class AgentLogger:
    """Write competition-compatible JSONL logs for Agent actions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        response: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        elapsed_seconds: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        row: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": float(elapsed_seconds),
        }
        if response is not None:
            row["response"] = response
        if tool_calls is not None:
            row["tool_calls"] = tool_calls
        if extra:
            row.update(extra)
        if "response" not in row and "tool_calls" not in row:
            row["response"] = ""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def timed_response(self, response: str, extra: Optional[Dict[str, Any]] = None) -> None:
        start = time.perf_counter()
        self.record(response=response, elapsed_seconds=time.perf_counter() - start, extra=extra)

