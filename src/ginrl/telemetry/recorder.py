"""The fan-out Recorder: JSONL is the record, Trackio is the human view.

Every `log_*` call validates the metric namespace, appends a typed record to
`runs/<run>/metrics.jsonl` (the source of truth gates read), and mirrors to
the dashboard sink. The dashboard is optional: any failure degrades to a
warning and the run continues with the JSONL intact.
"""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ginrl.telemetry.provenance import git_sha, spiel_facts_hash

# Closed namespace list (docs/OBSERVABILITY.md). A new namespace needs a
# commit message saying why.
NAMESPACES = frozenset(
    {
        "loss",
        "reg",
        "policy",
        "value",
        "opt",
        "belief",
        "ratings",
        "population",
        "style",
        "perf",
        "tripwire",
        "match",
    }
)

HISTOGRAM_BINS = 32


def check_metric(metric: str) -> str:
    """Validate `namespace/name` against the closed list. Returns the metric."""
    namespace = metric.split("/", 1)[0] if "/" in metric else ""
    if namespace not in NAMESPACES:
        raise ValueError(f"metric {metric!r} not in a known namespace {sorted(NAMESPACES)}")
    return metric


class TrackioDashboard:
    """Thin wrapper over trackio so the Recorder never calls it directly."""

    def __init__(self, project: str, run: str) -> None:
        import trackio

        trackio.init(project=project, name=run)
        self._trackio = trackio

    def scalar(self, step: int, metric: str, value: float) -> None:
        self._trackio.log({metric: value}, step=step)

    def histogram(self, step: int, metric: str, values: Sequence[float]) -> None:
        self._trackio.log({metric: self._trackio.Histogram(list(values))}, step=step)

    def table(self, step: int, metric: str, columns: list[str], rows: list[list[object]]) -> None:
        # No add_data on this API: rows go through the constructor.
        table = self._trackio.Table(columns=columns, data=[[str(v) for v in row] for row in rows])
        self._trackio.log({metric: table}, step=step)

    def finish(self) -> None:
        self._trackio.finish()


@dataclass
class RecorderConfig:
    run_dir: Path
    run_name: str
    project: str = "ginrl"
    config_hash: str = ""
    dashboard_enabled: bool = True
    flush_every: int = 100  # JSONL + dashboard flush cadence (records)


class Recorder:
    def __init__(
        self,
        config: RecorderConfig,
        dashboard: TrackioDashboard | None = None,
    ) -> None:
        self.config = config
        self.path = config.run_dir / "metrics.jsonl"
        config.run_dir.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        self._buffered = 0
        self._closed = False
        header = {
            "type": "header",
            "run": config.run_name,
            "git_sha": git_sha(),
            "config_hash": config.config_hash,
            "facts_hash": spiel_facts_hash(),
            "t": time.time(),
        }
        self._write(header)
        self._sink_ok = config.dashboard_enabled
        if dashboard is not None:
            self._dashboard = dashboard
        elif config.dashboard_enabled:
            try:
                self._dashboard = TrackioDashboard(config.project, config.run_name)
            except Exception as exc:
                warnings.warn(f"dashboard sink disabled at init: {exc}", stacklevel=2)
                self._sink_ok = False
                self._dashboard = None
        else:
            self._dashboard = None

    def _write(self, record: dict) -> None:
        self._file.write(json.dumps(record) + "\n")

    def _mirror(self, what: str, method: str, *args: object) -> None:
        if not self._sink_ok or self._dashboard is None:
            return
        try:
            getattr(self._dashboard, method)(*args)
        except Exception as exc:
            warnings.warn(
                f"dashboard sink failed on {what}; continuing: {exc}",
                stacklevel=2,
            )
            self._sink_ok = False

    def log_scalar(self, step: int, metric: str, value: float) -> None:
        check_metric(metric)
        self._write({"type": "scalar", "step": step, "metric": metric, "value": float(value)})
        self._buffered += 1
        # Mirror immediately; buffering only batches the file flush.
        self._mirror(metric, "scalar", step, metric, float(value))
        self._maybe_flush()

    def log_histogram(self, step: int, metric: str, values: Sequence[float]) -> None:
        check_metric(metric)
        arr = np.asarray(list(values), dtype=np.float64)
        counts, edges = np.histogram(arr, bins=HISTOGRAM_BINS)
        self._write(
            {
                "type": "histogram",
                "step": step,
                "metric": metric,
                "count": int(arr.size),
                "min": float(arr.min()) if arr.size else 0.0,
                "max": float(arr.max()) if arr.size else 0.0,
                "mean": float(arr.mean()) if arr.size else 0.0,
                "bins": [int(c) for c in counts],
                "edges": [float(e) for e in edges],
            }
        )
        self._buffered += 1
        self._mirror(metric, "histogram", step, metric, list(values))
        self._maybe_flush()

    def log_table(
        self,
        step: int,
        metric: str,
        columns: list[str],
        rows: list[list[object]],
    ) -> None:
        check_metric(metric)
        self._write(
            {
                "type": "table",
                "step": step,
                "metric": metric,
                "columns": columns,
                "rows": [[str(v) for v in row] for row in rows],
            }
        )
        self._buffered += 1
        self._mirror(metric, "table", step, metric, columns, rows)
        self._maybe_flush()

    def _maybe_flush(self) -> None:
        if self._buffered >= self.config.flush_every:
            self.flush()

    def flush(self) -> None:
        self._file.flush()
        self._buffered = 0

    def close(self) -> None:
        if not self._closed:
            self.flush()
            self._file.close()
            self._closed = True
            if self._sink_ok and self._dashboard is not None:
                try:
                    self._dashboard.finish()
                except Exception as exc:
                    warnings.warn(
                        f"dashboard finish failed; JSONL is complete: {exc}",
                        stacklevel=2,
                    )

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
