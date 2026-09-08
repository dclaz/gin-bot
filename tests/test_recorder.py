"""Recorder: JSONL source of truth, namespace discipline, sink isolation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ginrl.telemetry.recorder import Recorder, RecorderConfig

ROOT = Path(__file__).resolve().parents[1]


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_jsonl_has_header_scalars_histogram_table(tmp_path: Path) -> None:
    with Recorder(
        RecorderConfig(run_dir=tmp_path / "run", run_name="smoke", dashboard_enabled=False)
    ) as rec:
        rec.log_scalar(0, "loss/total", 1.5)
        rec.log_histogram(1, "style/deadwood_at_knock", [1.0, 2.0, 7.0, 7.0])
        rec.log_table(2, "ratings/ladder", ["agent", "elo"], [["a", 0]])
    records = _read_lines(tmp_path / "run" / "metrics.jsonl")
    kinds = [r["type"] for r in records]
    assert kinds == ["header", "scalar", "histogram", "table"]
    header = records[0]
    assert header["run"] == "smoke"
    assert set(header) >= {"git_sha", "config_hash", "facts_hash"}
    scalar = records[1]
    assert (scalar["step"], scalar["metric"], scalar["value"]) == (0, "loss/total", 1.5)
    hist = records[2]
    assert hist["count"] == 4 and sum(hist["bins"]) == 4 and hist["mean"] == 4.25
    table = records[3]
    assert table["columns"] == ["agent", "elo"] and table["rows"] == [["a", "0"]]


def test_unknown_namespace_is_rejected(tmp_path: Path) -> None:
    with Recorder(
        RecorderConfig(run_dir=tmp_path / "run", run_name="smoke", dashboard_enabled=False)
    ) as rec:
        with pytest.raises(ValueError, match="namespace"):
            rec.log_scalar(0, "vibes/total", 1.0)
        with pytest.raises(ValueError, match="namespace"):
            rec.log_histogram(0, "no-namespace", [1.0])


def test_dying_sink_does_not_kill_the_run(tmp_path: Path) -> None:
    class FlakySink:
        def __init__(self) -> None:
            self.calls = 0

        def scalar(self, step: int, metric: str, value: float) -> None:
            self.calls += 1
            if self.calls > 2:
                raise ConnectionError("dashboard died mid-run")

        def histogram(self, step: int, metric: str, values: object) -> None:
            raise ConnectionError("dashboard died mid-run")

        def table(self, step: int, metric: str, columns: object, rows: object) -> None:
            raise ConnectionError("dashboard died mid-run")

    with Recorder(
        RecorderConfig(run_dir=tmp_path / "run", run_name="smoke"),
        dashboard=FlakySink(),  # type: ignore[arg-type]
    ) as rec:
        with pytest.warns(UserWarning, match="dashboard sink failed"):
            for i in range(10):
                rec.log_scalar(i, "loss/total", float(i))
    records = _read_lines(tmp_path / "run" / "metrics.jsonl")
    assert len(records) == 11  # header + 10 scalars, nothing lost


REAL_SINK_SCRIPT = """
import sys
sys.path.insert(0, {src!r})
from pathlib import Path
from ginrl.telemetry.recorder import Recorder, RecorderConfig
run_dir = Path({run_dir!r})
with Recorder(
    RecorderConfig(run_dir=run_dir, run_name="real-sink", dashboard_enabled=True)
) as rec:
    rec.log_scalar(0, "loss/total", 1.0)
    rec.log_histogram(0, "style/deadwood_at_knock", [1.0, 7.0, 7.0])
    rec.log_table(0, "ratings/ladder", ["agent", "elo"], [["a", 0]])
print("real sink ok")
"""


def test_real_trackio_sink_accepts_all_record_kinds(tmp_path: Path) -> None:
    """The true Trackio API surface (init/log/Histogram/Table/finish).

    Runs in a subprocess because TRACKIO_DIR is frozen at trackio import
    time, and because a warning-degrading stub cannot catch API drift: the
    Table path once called a nonexistent add_data without failing.
    UserWarnings are errors inside the subprocess, so any dashboard
    degradation here fails instead of degrading.
    """
    store = tmp_path / "trackio_store"
    run_dir = tmp_path / "run"
    env = dict(os.environ, TRACKIO_DIR=str(store), PYTHONPATH=str(ROOT / "src"))
    proc = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::UserWarning",
            "-c",
            REAL_SINK_SCRIPT.format(src=str(ROOT / "src"), run_dir=str(run_dir)),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert "real sink ok" in proc.stdout
    assert len((run_dir / "metrics.jsonl").read_text().splitlines()) == 4
    assert any(store.rglob("*.sqlite")) or any(store.rglob("*.db")), "no trackio store written"
