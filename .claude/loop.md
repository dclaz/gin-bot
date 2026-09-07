# Watch the active training run

Read-only. This prompt observes a run; it never edits code, config or gates.

1. Find the newest directory under `runs/` and read the tail of its
   `metrics.jsonl`. That file is the source of truth — do not open a dashboard.
2. Read the `tripwires:` block in `configs/gates.yaml`. Every tripwire logs both
   its current value and its threshold, so compare the pair, not the value alone.
3. Report exactly two lines:
   - step, wall-clock, `ratings/points_per_hand_vs_anchor`,
     `policy/entropy_normalised`, `perf/env_steps_per_sec`;
   - the tripwire currently closest to its threshold, as `name: value/threshold`.
4. If a tripwire has fired, say so, name it, give the path of its
   `tripwire_<step>.json` dump, and stop the loop. Do not diagnose, do not fix,
   do not restart the run.
5. If the run has finished or the JSONL has not grown since the last wakeup, say
   which and stop the loop.

Long runs must be backgrounded: scheduled wakeups fire only when Claude is idle,
and missed fires are not caught up, so a foreground run blocks every check.
