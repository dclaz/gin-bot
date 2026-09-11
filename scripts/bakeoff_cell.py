"""Run one bake-off cell; writes cell.json. Used by scripts/gate_p4.py.

Subprocess-isolated (no fork): each invocation is a fresh interpreter, so
parallel cells cannot share torch/MPS state.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from ginrl.eval.bakeoff import cell_dict, run_cell  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torso", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--eval-deals-final", type=int, default=2000)
    parser.add_argument("--belief-seed", type=int, required=True)
    parser.add_argument("--belief-states", required=True)
    parser.add_argument("--mask-beliefs", action="store_true")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    with open(args.belief_states, "rb") as fh:
        belief_states = pickle.load(fh)
    cell = run_cell(
        args.torso,
        args.seed,
        total_steps=args.steps,
        device=torch.device("cpu"),
        parent=Path(args.parent),
        eval_deals_final=args.eval_deals_final,
        belief_states=belief_states,
        belief_seed=args.belief_seed,
        mask_beliefs=args.mask_beliefs,
        hidden=args.hidden,
        layers=args.layers,
        tag=args.tag,
    )
    out = Path(args.parent) / (args.tag or f"{args.torso}-s{args.seed}") / "cell.json"
    out.write_text(json.dumps(cell_dict(cell), indent=2))
    print(
        f"[{args.torso} s{args.seed}] score={cell.final_score:+.3f} "
        f"[{cell.score_lo:+.3f},{cell.score_hi:+.3f}] auc={cell.belief_auc:.3f} "
        f"ent={cell.final_entropy:.3f} rows/s={cell.rows_per_sec:.0f} "
        f"infer={cell.infer_ms:.2f}ms",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
