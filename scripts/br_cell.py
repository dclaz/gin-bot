"""One RL-BR cell: train a best response vs a fixed full-game opponent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from ginrl.agents.baselines import HeuristicAgent, RandomAgent  # noqa: E402
from ginrl.agents.learned import GinNetAgent  # noqa: E402
from ginrl.agents.spiel_bots import SimpleGinRummyAgent  # noqa: E402
from ginrl.config import HandConfig, TrainerConfig  # noqa: E402
from ginrl.env.features import feature_dim  # noqa: E402
from ginrl.eval.rlbr_gin import train_br_gin  # noqa: E402
from ginrl.nets.gin import GinNet  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["champion", "heur", "bot", "random"])
    ap.add_argument("--champ", default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--eval-deals", type=int, default=1000)
    args = ap.parse_args()

    deck = HandConfig().deck_size

    def factory():
        if args.target == "champion":
            net = GinNet("mlp", feat_dim=feature_dim(deck), deck=deck, hidden=128)
            net.load_state_dict(torch.load(args.champ, map_location="cpu", weights_only=True))
            net.eval()
            return GinNetAgent(net, feature_dim(deck), torch.device("cpu"), seed=1)
        if args.target == "heur":
            return HeuristicAgent()
        if args.target == "bot":
            return SimpleGinRummyAgent()
        return RandomAgent(seed=args.seed)

    cfg = TrainerConfig(
        total_steps=args.steps,
        n_envs=16,
        rollout_len=128,
        epochs=2,
        minibatches=4,
        magnet_mode="uniform",
        reg_coef=0.0,
    )
    run = Path(args.parent) / f"br-{args.target}-s{args.seed}"
    pph = train_br_gin(
        factory,
        args.target,
        HandConfig(),
        "mlp",
        cfg,
        args.seed,
        torch.device("cpu"),
        run,
        eval_deals=args.eval_deals,
    )
    (run / "br.json").write_text(json.dumps({"target": args.target, "bound": pph}))
    print(f"[br {args.target} s{args.seed}] bound={pph:+.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
