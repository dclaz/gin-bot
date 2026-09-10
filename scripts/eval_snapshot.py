"""Background ladder eval for one match-training snapshot.

Spawned by train_match after each snapshot; the learner never blocks on it
(plan: rating tournaments run off the critical path). Appends legs to the
run's game_record.jsonl (small O_APPEND writes), refits ratings over the
whole record, and writes run/evals/round_<N>.json. A crash here fails loudly
in its own log but never takes down the run: the learner only polls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from ginrl.config import HandConfig, Seeds  # noqa: E402
from ginrl.env.features import feature_dim  # noqa: E402
from ginrl.env.game import HandEnv  # noqa: E402
from ginrl.env.match import MatchConfig  # noqa: E402
from ginrl.eval.arena import Arena  # noqa: E402
from ginrl.eval.belief import belief_auc, collect_belief_states  # noqa: E402
from ginrl.eval.ratings import fit_ratings  # noqa: E402
from ginrl.eval.record import append_summary, load_deals  # noqa: E402
from ginrl.eval.styles import profile  # noqa: E402
from ginrl.nets.gin import GinNet  # noqa: E402
from ginrl.train.match import (  # noqa: E402
    ANCHOR,
    _learner_agent,
    ladder_members,
    match_win_rate,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--deals", type=int, default=100)
    ap.add_argument("--belief-n", type=int, default=500)
    ap.add_argument("--probe-matches", type=int, default=20)
    ap.add_argument("--target", type=int, default=100)
    args = ap.parse_args()

    try:
        os.nice(10)
    except OSError:
        pass
    torch.set_num_threads(1)
    run = Path(args.run)
    hand_config = HandConfig()
    deck = hand_config.deck_size
    feat_dim = feature_dim(deck)
    dev = torch.device("cpu")
    net = GinNet("mlp", feat_dim=feat_dim, deck=deck, hidden=128)
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    # Tolerate a training-envelope checkpoint ({"net": state, ...}); the
    # canonical snapshot format is the raw state dict.
    net.load_state_dict(blob["net"] if isinstance(blob, dict) and "net" in blob else blob)
    net.eval()

    match_config = MatchConfig(hand=hand_config, target_score=args.target, max_hands=50)
    arena = Arena(config=hand_config, seeds=Seeds(args.seed))
    learner = _learner_agent(net, feat_dim, dev, args.agent, args.seed)
    record_path = run / "game_record.jsonl"
    config_hash = f"eval:{args.round}"
    summaries = {}
    failed = []
    for member in ladder_members(hand_config, args.seed):
        try:
            summary = arena.duplicate_summary(learner, member, args.seed, args.deals)
        except Exception as e:  # one bad member must not void the round
            failed.append({"member": member.name, "error": str(e)[:200]})
            print(f"[eval r{args.round}] {member.name} FAILED: {str(e)[:120]}", flush=True)
            continue
        summaries[member.name] = summary
        append_summary(record_path, summary, config_hash)
    out: dict = {"round": args.round, "members_failed": failed}
    if ANCHOR not in summaries:
        out["error"] = "anchor crashed; no refit"
        (run / "evals").mkdir(exist_ok=True)
        (run / "evals" / f"round_{args.round}.json").write_text(json.dumps(out, indent=2))
        print(f"[eval r{args.round}] ANCHOR FAILED, skipping refit", flush=True)
        return 0
    anchor_pph = summaries[ANCHOR].points_per_hand().mean
    fit = fit_ratings(load_deals(record_path), anchor=ANCHOR)
    games = []
    for summary in summaries.values():
        for pair in summary.pairs:
            games.extend([pair.leg1, pair.leg2])
    prof = profile(args.agent, games, lambda: HandEnv(hand_config, Seeds(args.seed)))
    states = collect_belief_states(hand_config, args.belief_n, seed=args.seed * 7 + 1)
    aux_auc, _ = belief_auc(net, states, dev)
    wmean, wlo, whi = (
        match_win_rate(net, feat_dim, dev, match_config, args.seed, args.probe_matches)
        if args.probe_matches > 0
        else (0.0, 0.0, 0.0)
    )
    out.update(
        {
            "anchor_pph": anchor_pph,
            "elo": fit.elo(args.agent),
            "elo_ci": list(fit.elo_ci(args.agent)),
            "cyclic_fraction": fit.cyclic_fraction,
            "first_player_edge_elo": fit.edge_elo,
            "gin_rate": prof.gin_rate,
            "knock_rate": prof.knock_rate,
            "undercut_rate": prof.undercut_rate,
            "wall_rate": prof.wall_rate,
            "mean_turns_to_knock": prof.mean_turns_to_knock,
            "mean_deadwood_at_knock": prof.mean_deadwood_at_knock,
            "pile_draw_rate": prof.pile_draw_rate,
            "danger_discard_rate": prof.danger_discard_rate,
            "belief_auc": aux_auc,
            "match_win_mean": wmean,
            "match_win_lo": wlo,
            "match_win_hi": whi,
        }
    )
    (run / "evals").mkdir(exist_ok=True)
    (run / "evals" / f"round_{args.round}.json").write_text(json.dumps(out, indent=2))
    print(
        f"[eval r{args.round}] pph={anchor_pph:+.2f} elo={out['elo']:+.0f} "
        f"win={wmean:.2f} auc={aux_auc:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
