# RESULTS — gated measurements ledger

Every number here comes from `runs/<run>/metrics.jsonl` (the source of truth)
or a gate's stdout on a committed tree. Nothing is scraped from a dashboard.
Append-only: new runs add sections, never rewrite old ones.

## Phase 5, first full gate (RED — strength, not harness)

- Run: `runs/p5_s0` — 3M steps fresh + 2M resumed, seed 0, torso mlp,
  uniform magnet, reg_coef 0.1, lr 3e-4 linear, reward_scale 0.02,
  n_envs 16, rollout 128. Trained under `8796616`/`08f9a49`, evaluated
  under `08f9a49`/`75b3d4e` (worker crash fixes) — see git log.
- Champion: `runs/p5_s0/best.pt` (copy of `snapshots/round_2400.pt`).
- Gate: `scripts/gate_p5.py --run runs/p5_s0 --agent p5s0 --br-dir runs/br_p5`
  (cells: 300k steps each, OMP=MKL=1, CPU — CPU beat MPS 8x at cell batch:
  infer-2048 0.8 vs 1.5ms, update-512 1.1 vs 8.9ms). Exit 1, 5/10.
- Snapshot trend vs simple_bot (500 duplicate deals):
  r500 -45.02, r1000 -40.22, r1400 -21.14, r1465 -21.52,
  r2000 -17.92, r2400/best -17.54, final -17.02. Climbing, decelerating
  (~1 pt/Mstep at the end; parity ~15M steps out at that rate).

| check | result |
|---|---|
| beats-anchor (20k deals) | -16.41 [-16.61,-16.21] FAIL |
| match-win-rate to 100 (600 matches) | 0.020 [0.010,0.032] FAIL |
| knock-moves-with-score | thin buckets lose=116 win=5 FAIL (unmeasurable: champ almost never ahead) |
| beats-heuristic (2k deals) | -14.62 [-15.27,-13.99] FAIL |
| rlbr-bound {champ,heur,bot,random} | {-11.88,-35.02,-49.34,+12.55} FAIL (champ not least exploitable) |
| elo-rising | +326 (margin 50) PASS |
| cyclic-ok | 0.000 PASS |
| first-player-edge | [-0.0,+13.0] PASS |
| logging-overhead | peak 0.0003 PASS |
| reproducible-eval | bit-identical PASS |

- Sensitivity at 5M+1M (single seed, common start `checkpoint.pt`, 500-deal
  readout vs anchor): A control decay-to-zero -16.03, B magnet-off -15.54,
  C steady-lr-1e-4 -16.52 — all within noise (±1.3); C's entropy collapsed
  0.48->0.22. Verdict: LR starvation and magnet drag both rejected as levers;
  continue the status-quo recipe on a fresh schedule. (The 5M endpoint had
  lr=reg=0 by construction — frozen, not converged.)
- Calibration verdict: no `gates.yaml` value changed. The five failures are
  strength gaps (champion loses to anchor/heuristic by ~15 pph, wins 2% of
  matches); moving a threshold to meet them would be editing the gate to
  pass. `simplebot_pph_margin: 0.0` stands as the bar, `elo_gain_margin: 50`
  is validated achievable (+326), `knock_min_bucket_n: 30` stays — a
  competitive champion must fill both buckets. RL-BR reading: 300k-step
  from-scratch learners cannot touch full-deck heuristic/bot (bounds -35/-49
  vs their own -20/-21 head-to-head), so these bounds are loose lower bounds;
  the ordering signal (champ bound above heur bound = champ weaker) is the
  usable part and agrees with head-to-head.
