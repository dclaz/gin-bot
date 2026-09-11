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
- Second full gate at 15M (frozen `runs/gate15M`, cells `runs/br_p5_15M`):
  4/10. beats-anchor -9.94 [-10.13,-9.74], matches 0.068 [0.048,0.090],
  knock buckets 189/25 (win bucket near the 30 minimum), heuristic -8.09
  [-8.72,-7.47], champ BR bound -18.44 (from -11.88: less exploitable),
  elo +471, cyclic 0.000, overhead 0.0003, repro bit-identical. New FAIL:
  first-player-edge [+3.0,+8.6] — the CI no longer covers 0. That is a
  sample-size artifact of the criterion, not a new problem: 3x the legs
  resolved a real but tiny +6 Elo seat effect (50.9/49.1) that the
  CI-covers-0 form must eventually fail for ANY nonzero edge as data grows.
  Duplicate comparisons swap seats so nothing is biased. Proposed (not done):
  tolerance form |edge| < 25 Elo; needs human sign-off since it flips a
  check to PASS. Count went 5/10 -> 4/10 while every strength number
  improved — the count is not the trend.
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
- Third full gate at 25M (`runs/p5_s0`, 25,001,984 steps, best
  `snapshots/round_12000.pt`): 5/10 pass. beats-anchor -7.28
  [-7.47,-7.09] n=20000 (from -9.94), matches 0.135 [0.108,0.163] n=600
  (from 0.068), knock buckets PASS 167/55 (win bucket filled),
  heuristic -5.11 [-5.74,-4.50] (from -8.09), champ BR bound -29.93 vs
  heur -35.02 vs bot -49.34 (ordering champ > heur > bot preserved; the
  champ bound moved down from -18.44, read as a tighter lower bound from
  more BR training, not a regression — the usable ordering signal is
  unchanged), elo +530 (from +471), cyclic 0.000, overhead 0.0003, repro
  bit-identical. first-player-edge [+4.8,+9.1] FAIL — same +6ish Elo seat
  effect as 15M's [+3.0,+8.6], still inside the proposed |25| tolerance;
  sign-off still pending, gate file untouched. Trend: +2.7 anchor / +3.0
  heuristic pph per 10M steps; linear arithmetic points at heuristic
  parity ~42M and anchor parity ~52M — a projection, not a promise.
- Fourth full gate at 50M (`runs/p5_s0`, 50,001,920 steps, best
  `snapshots/round_23400.pt`, first 25M single-process + last 25M on the
  8-then-4-worker pool): 6/10 pass. beats-anchor -4.78 [-4.97,-4.59]
  n=20000 (from -7.28), matches 0.247 [0.213,0.280] n=600 (from 0.135),
  knock buckets PASS 134/101, heuristic -2.18 [-2.78,-1.65] (from -5.11),
  champ BR bound -33.86 vs heur -35.02 vs bot -49.34 (ordering preserved),
  elo +591 (from +530), cyclic 0.000, first-player-edge [+4.2,+7.1] PASS
  under the signed-off |25| tolerance, overhead 0.0003, repro
  bit-identical. Zero eval-worker failures across the whole pool segment.
  Slope 25M->50M fell to +0.10 anchor pph/M (from +0.27): the grind works
  but is decelerating — current-slope parity sits past 100M. Schedule
  exhausted (alpha/lr ~ 0); entropy has no bonus term, so post-anneal the
  policy can only sharpen. Critic explained variance flat ~0.49.
- Post-50M probes (common start: 50M checkpoint; 500-deal anchor readout,
  control -4.86 [-6.11,-3.63] reproduces the gate -4.78): B, magnet-alive
  schedule (resume total=100M, 2.5M steps): -11.41 [-12.66,-10.18] — the
  revived uniform-magnet drags the converged policy back toward random
  faster than it can re-learn. Magnet schedules must decrease
  monotonically; this also kills the resume-to-100M grind (same reheat
  mechanics). D'', vf_coef 0.5->1.0 on the dead schedule (3M steps):
  -4.99 [-6.21,-3.85], identical to control; explained variance 0.33->0.23
  (no lr, no learning either way). Post-convergence schedule tweaks are
  dead ends. The 50M champion stands on this recipe; remaining levers are
  an earlier-checkpoint reheat, capacity reopen, or accepting into Phase 6.
