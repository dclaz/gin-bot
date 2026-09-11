# Metrics guide — what the dashboard is showing

Source of truth is `runs/<run>/metrics.jsonl`; Trackio is just the viewer.
Two cadences: **every round** (~17k points — dense training curves) and
**every eval** (~160 points — sparse readout dots). Eval metrics come from
duplicate-deal games against the reference bot, heuristic family, and past
checkpoints, so they are low-variance and comparable across time.

## The one chart to watch

**`ratings/points_per_hand_vs_anchor`** — mean paired score per hand vs
`SimpleGinRummyBot`, with CI. Game units, no model, no tuning knobs. If
this is going up, the agent is getting better. Everything else is
diagnostics for *why* (or *why not*).

## loss/ — is the optimiser minimising (every round)

| metric | meaning |
|---|---|
| `loss/policy` | PPO policy loss (advantage-weighted log-probs, clipped). Should trend down early, then wander. |
| `loss/value` | Critic error (predicted vs actual return). Down = the critic understands the game. |
| `loss/belief_bce` | Opponent-hand prediction error (auxiliary head). Down = history features are being used. |
| `loss/total` | Weighted sum the optimiser actually steps on. |

## reg/ — regularisation toward the magnet (every round)

| metric | meaning |
|---|---|
| `reg/kl_to_magnet` | How far the policy has drifted from the magnet (uniform) policy. Rises as it learns something specific. |
| `reg/alpha` | Magnet pull strength. Anneals to 0 on schedule — learning is "done" when this is 0. |
| `reg/clip_frac` | Share of decisions where PPO clipping bit. Persistently high = updates are fighting the clip range. |

## policy/ — is it still exploring (every round)

| metric | meaning |
|---|---|
| `policy/entropy` | Randomness of the policy. Falls as the agent gets decisive; a collapse to ~0 early means it stopped exploring. |

## value/ — is the critic learning (every round)

| metric | meaning |
|---|---|
| `value/explained_variance` | Fraction of return variance the critic explains (1 = perfect, 0 = useless, negative = worse than guessing). The critic-health number. |

## opt/ — optimiser state (every round)

| metric | meaning |
|---|---|
| `opt/lr` | Current learning rate (linear anneal to 0 — frozen at 0 means the schedule ended, not that it converged). |
| `opt/grad_norm` | Gradient size. Spikes or a slide to 0 both deserve a look. |

## belief/ — is history reaching the network (every eval)

| metric | meaning |
|---|---|
| `belief/auc` | Opponent-hand prediction quality (0.5 = guessing, 1 = perfect). Flat near 0.5 = the history features are being ignored. |

## ratings/ — is it getting better (every eval)

| metric | meaning |
|---|---|
| `ratings/paired_score` | Same-duel readout as the headline, per eval snapshot. |
| `ratings/current_elo` | Elo vs the ladder (anchor, heuristics, past selves), with `ratings/elo_ci_width`. Secondary to points-per-hand; useful once many agents exist. Never read without `ratings/cyclic_fraction`. |
| `ratings/cyclic_fraction` | How rock-paper-scissors the pool is (0 here = transitive, Elo can be believed). |
| `ratings/first_player_edge_elo` | Measured first-seat advantage (~+6 Elo — a real gin property, first move on the upcard — not a bug; gate tolerance is ±25). |

## match/ — episode health (rounds) and the real objective (evals)

| metric | meaning |
|---|---|
| `match/hands`, `match/matches`, `match/caps` | Cumulative counters — training is generating full matches, not stalling. `caps` = matches hitting the hand cap. |
| `match/learner_win_frac` | Training win rate **including the sparring mix** (mirror, past selves, heuristics). A plumbing heartbeat, not a strength claim. |
| `match/win_rate_simplebot` (+`_lo`/`_hi`) | Duplicate matches to 100 vs the anchor with CI. Lower variance than Elo; the match is the actual objective, so where this disagrees with points-per-hand, this one is right. |
| `match/eval_lag_rounds` | How far behind the eval workers are. Rising = evaluators can't keep up. |
| `match/eval_workers_failed` | Crashed eval snapshots (non-fatal; training continues). Should stay flat after the Newton fix. |

## style/ — what kind of player is emerging (every eval)

The research question, live. Not decoration.

| metric | meaning |
|---|---|
| `style/gin_rate` | Share of wins that are gin (big bonus) vs knock. Rising = aiming for gin. |
| `style/knock_rate` | How often it knocks. The other half of the gin-vs-knock tradeoff. |
| `style/undercut_rate` | How often its knocks get undercut. High = knocking too loose. |
| `style/wall_rate` | Hands ending in a wall (deck exhaustion). High = passive play. |
| `style/mean_turns_to_knock` | Speed. Falling = knocking earlier. |
| `style/mean_deadwood_at_knock` | Leniency. Rising = knocking with worse hands. |
| `style/pile_draw_rate` | Upcard appetite. |
| `style/danger_discard_rate` | Discards the opponent demonstrably wants. Falling = safer discards. |

Plot `style/gin_rate` against `ratings/current_elo` and the
knock-early-vs-aim-for-gin answer draws itself.

## perf/ — is the machine being used (every round)

| metric | meaning |
|---|---|
| `perf/rows_per_sec` | End-to-end training throughput (collect + update). ~1,800 single-process, ~3,600 with the 8-worker pool. |
| `perf/logging_overhead_frac` | Share of round time spent logging (gated < 0.02 — telemetry must not perturb the run). |

## Not on this dashboard (yet)

`population/*` (pool transitivity — Phase 6 tournament),
`tripwire/*` thresholds, per-phase entropy, and value match forecasts are
defined in `docs/OBSERVABILITY.md` but not emitted by Phase 5 training.
No second dashboard will be added; new series appear here when a phase
needs them.
