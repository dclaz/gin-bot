# Implementation Plan

A phased build for a state-of-the-art Gin Rummy agent on OpenSpiel, designed to
be executed by Claude Code with minimal supervision. Every phase ends in an
executable gate. The gate, not the agent's judgement, decides whether the phase
is finished.

---

## 0. How this plan validates and corrects itself

Three layers, in increasing cost and decreasing frequency.

**Layer 1 — Invariants.** Fast tests that run on every commit (`make test`,
target < 60s). Unit tests, property tests against the engine, and a drift check
on generated facts. These catch "the code no longer means what it said".

**Layer 2 — Gates.** One per phase (`make gate-pN`). Slower (seconds to tens of
minutes), thresholded, and *append-only*: a gate may be added, never weakened,
except via a commit whose message gives the old value, the new value and why.
`make gate-all` runs every gate in order and is the release check. Thresholds
live in `configs/gates.yaml`; `make status` prints that file's SHA-256 so
silent loosening is visible in the run log.

**Layer 3 — Tripwires.** Assertions inside long training runs, checked every
`eval_every` steps. A tripwire fires, dumps state to
`runs/<id>/tripwire_<step>.json`, and halts. Halting a bad run at minute 8
instead of hour 6 is the single highest-return piece of engineering in this
project. See §Tripwires.

**Underneath all three — Telemetry.** Not a fourth layer: it is what the three
above read. Continuous scalars, histograms and ratings streamed to
`runs/<run>/metrics.jsonl` (source of truth) and Trackio (dashboard). Tripwires log both their current value and their threshold, so
an approaching failure is visible for an hour before it fires. Full spec in
`docs/OBSERVABILITY.md`.

**The correction loop.** When a gate fails, the agent must, in order:
(1) reproduce with the seed printed by the gate; (2) write a one-paragraph
diagnosis *before* editing code, and carry it into the fix's commit message;
(3) fix forward or
`git revert`; (4) re-run the gate. Three failed diagnoses on the same gate is a
stop-and-ask condition. Editing the gate is never step 3.

**Generated facts.** `make probe` runs `scripts/probe_env.py`, which interrogates
the live engine and hardware and writes `docs/ENV_FACTS.md` (human) and
`src/ginrl/spiel_facts.py` (machine). All code imports constants from the latter.
`make facts-check` regenerates and diffs; drift fails the build loudly. This is
what stops the agent re-deriving the action layout in every session.

---

## What exists

**The documents, and nothing else.** This tree contains `README.md`,
`CLAUDE.md`, `METHODOLOGY.md`, this plan, `docs/OBSERVABILITY.md` and
`docs/OBSERVABILITY.md`. There is no `src/`, no `Makefile`, no `configs/`, no
`tests/`, no `pyproject.toml`, no `scripts/`, and no commit on `main`.

An earlier session prototyped `scripts/probe_env.py`,
`src/ginrl/telemetry/recorder.py`, `src/ginrl/eval/{ratings,report_ratings,population}.py`,
`configs/gates.yaml` and a test suite. **Those files were never committed and are
not recoverable.** What they were built to is written down:
`docs/OBSERVABILITY.md` § Ratings is the specification for the ratings and
population modules, and § Stack for the Recorder. Every artifact named in the
phases below is to be written.

So: `make` does not exist yet either. Phase 0 creates the Makefile, and until it
does, the commands in `CLAUDE.md` describe the target state rather than the
current one.

**A note on the file paths below.** They are a suggested layout, not a
contract. Imports are absolute from `src/ginrl/`; beyond that, split or merge
modules as the code wants. No gate checks a filename. What the gates check is
behaviour.

---

## Phase 0 — Environment lock-in

**Goal.** Make the environment a known quantity, once, and write it down.

**Do.** Phase 0 bootstraps its own tooling. Nothing below exists yet, so build
it in this order.

1. `uv init`, `uv python pin 3.13`, then `uv add`:
   - runtime: `open-spiel==2.0.2` (pinned — see Expected surprises), `torch`,
     `numpy`, `scipy` (the Bradley-Terry fit, the Hodge decomposition and Nash
     averaging's LP all use it), `pyyaml` (reads `configs/gates.yaml`),
     `trackio`;
   - dev: `pytest`, `ruff`.
   Anything beyond this list needs a commit message saying what capability is missing
   — see CLAUDE.md, "Do not". Commit `uv.lock`.
2. **Write the `Makefile`.** Every target named in CLAUDE.md's Commands section:
   `setup`, `probe`, `facts-check`, `test`, `lint`, `gate-p0`..`gate-p7`,
   `gate-all`, `status`, `board`, `elo`. Gates for
   phases not yet reached should exit non-zero with "not implemented", never 0.
   Export `PYTORCH_ENABLE_MPS_FALLBACK=1`, and write the `.env` that CLAUDE.md
   says carries it.
3. **Write `configs/gates.yaml`** with every threshold this plan names, and an
   explicit `TODO` for each one that is uncalibrated (Phase 4 onward). `make
   status` prints its SHA-256, which is the only record that a threshold has not
   been quietly loosened — record it in the commit that closes the phase.
4. **Write `tests/test_repo_consistency.py` first**, before any other test. Every
   defect in the 2026-09-07 review was cross-file drift, and this is the file
   that catches that class: Makefile gate names match `gates.yaml` keys in both
   directions, make targets cited in docs exist, no removed ruff rules in the
   ignore list, and the tiering decisions hold. Parse structure — TOML, YAML,
   non-comment lines — do not grep file contents, or explanations get read as
   rules.
5. Write `scripts/probe_env.py`. It records: platform, arch, CPU core counts by
   performance level, Python, torch, MPS availability and bf16 support,
   `open_spiel` version, and — the important part — the gin rummy game facts:
   action count, observation size, parameter defaults, the full action-id →
   action-string table, which tensors and struct methods exist, whether
   `resample_from_infostate` works, and a CPU-vs-accelerator microbenchmark.
   *The real torso does not exist until Phase 3, and Phase 4 compares four of
   them*, so benchmark a representative MLP at the configured batch sizes as a
   stand-in and re-run the device decision per architecture in Phase 4.
6. Emit `docs/ENV_FACTS.md` and `src/ginrl/spiel_facts.py`.
7. Add `make facts-check` to CI and to the pre-commit path. Confirm Trackio's
   CLI syntax while you are here — the docs cite `trackio list` / `trackio get` /
   `trackio query --sql`, unverified on this machine; fix the docs if it differs.

**Gate `gate-p0`.**
- `make lint test` passes, and `make gate-p0` itself runs — the tooling this
  phase builds is the first thing the phase proves.
- `configs/gates.yaml` exists and every gate name in the `Makefile` resolves to a
  key in it, in both directions.
- `uv run python -c "import pyspiel, torch"` succeeds.
- `docs/ENV_FACTS.md` and `src/ginrl/spiel_facts.py` exist and are newer than
  `pyproject.toml`.
- `make facts-check` exits 0.
- Asserted engine facts hold: 241 actions, 644 observation dims,
  `information_state_tensor` raises, `GinRummyUtils` importable,
  `make_simple_gin_rummy_bot` constructs.
- The device benchmark wrote a recommendation and it is one of `cpu` / `mps`.

**Expected surprises.** None on the wheel — `open-spiel==2.0.2` has a
cp313/macosx_11_0_arm64 wheel. If a future version drops it, pin to 2.0.2 rather
than building from source.

---

## Phase 1 — Game layer

**Goal.** A correct, tested, history-aware view of the game. This is the phase
that determines the ceiling of everything after it.

**Do.**
1. `ginrl/env/game.py` — thin wrapper: load game with parameters, step, terminal
   returns, legal action masks, seat swapping, and *duplicate deal* support (the
   same shuffled deck replayed with seats reversed).
   `ginrl/env/match.py` — **`MatchEnv`**, successive hands to a target score
   (default 100, the North American and EAAI standard). The match is the
   episode; hand scores are intermediate rewards; the running score is part of
   the state. Duplicate evaluation extends to **duplicate matches**: the same
   seed sequence replayed with seats swapped. Hand-level play stays available
   and is what Phases 3-4 use. See METHODOLOGY §1.
2. `ginrl/env/melds.py` — a thin adapter over `pyspiel.gin_rummy.GinRummyUtils`.
   No original meld logic.
3. `ginrl/env/meld_policy.py` — a **rule-based policy for the meld and layoff
   phases**. When the game asks which melds to lay or which cards to lay off,
   the answer is a deterministic optimisation, not a strategic choice: lay the
   best meld group (`best_meld_group`), lay off every card that reduces deadwood.
   Hard-coding this removes 185 of 241 actions from the learned policy and is the
   largest single win available in this project. The learned policy then covers
   only: draw upcard, draw stock, pass, knock, and 52 discards.
   - Caveat to test, not assume: laying a *sub-maximal* meld set can occasionally
     deny the opponent a layoff. Phase 6 includes an ablation that lets the
     network choose melds; if it wins by more than the gate's noise floor, revisit.
4. `ginrl/env/features.py` — `BeliefTracker`, a per-seat, per-episode accumulator
   built from `state.to_observation_struct(p).to_dict()` plus the action stream.
   **The list below is a floor, not a specification.** It is what we are confident
   the agent cannot play without; adding to it is expected, and the two gates
   below — information hygiene and history necessity — are the actual contract.
   The negative-information channel is the item most easily missed and the one
   most worth getting right. At minimum it maintains:
   - own hand (52), current best meld decomposition, deadwood, and per-card
     "cost to discard" / meld-adjacency counts;
   - discard pile contents and order, top card, stock size, turn index;
   - **cards known to be in the opponent's hand** (they drew them from the pile
     and have not discarded them);
   - **cards the opponent declined** (passed on the first upcard, or left on the
     pile when it was their turn) — the negative-information channel;
   - cards provably in neither hand (discarded and buried);
   - knock-legality flags and distance-to-knock;
   - **match state**: own score, opponent score, differential, points to target.
     Without these the policy cannot be score-dependent, and the score-dependent
     knock threshold is the interesting half of the research question.
5. `ginrl/env/vector_env.py` — batched stepping across N independent games in one
   process, so inference is batched. Multi-process actors come later, and only if
   Phase 5's profile says so.

**Gate `gate-p1`.**
- **Meld oracle agreement.** Over 20 000 random terminal states, our computed
  deadwood and meld group agree with the engine's own scoring; `returns()` is
  reproduced from our numbers to the cent.
- **Zero-sum.** `sum(returns()) == 0` on 50 000 random playouts.
- **Action round-trip.** For all 241 actions, our encoder/decoder round-trips
  against `state.action_to_string`.
- **Mask soundness.** The legal mask is never all-zero at a decision node, and
  never marks an action the engine rejects (fuzzed over 100 000 states).
- **Information hygiene.** For 5 000 states, resample the hidden state with
  `resample_from_infostate` 32 times; the feature vector must be *bit-identical*
  across resamples. Any variation is a leak of the opponent's hand into our
  features and fails the gate. This is the single most important test in the repo.
- **History necessity.** A held-out probe classifier trained on raw observation
  tensors must do measurably worse at predicting "is card c in the opponent's
  hand" than one trained on `BeliefTracker` features. If it does not, the tracker
  is buggy or the feature set is redundant — investigate before proceeding.
- **Throughput recorded, and no regression.** Measure feature-extracted
  steps/s/core, write it to `docs/ENV_FACTS.md`, and fail only on a >2×
  regression against that recorded baseline on the same machine. There is no
  absolute floor: the number is hardware-dependent, and a correct implementation
  must not fail this gate for running on a slower box. (Reference point: 36 000
  *raw* steps/s/core measured on x86 without features.) Landmine 12 is the reason
  this is a regression check and not a target — the environment is not the
  bottleneck, so optimising it is usually the wrong move.

---

## Phase 2 — Opponents and the evaluation harness

**Goal.** Be able to answer "is A better than B, and by how much, with what
confidence" before there is anything to evaluate. Built second, not last.

**Do.**
1. `ginrl/agents/` — a common `Agent` protocol, then: `RandomAgent`,
   `SimpleBotRunner` (wrapping `make_simple_gin_rummy_bot` with the correct
   fresh-per-game + `inform_action` protocol), `HeuristicAgent` parameterised by
   `(knock_threshold, discard_danger_weight, pile_draw_aggression)`, and
   `ISMCTSAgent` (a determinized search baseline via `resample_from_infostate`).

   **On external benchmarks.** The realistic opponent set is in-process and
   already available; do not take on a second engine or a JVM bridge for this.
   - `SimpleGinRummyBot` — the anchor. Almost certainly a port of
     `SimpleGinRummyPlayer` from the EAAI reference implementation, which would
     make our 0-Elo anchor the field's own reference player. Worth confirming in
     Phase 2 by comparing behaviour, and worth recording either way.
   - **`pyspiel.ISMCTSBot` at several fixed budgets** — the most valuable
     benchmark available, for two reasons. It is the published baseline used for
     gin rummy in the 2026 gold-standard study, so a number against it is
     comparable to the literature; and budget is a *dial*, which gives a ladder
     of increasing strength rather than a single pass/fail opponent. Pin the
     budgets in config and treat them as distinct ladder members.
   - RLCard's `gin-rummy-novice-rule` and the EAAI Java bots (Heisenbot and the
     tournament field) are **not** worth wiring in: RLCard implements a
     different rule set, so cross-engine scores are not comparable, and the
     EAAI code is Java with no stated licence. If an external comparison is ever
     wanted, the cheap version is to match OpenSpiel's parameters to EAAI's
     (they already agree: gin 25, undercut 25, knock 10, match to 100) and cite
     published win rates rather than run their code.
2. `ginrl/eval/arena.py` — head-to-head with **duplicate deals**: every deck is
   played twice with the seats swapped, and the two results are paired. This is
   common random numbers and it typically cuts the variance of a score
   difference by a large factor in a game with this much luck. Report paired
   mean score/game, win rate, and a bootstrap CI.
3. `ginrl/eval/stats.py` — sequential testing helper that answers "how many
   duplicate deals do I need to resolve a 0.5 point/game difference at 95%",
   and refuses to report a result below that count.
4. `ginrl/eval/styles.py` — the behavioural profile computed for any agent:
   gin rate, knock rate, undercut rate, mean turns to knock, mean deadwood at
   knock, deadwood-over-time curve, pile-draw rate, rate of discarding a card
   the opponent is known to want.
5. `ginrl/telemetry/recorder.py` — the fan-out Recorder, wired into the arena so
   every later phase logs through it from the start. Two sinks, JSONL and
   Trackio; do not add a third. Spec in `docs/OBSERVABILITY.md`.
6. `ginrl/eval/ratings.py`, `report_ratings.py`, `population.py` — anchored
   Bradley-Terry with CIs, the Helmholtz-Hodge diagnostic, and Nash averaging.
   Wire them into the arena and tournament scripts and add the `tests/gates/`
   assertions that exercise them. `docs/OBSERVABILITY.md` § Ratings is the
   specification, down to the measured anchors the tests should reproduce.
7. `runs/game_record.jsonl` — the append-only record of every evaluation game
   ever played. Ratings are always refit from this file, never accumulated in
   memory, so any rating is reproducible from disk.

**Gate `gate-p2`.**
- Random vs `SimpleGinRummyBot`: the bot wins by a margin whose CI excludes zero,
  with the direction and rough magnitude recorded as a regression baseline.
- `HeuristicAgent` at its default config beats `RandomAgent` decisively.
- **Duplicate-deal variance check.** For the same matchup and sample count,
  paired duplicate evaluation has materially lower variance than unpaired. If
  not, the pairing is wired wrong.
- **Null calibration.** An agent evaluated against a copy of itself over 10 000
  duplicate deals returns a mean score whose CI contains 0. A harness that finds
  a seat advantage against a clone is broken.
- Behavioural profiler runs on every baseline and produces a table.
- Reference numbers to reproduce from Phase 0's probe: `SimpleGinRummyBot`
  self-play averages ≈ 34 decisions/game and ≈ 16.4 points/hand in magnitude, with
  no wall/draw endings.
- **Rating recovery.** On synthetic games generated from known Elos, the fit
  recovers them within the configured tolerance, and the CI covers the truth at
  the nominal rate.
- **Cycle detection.** A synthetic rock-paper-scissors population yields
  `cyclic_fraction ≈ 0.9994`; a synthetic transitive ladder yields `≈ 0.0006`. Both
  are unit tests, because this is the number that decides whether Elo may be
  believed.
- **Undefeated agent.** An agent with a perfect record gets a finite rating.
  Without the L2 prior the MLE diverges; the test pins this.
- **Seat effect.** With correct mirroring, `first_player_edge_elo` sits within
  its CI of zero. A non-zero value here means the harness is not mirroring —
  this is a free correctness check on Phase 2 itself.
- **Telemetry smoke.** A short run produces a JSONL with header, scalars,
  histogram and table records, and killing the Trackio sink mid-run does not
  kill the run.

---

## Phase 3 — Trainer core, validated where truth is computable

**Goal.** A regularised policy-gradient self-play trainer that is *proven* to
converge, on games where convergence can be measured exactly. Gin rummy is not
one of those games, so it is not used here.

**Do.**
1. `ginrl/algos/regpg.py` — one trainer, three regularisation modes selected by
   config (see METHODOLOGY.md §3 for the derivation):
   - `magnet=uniform` → PPO with an entropy bonus, the Rudolph et al. baseline;
   - `magnet=snapshot` → MMD proper: KL to a magnet policy refreshed every
     `magnet_every` updates;
   - `magnet=ema` → magnet as an EMA of the learner's parameters.
   **Only `snapshot` is required to pass `gate-p3`.** It is MMD as published and
   it is what the ablation guard below contrasts against. Build it, clear the
   gate, then add `uniform` and `ema` — they are cheap once the trainer works,
   and expensive as three unvalidated implementations blocking the same gate.
   Shared: clipped surrogate or NeuRD-style logit update, GAE, action masking
   applied to logits before the softmax, annealed learning rate and
   regularisation coefficient (linear and power-law schedules).
2. `ginrl/nets/` — a shared torso feeding three heads: policy over the reduced
   action set, value, and an auxiliary **opponent-hand head** (52-way multilabel
   BCE). *The three heads are fixed* — the belief head backs a tripwire and the
   Phase 6 inspectability story. *The torso is a default*: a residual MLP over
   the flat feature vector is the baseline, but the discard pile is an ordered
   sequence and the hand is a set, so a sequence or set encoder is a legitimate
   thing to try. Judge it on `gate-p4`, `belief/auc` and `perf/`, not on
   resemblance to the baseline. Keep it small enough that CPU-vs-accelerator is a
   real question; the probe decides which.
3. Self-play driver: both seats share weights; seats alternate; returns are
   recorded per seat.
4. `ginrl/eval/exploitability.py` — wrapper over OpenSpiel's exact `nash_conv`
   for the small games.
5. `ginrl/eval/rlbr.py` — **RL best response, built here rather than in Phase 5.**
   It is itself a training run and can be undertrained, mistuned or simply
   broken, and every one of those failure modes reads as "the champion is hard
   to exploit". Kuhn and Leduc are the only place its answer can be checked
   against a known one, so it is built and calibrated here, against
   `exploitability.py`, before it is ever pointed at gin rummy.
6. Wire the Recorder into the trainer: `loss/`, `reg/`, `policy/`, `value/`,
   `opt/` and `perf/` namespaces, plus exploitability as a scalar on the small
   games. Seeing exploitability descend on Leduc in a live chart is the fastest
   way to know the trainer works.

**Gate `gate-p3` — the ladder.** All thresholds in `configs/gates.yaml`.
- **Kuhn poker.** Exploitability drops below 0.01 within the configured budget.
  Anchor: the uniform random policy has NashConv 0.9167 — verified, use it as a
  unit-test constant.
- **Leduc poker.** Exploitability drops below 0.20 within budget. Anchor:
  uniform random NashConv is 4.7472.
- **Last-iterate check.** Exploitability of the *final* iterate, not a running
  average, is what is measured. This is the property regularisation is bought for.
- **Ablation guard.** The same trainer with regularisation disabled must do
  *worse* on Leduc. If it does not, the regularisation is not wired in.
- **Determinism.** Two runs with the same seed produce bit-identical
  checkpoints on CPU.
- **RL-BR calibration.** RL-BR against a fixed uniform-random Kuhn policy
  recovers the exact NashConv (0.9167) within the configured tolerance, and does
  the same on Leduc (4.7472). An RL-BR that cannot reproduce a known
  exploitability is not measuring exploitability, and the Phase 5 bound built on
  it would be decoration. This is the gate that makes the project's headline
  worst-case number mean anything.

If this gate does not pass, nothing downstream is worth running. Do not proceed
by weakening it.

---

## Phase 4 — Reduced gin rummy

**Goal.** Run the real game's code path on a small deck, where runs are minutes
and bugs are visible.

**Do.**
1. Configure e.g. `num_ranks=5, num_suits=2, hand_size=3` (deck 10). Respect
   `num_ranks*num_suits >= 2*hand_size + WALL_STOCK_SIZE + 1`, importing
   `WALL_STOCK_SIZE` rather than retyping the 3.
2. Train with the Phase 3 trainer and the Phase 1 features.
3. **Architecture bake-off.** This phase exists because runs are minutes, which
   makes it the only affordable place to compare architectures rather than
   assume one. Train at least the four torsos in METHODOLOGY §3 — (A) residual
   MLP over flat features, (B) set encoder over card embeddings, (C) sequence
   encoder over the action history, (D) MLP on the raw 644-dim observation as
   the no-features control — under an equal gradient-step budget, three seeds
   each. The three heads are fixed across all four; only the torso varies.
4. Note: the action space stays 241 and the observation stays 644 even at this
   size — masking handles it; do not "optimise" the layout.

**Gate `gate-p4`.**
- Beats `RandomAgent` and `HeuristicAgent` on the reduced game with CIs excluding
  zero, over duplicate deals.
- The auxiliary opponent-hand head beats a **strong** baseline on held-out
  states: not marginal card frequency, but `BeliefTracker`'s own deterministic
  knowledge. The tracker already feeds the network the cards it *knows* the
  opponent holds, so a head that echoes that input scores well while having
  learned nothing. Score `belief/auc` only over cards the tracker has *not*
  determined — that is the set where inference is actually happening, and a flat
  AUC there is the tripwire worth having.
- Training is stable across three seeds: final-score spread within the configured
  band, no entropy collapse.
- **Architecture comparison reported, winner not prescribed.** A table over the
  four torsos: mean paired score with CI, `belief/auc` on undetermined cards,
  parameter count, and steps/s. Equal budget, three seeds, duplicate deals. The
  gate requires the table and the equal-budget discipline, *not* a particular
  winner — whichever torso wins carries forward, with its margin recorded in the
  commit that closes the phase.
- **The no-features control is informative either way.** If (D) is competitive
  with (A), METHODOLOGY §4's central premise — that hand-engineered structure
  beats learned embeddings here — does not hold in our rule set, and that is a
  finding to record in Phase 4 rather than discover in the write-up.
- A **spike, timeboxed to one session**: attempt exact `nash_conv` on the smallest
  legal config. It was killed by the OOM reaper in our probe on a much larger
  machine, so the expected outcome is "confirmed infeasible" — record that in
  the commit message and move on. Do not let this become a project.

---

## Phase 5 — Full game

**Goal.** The main agent, trained and evaluated on **matches** (METHODOLOGY §1).
Phases 3-4 were hand-level for speed; this is where the real objective starts.
The match is the episode, hand scores are intermediate rewards, and the match
score is in the state — a policy that cannot see the score cannot learn the
score-dependent knock threshold, which is the interesting half of the question.

**Do.**
1. **Re-benchmark CPU against MPS at the real model size and batch size, on
   this phase's actual torso, before starting any long run.** Phase 0's device
   recommendation came from a stand-in MLP; Phase 4's came from a reduced-deck
   model. Neither is evidence about the full-game network, and the ordering can
   invert with size — see landmine 9. Getting this wrong costs the whole run and
   the check costs minutes. Record both numbers in the run's config.
2. Profile the *feature path*, not raw stepping. The ~36k
   steps/s/core figure is raw `step` + observation + mask; it does not include
   `to_observation_struct().to_dict()` per seat per decision or `BeliefTracker`
   accumulation, which is the pipeline actually running. Measure that before
   concluding the environment is cheap. If inference still dominates, fix
   batching before adding processes. Multi-process actors (`spawn`, the
   accelerator confined to the learner process) only if the profile justifies it.
3. Train with annealed regularisation. Checkpoint on a schedule and keep the
   **best** checkpoint by ladder performance, not the last — a known
   several-point effect in this game.
4. Apply `ginrl/eval/rlbr.py` — built and calibrated in Phase 3 against known
   Kuhn/Leduc exploitability, so its answer here is a measurement rather than an
   assertion. Freeze the champion, train a fresh PPO agent against it as a
   single-agent MDP, report the value achieved. This is a lower bound on
   exploitability and the only tractable worst-case measure here. Also run the
   ISMCTS-BR variant for a second opinion.
5. Maintain a fixed **evaluation ladder** — random, simple bot, heuristic family,
   ISMCTS at fixed budget, previous champions — that never enters training.
   Every `elo_every` steps, play duplicate deals against each ladder member and
   a sample of past checkpoints, append to `runs/game_record.jsonl`, refit the
   Bradley-Terry model over the whole record anchored on `SimpleGinRummyBot = 0`,
   and log `ratings/current_elo` with its CI, `ratings/points_per_hand_vs_anchor`
   and `ratings/cyclic_fraction`. Run this off the learner's critical path.
6. Log the `style/` namespace throughout, so the emerging playing style — gin
   rate, turns to knock, deadwood at knock — is visible as it forms rather than
   measured once at the end.

**Gate `gate-p5`.**
- Beats `SimpleGinRummyBot` by a margin exceeding the configured threshold over
  ≥ 20 000 duplicate deals, CI excluding it.
- **Match win rate** against `SimpleGinRummyBot` over duplicate matches to 100,
  CI excluding 50%. Reported beside points/hand, never instead of it: points per
  hand is the low-variance training signal, the match is the objective.
- **The knock threshold moves with the score.** Mean deadwood at knock, bucketed
  by score differential, is not flat — a policy that knocks identically at +90
  and −90 has not learned the score-dependent policy and is a hand-level agent
  wearing a match-level wrapper. Report the curve; the gate wants a CI-separated
  difference between the extreme buckets.
- Beats the tuned `HeuristicAgent` and the previous champion.
- RL-BR bound on the champion is lower than the RL-BR bound on every baseline —
  i.e. the agent is not merely strong head-to-head but harder to exploit. Report
  the number; do not claim unexploitability.
- **Elo is rising and means something.** `ratings/current_elo` improves against
  the fixed anchor by the configured margin, with a CI that excludes the
  starting rating, *and* `ratings/cyclic_fraction` stays below its threshold
  throughout. A rising Elo with a high cyclic fraction fails this gate: it is
  the signature of an agent walking in a circle against its own past selves.
- `ratings/first_player_edge_elo` remains within its CI of zero.
- `perf/logging_overhead_frac` below 2% over the full run.
- No tripwire fired in the final run.
- The run is reproducible: rerunning from the recorded config + seed lands within
  the configured tolerance.

---

## Phase 6 — The style study: knock early, aim for gin, or balance?

**Goal.** Answer the question the project was started for, with a proper
evaluation methodology rather than an anecdote.

**Do.**
1. Build the population defined in **METHODOLOGY §6** — six families, every
   member scored on the **true** game payoff regardless of what it was trained
   on. The grid there is a starting point: bracket first, refine where the
   behavioural profile actually moves. Every cell is a trained agent, so this is
   the project's largest compute commitment; the gate wants a complete payoff
   matrix over whatever population you built, not a particular population.
2. Round-robin with duplicate deals; build the empirical payoff matrix with CIs.
3. **Primary ranking: Nash averaging** (`ginrl.eval.population.nash_averaging`)
   — the maximum-entropy Nash of the symmetric zero-sum meta-game. Clone
   invariant, which matters because this population is near-clones by
   construction (five `gin_seeker(λ)`, five `heuristic(T)`, …). Report the mass
   per strategy, the support, and the zero-sum residual.
   **α-Rank is Tier 3**: available via `scripts/tournament.py --alpha-rank`,
   off by default, in no gate. Run it only if the evolutionary framing earns a
   place in the write-up.
   - Use `use_inf_alpha=True` with `inf_alpha_eps`, or sweep α with
     `sweep_pi_vs_alpha` and report the plateau. A raw `alpha=1e2` overflows
     `np.exp` — verified.
   - Run `clone_invariance_error` on the *real* payoff matrix and report it, so
     the reader can see how much the α-Rank ordering depends on how finely each
     style family was swept.
4. *(Tier 2 — the write-up figure, not gated.)* **Run `schur_spectrum` first and
   let it choose k.** It reports the
   transitive/cyclic energy split and the number of independent cyclic planes.
   Fitting mElo with k=1 when the data want k=2 flattens real structure; fitting
   k=2 when the data want k=1 fits noise.
5. **Fit mElo₂ₖ over the population** (`ginrl.eval.ratings.fit_melo`) and produce
   the style wheel: each agent plotted at its cyclic embedding `c_i`, angle
   giving its cyclic character, radius giving how strongly cyclic it is. Report
   `cycle_explained` and `cyclic_strength` beside it. This is the figure that
   most directly answers the question the project was started for — it shows the
   axes along which the styles beat one another, not just that they do.
6. **Spinning-top check.** Compute `cyclic_energy` over mid-training checkpoints
   and over near-champion checkpoints separately. Czarnecki et al. (2020) predict
   the former is larger. Reporting this either way is a result.
7. Bootstrap the whole pipeline: resample the payoff matrix from its per-pair
   variance 1 000 times, recompute the Nash averaging ranking, and report rank
   stability. (If α-Rank was run at all, bootstrap it too and report both.)
8. Cross-tabulate rank against the behavioural profile from Phase 2: gin rate,
   mean deadwood at knock, mean turns to knock.

**Gate `gate-p6`.**
- Payoff matrix complete, every cell with a CI and a recorded deal count.
- Nash averaging: the ranking is stable under bootstrap at the configured
  threshold, and the zero-sum residual is below its threshold.
- **Clone invariance measured, not assumed.** Duplicating a real strategy from
  the real payoff matrix moves no other strategy's Nash mass by more than the
  configured tolerance. Unit-test anchors: on a synthetic style meta-game, Nash
  drift is 0.0000 and α-Rank drift is 0.2225.
- If α-Rank was run at all, it ran warning-free and is reported with its
  clone-drift number attached. It is not required.
- Leave-one-out check reported for each strategy under both methods.
- **Schur spectrum sanity.** Unit-test anchors: rock-paper-scissors → 99.9%
  cyclic energy, one plane, k=1; transitive ladder → 99.9% transitive, k=0; two
  independent 3-cycles → two planes at 0.53/0.47, k=2. The k used for mElo must
  be the one the spectrum suggested, recorded in the run config.
- **mElo sanity.** On a transitive control population, `cycle_explained` is
  below the configured floor and mElo's ratings agree with the plain
  Bradley-Terry fit. Unit-test anchors: rock-paper-scissors gives 85.9% cycle
  explained with strategies 120° apart; a transitive ladder gives 0.0% and a
  cyclic strength of 0.000. A mElo that finds cycles in transitive data is
  overfitting and fails the gate.
- `docs/RESULTS.md` regenerates end-to-end from artifacts with one command.

**Prior worth testing, not assuming.** Recent published work on gin rummy (in
the RLCard rule set) found that strong play gins in well under 2% of hands and
wins by knocking early with low deadwood — and that paying three times more for
a gin failed to induce gin-chasing. Our engine has different scoring
(`gin_bonus=25`, `undercut_bonus=25`), so this is a hypothesis to test here, not
a result to repeat.

---

## Phase 7 — Rules sweep and write-up

**Goal.** Turn "which style wins" into "when does each style win".

**Do.**
1. Sweep the *game's own* parameters, retraining a `nash` agent per cell.
   **Match cells on convergence, not on step count.** A reduced fixed budget
   confounds the thing being measured: §6.2 predicts gin rate *falls* as strength
   rises, so an undertrained cell shows a high gin rate for reasons that have
   nothing to do with its `gin_bonus`, and the resulting curve is a picture of
   which cells got further. Train each cell until `points_per_hand_vs_anchor`
   plateaus within the configured tolerance, and report the budget each cell
   needed — that number is a result in itself. **METHODOLOGY §7** has the strategy: `gin_bonus` is the axis
   that matters, start at {0, 25, 100}, locate the crossover, then spend the
   remaining budget refining that bracket rather than filling an even grid.
   `undercut_bonus`, `knock_card` and `oklahoma=True` are secondary.
2. For each rule set, report the equilibrium gin rate, knock timing, and deadwood
   at knock. The output is a curve: *how large must the gin bonus be before
   equilibrium play chases gin?* That is a genuinely interesting and cheap result.
3. Regenerate `docs/RESULTS.md`, and fold conclusions back into `METHODOLOGY.md`
   and `README.md`.

**Gate `gate-p7`.**
- Every sweep cell has a trained agent, a profile, and a seed.
- `README.md` and `METHODOLOGY.md` contain no placeholder text.
- `make gate-all` passes from a clean clone.

---

## Tripwires

Checked every `eval_every` steps during training. Each fires once, dumps, halts.

| Tripwire | Condition | Usual cause |
|---|---|---|
| Entropy collapse | policy entropy < floor before the schedule allows | regularisation annealed too fast, or a masking bug |
| Entropy stuck | entropy within 1% of maximum after N updates | gradients not reaching the policy head |
| Value blow-up | value loss > k× its running median | reward not scaled; ±123 range fed raw |
| KL spike | update KL > 10× target | learning rate schedule wrong |
| Mask violation | any sampled action illegal | masking applied after softmax instead of to logits |
| Degenerate action | one action > 95% of decisions over a window | collapsed policy; check the reduced action set |
| Ladder regression | score vs fixed ladder falls for 3 consecutive evals | overfitting to self; keep the best checkpoint |
| NaN / Inf | any non-finite loss or gradient | float32 overflow, or a float64→MPS cast |
| Belief head flat | auxiliary AUC ≈ 0.5 after warmup | history features not connected |
| Throughput drop | steps/s below floor | accidental sync, or CPU/MPS thrash |
| Intransitive pool | `ratings/cyclic_fraction` above threshold | agent is cycling against past selves; Elo is no longer meaningful |
| Elo stalled | no CI-separated Elo gain over N evaluations | converged, or the ladder is saturated |
| Logging overhead | `perf/logging_overhead_frac` > 2% | flushing too often, or `.item()` in the hot loop |

---

## Schedule and risk

| Phase | Rough effort | Main risk |
|---|---|---|
| 0 | hours | none; wheel exists |
| 1 | 1–2 sessions | information leakage into features; the hygiene test is the guard |
| 2 | 1 session | under-powered evaluation; the null-calibration gate is the guard |
| 3 | 2–3 sessions | trainer bugs; Kuhn/Leduc exploitability is ground truth |
| 4 | 1 session | fast feedback by design |
| 5 | 2–4 sessions + compute | wasted long runs; tripwires are the guard |
| 6 | 1–2 sessions | α-Rank misuse; bootstrap + LOO are the guards |
| 7 | 1–2 sessions | scope creep; reduced budget per cell |

**Compute sanity.** At ~34 decisions/hand and ~36k env steps/s/core, the
environment can produce far more experience than an M4 can learn from. Budget by
gradient steps and wall-clock, not by episodes, and expect the learner — not the
engine — to be the constraint.
