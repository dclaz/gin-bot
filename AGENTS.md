# CLAUDE.md — gin-bot

Gin Rummy RL research repo. OpenSpiel engine, PyTorch, Apple Silicon, uv.

## Prime directive

Work phase by phase against `IMPLEMENTATION_PLAN.md`. A phase is done when
`make gate-pN` exits 0 — not when the code "looks right". Never edit a gate
to make it pass; changing a gate threshold requires the commit that changes it to
say, in its message, what the old value was, what the new one is, and why.

## What is fixed, and what is a default

These documents state three kinds of claim. Read the voice, because it decides
how much room you have.

- **Verified facts** — "Verified engine facts" and "Landmines" below. Probed
  against the live engine. Not revisable; re-derive them with `make probe` if you
  doubt them, do not reason around them.
- **Load-bearing decisions** — the acceptance gates, the evaluation tiering in
  METHODOLOGY §5.1, information hygiene as a bit-exact test, duplicate deals,
  refit-not-incremental ratings, JSONL as the source of truth. Each was reasoned
  to at cost and each has a test behind it. Changing one needs the reason in the
  commit message that changes it.
- **Defaults** — everything else: module layout, network torso, feature set,
  sweep grids, which magnet modes exist. These are a starting point, not a
  specification. **You may change a default if the phase gate still passes and
  you say so in the commit message.** A better idea that clears the
  same gate is a contribution, not a deviation.

The gates are the contract. Where a default and a gate disagree, the gate wins.

## Gate integrity

There is no mechanical guard on the gates, so the record is the git history.
Never edit a gate to make it pass. A commit that changes a threshold must say so
in its message, with the old value, the new value and the reason.

`make status` prints the SHA-256 of `configs/gates.yaml`. Compare it against the
hash in the commit that last changed the file — `git log -p -- configs/gates.yaml`
— not against your working tree. A tampered threshold that has been committed
becomes HEAD, after which a working-tree comparison correctly reports no drift
and tells you nothing. That has happened here before.

## Stop and ask the human when

- A gate fails three times with three different fixes.
- A fix requires changing a number in `configs/gates.yaml`.
- You are about to start a training run longer than 30 minutes — and before you
  ask, re-benchmark CPU against MPS at that run's real model and batch size, and
  bring both numbers, plus a ≥1M-decision throughput measurement at the run's
  real config to ground the duration estimate. Picking the wrong device costs
  the whole run; estimating from utilisation costs the schedule.
- `make facts-check` reports drift you did not cause.

## Environment (do not rediscover this)

- `uv` only. Never `pip install`, never `python -m venv`, never `conda`.
  Add deps with `uv add`, run things with `uv run`. `uv.lock` is committed.
- Python 3.13, macOS arm64. `open-spiel==2.0.2` ships a `cp313 macosx_11_0_arm64`
  wheel — installation is a wheel download, not a CMake build. If you find
  yourself compiling OpenSpiel, you have made a mistake; stop.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in `.env` and by the Makefile.
- Device is chosen by `ginrl.utils.device.pick_device()`, which reads
  `RECOMMENDED_LEARNER_DEVICE` from the generated `src/ginrl/spiel_facts.py`.
  (`docs/ENV_FACTS.md` is the human-readable twin; code reads the Python file.)
  Do not hardcode `"mps"` or `"cuda"` anywhere.
- **Machine migration checklist** (moving to new hardware re-runs Phase 0's
  logic, not its numbers): `make probe` first (regenerates facts *and* the
  device recommendation — never copy them across machines), then
  `make facts-check`, then `make gate-p0` as smoke. Sustained allocation
  stays under ~48 GB on a 64 GB machine; the replay window, not the parameter
  count, is what unified memory buys.


## Commands

```
make setup        # uv sync + probe env + write generated facts
make probe        # re-run env probe, regenerate docs/ENV_FACTS.md + spiel_facts.py
make facts-check  # fail if generated facts drift from the live engine
make test         # pytest -q (unit + property tests, < 60s)
make lint         # ruff check + ruff format --check
make gate-pN      # phase N acceptance gate (N = 0..7)
make gate-all     # every gate in order; the release check
make status       # git sha, phase, last gate results, ratings, run provenance
make board        # Trackio dashboard (local, no account)
make elo          # refit ratings from the game record, print the ladder
```

To inspect a live or finished run without a human looking at a chart:
`trackio list`, `trackio get` and `trackio query --sql` (all `--project ginrl`),
or read `runs/<run>/metrics.jsonl` directly — the JSONL is the source of truth
and the only thing gates may read.

## Verified engine facts

Verified by running OpenSpiel 2.0.2. Re-verify on the M4 via `make probe`;
`src/ginrl/spiel_facts.py` is generated, never hand-edited.

- `gin_rummy`: 2 players, zero-sum, sequential, imperfect info, explicit chance.
- `num_distinct_actions() == 241`; `observation_tensor_size() == 644`;
  `max_game_length() == 300`; utility range ±123.
- Action layout: `0..51` discard a card (index = `suit*13 + rank`, `As=0 … Ks=12,
  Ac=13 …`), `52` draw upcard, `53` draw stock, `54` pass, `55` knock,
  `56..240` the 185 meld declarations. Constants live in `pyspiel.gin_rummy`
  (`DRAW_UPCARD_ACTION`, `KNOCK_ACTION`, `MELD_ACTION_BASE`, …) — import them,
  do not retype the integers.
- Game params and defaults: `gin_bonus=25`, `undercut_bonus=25`, `knock_card=10`,
  `hand_size=10`, `num_ranks=13`, `num_suits=4`, `oklahoma=False`.
- **There is no information state tensor.** `information_state_tensor()` raises.
  Only `observation_tensor()` exists, and it is *not* a sufficient statistic for
  the information state. History features are our responsibility — see landmine 1.
- `pyspiel.gin_rummy.GinRummyUtils(num_ranks, num_suits, hand_size)` provides
  `min_deadwood`, `best_meld_group`, `all_melds`, `legal_melds`, `legal_discards`,
  `all_layoffs`, `meld_to_int`, `int_to_meld`, `card_value`. **Use these.** Do not
  write a meld solver.
- `state.to_dict()` / `state.to_observation_struct(p).to_dict()` give typed JSON
  (OpenSpiel 2.0 structs). This is the feature source of record.
- `state.resample_from_infostate(...)` works — used for the leakage test and for
  determinized search.
- `pyspiel.make_simple_gin_rummy_bot(game.get_parameters(), player_id)` is the
  reference opponent. `pyspiel.ISMCTSBot` also runs on this game.

## Landmines

1. **Observation is not Markov.** A single `observation_tensor` / observation
   struct does not record which cards the opponent *took from the discard pile*,
   or which upcards they declined. An agent trained on raw observations is
   blind to the main information channel in gin rummy. Every policy consumes
   features from `ginrl.env.features.BeliefTracker`, which accumulates history
   per player. Any new feature must pass `tests/test_information_hygiene.py`.
2. **~390 of the 644 observation dims are always zero in normal play** (they
   encode laid melds, only populated after a knock). Do not feed the raw tensor
   and hope. We build features from the observation struct.
3. **`SimpleGinRummyBot` is stateful and `restart_at()` raises `NotImplemented`.**
   Construct a *fresh pair of bots per game*, and call `inform_action(state,
   player, action)` on the non-acting bot for every action including chance.
   Reusing a bot across games throws a `CHECK_TRUE` failure from C++. Use
   `ginrl.agents.simple_spiel_bot.SimpleBotRunner`, which does this correctly.
4. **Reduced-size gin rummy is real but padded.** `num_ranks`/`num_suits`/
   `hand_size` genuinely shrink the deck and hands, but the action space stays
   241 and the observation stays 644. Constraint:
   `num_ranks * num_suits >= 2*hand_size + WALL_STOCK_SIZE + 1` (`WALL_STOCK_SIZE == 2`).
5. **Exact exploitability on gin rummy is out of reach at every size.**
   `exploitability.nash_conv` was killed by the OOM reaper even on the smallest
   legal config (4×2 deck, hand_size 2). Exact exploitability is for Kuhn and
   Leduc only. For gin rummy use RL-BR (`ginrl.eval.rlbr`), which gives a lower
   bound. Do not spend a session trying to make `nash_conv` work here.
6. **`alpharank.compute` overflows** `np.exp` at `alpha=1e2`. Use
   `use_inf_alpha=True` with `inf_alpha_eps`, or sweep alpha with
   `alpharank.sweep_pi_vs_alpha` and report the plateau. A `RuntimeWarning` in
   a tournament log is a bug, not noise.
7. **alpha-Rank is not clone invariant, and our population is near-clones.**
   Cloning one strategy into four seeds moved another strategy's alpha-Rank mass
   by 0.22 in our test, while Nash averaging moved 0.0000. The headline
   population ranking is `ginrl.eval.population.nash_averaging` (max-entropy
   Nash of the symmetric zero-sum meta-game). alpha-Rank is reported as an
   evolutionary lens with its clone-drift number attached. Do not swap them.
   Note also that alpha-Rank at high alpha reads only the *sign* of each
   matchup; in gin rummy the margin is the payoff.
   alpha-Rank is Tier 3: off by default, not in any gate. Nash averaging is the
   population ranking. Do not confuse the latter with multidimensional Elo:
   both are Balduzzi et al. 2018, but mElo is section 3 (a predictive model that
   can express cycles) and Nash averaging is section 4 (a clone-invariant
   evaluation method). Our Bradley-Terry fit is mElo with k=0, so they are one
   model family, not two. Report `ratings/melo_cyclic_strength`, never `||C||`
   — the embedding is identified only up to rotation. Four different quantities
   have "cyclic" in the name and they answer different questions; the glossary
   in `docs/OBSERVABILITY.md` is authoritative. Do not invent a fifth.
8. **MPS has no float64.** Anything that touches `torch` is float32 or bfloat16.
   NumPy defaults to float64, so cast at the boundary. `.numpy()` on an MPS
   tensor needs a `.cpu()` first.
9. **MPS is often slower than CPU for our model sizes.** The learner runs on
   whichever device was benchmarked faster *at the size actually being trained*.
   Per-kernel dispatch overhead means the ordering can invert between a small
   model and a large one, so a recommendation made at one size is not evidence
   about another: `make probe` at Phase 0 measures a stand-in, Phase 4 measures a
   reduced-deck model, and neither settles the full-game network. Re-benchmark
   before any expensive run. Actors always run on CPU. One process touches MPS;
   never fork after initialising an MPS context.
10. **Elo lies when the pool is intransitive.** Self-play against past
   checkpoints can show a smoothly rising Elo while the agent cycles. Never
   report Elo without `ratings/cyclic_fraction` beside it; in a pure
   rock-paper-scissors pool our own unit test produces three Elos within two
   points of each other. Ratings come from `ginrl.eval.ratings.fit_ratings`
   (anchored Bradley-Terry, refit over the whole record) — never from an
   incremental Elo update loop.
11. **Telemetry must not perturb the run.** Accumulate scalars and flush on an
   interval; no `.item()` per element in the hot loop; rating tournaments run
   outside the learner step. `perf/logging_overhead_frac` is logged and gated.
12. Env stepping is ~36k steps/s/core with observation + mask (measured on x86;
   re-measure on M4). That number is *raw* stepping: it excludes
   `to_observation_struct().to_dict()` per seat per decision and `BeliefTracker`
   accumulation, which is the pipeline actually running. So the fair claim is
   "raw stepping is not the bottleneck, inference batching probably is" — the
   feature path is unmeasured until Phase 1 measures it. Profile before
   optimising either.
13. **The engine calls no-progress pile-cycling a draw.** A hand where both
   players keep taking the upcard without drawing stock ends 0-0 with the
   stock untouched (observed: dead after both players consecutively take and
   re-discard the same upcard). It is not a wall (stock exhaustion) but our
   accounting files it under no-knock endings all the same. Any agent that
   over-takes from the pile will "wall" every game without ever touching the
   stock — check the action log, not the stock, when walls spike.
14. **Simultaneous shared-net self-play falls into long-lived bad cycles.**
   On Kuhn a seed landed in an inverted pattern (bluff J always, slowplay K
   always) and sat at NashConv ~0.5 for ~8M steps before escaping; regimens
   that avoid the trap floor above it (uniform magnet: Kuhn ~0.03-0.10,
   Leduc ~0.7-1.0 across seeds). A single seed's settle number means nothing
   and last-iterate minima do not hold — pin recipe+seed from a multi-seed
   comparison, and distrust any ladder number from a lucky init (a seed whose
   init entropy/NC starts near equilibrium never had to navigate there).
14. **A failed take test must route to stock, not to `legal[0]`.** At a Draw
   phase the legal actions sort 52 (pile) before 53 (stock), so a fallthrough
   `return legal[0]` takes the upcard unconditionally. Worse, a draw test that
   simulates a different discard rule than the one the agent will actually use
   takes cards the discard step immediately spits back (the pair loops into
   landmine 13). Simulate the real discard choice in the take test.
15. **Hands from one match are not independent observations.** Bootstrap
   resampling for match play must resample whole matches (or paired-match
   bundles), then seeds — never individual hands, and never legs across
   training seeds. Our hand-level arena correctly bootstraps deals; the match
   evaluator must bootstrap one level up. Feeding match hands to the
   Bradley-Terry fit as independent rows is the same error (OBSERVABILITY
   says so too).
16. **Determinized vote is not search.** Calling `resample_from_infostate`,
   solving each sample as perfect information and voting is strategy fusion:
   each particle gets its own best action and the vote is consistent with no
   information set. `ISMCTSAgent` is a determinized-search *baseline* for the
   ladder, not the solver design — the Phase 6 solver keeps one average
   strategy over the particle range keyed by legal information sets.
13. **`discard_pile` excludes the takeable card.** The struct's `discard_pile`
   is the buried pile only; the takeable card lives in the `upcard` field
   (`None` at Discard time, when `pile[-1]` is the just-discarded card).
   Taking "the upcard" delivers the `upcard` field's card and leaves the pile
   untouched. Recording takes from `pile[-1]` attributes buried cards to the
   opponent's hand and inverts the belief signal. Undercut is strict, too: a
   tied deadwood scores 0-0 for the knocker, not undercut+25 (METHODOLOGY §1
   said "ties or beats" — wrong; fixed in Phase 1).
17. **Team tuples move agent objects; seats are bound by begin_game.** A seat
   flip reorders the tuple, so tuple position is never the learner — fish it
   from a side list by identity (`_seat_team`/`learners`, with an assert).
   Fishing by position silently built double-opponent teams here: the manual
   set then covered both seats and learner-side Layoff stopped
   auto-resolving. Same family: on any hand boundary, rebind seats (flip)
   *before* reset+wire, so each stateful bot's `begin_game` sees its new seat.
17. **Decision value needs a trained policy.** At 300k steps the learned-vs-
   ablated paired score reads indifferent (-0.18 [-0.45,+0.09] over 3000
   deals); at 1M steps it reads +0.12 with CI excluding zero. Measuring the
   belief ablation on an undertrained net "proves" beliefs are useless.
   Gate-p4 measures decision value on a 1M winner continuation, never on
   bake-off cells.
18. **Exact policy values are out of reach even on tiny gin.** Full-tree
   enumeration under uniform-random visits 3.9M unique nodes in 80s without
   finishing on the degenerate 5x1/hand-1 config (extends landmine 5, which
   covers nash_conv). The match "exact solver" check is bit-exact agreement
   against an independent raw-engine implementation (tests/test_match_exact.py),
   never enumeration.

## Observability

Full spec in `docs/OBSERVABILITY.md`. The parts that constrain code:

- All logging goes through `ginrl.telemetry.recorder.Recorder`. Never call
  `trackio.log` directly — the fan-out, sampling, provenance and failure
  isolation live in the Recorder.
- **Two sinks, not two dashboards.** JSONL is the record; Trackio is the human
  view. Do not add a second dashboard. Use `trackio.Histogram` for
  distributions — it exists, and an earlier draft wrongly added TensorBoard on
  the assumption that it did not.
- `runs/<run>/metrics.jsonl` is the source of truth. Gates, `docs/RESULTS.md`
  and any claim in the write-up read the JSONL. Nothing is ever scraped from a
  dashboard.
- A failing dashboard sink must never take down a training run. Optional sinks
  are wrapped and degrade to a warning.
- Metric names are namespaced: `loss/`, `reg/`, `policy/`, `value/`, `opt/`,
  `belief/`, `ratings/`, `population/`, `style/`, `perf/`, `tripwire/`. That
  list is closed — `docs/OBSERVABILITY.md` defines each one, and a new namespace
  needs a commit message saying why. Every tripwire logs both its current value and its
  threshold, so you can watch it approach.
- `style/*` — gin rate, knock rate, deadwood at knock, turns to knock — is not
  decoration. It is the research question, live. Log it from Phase 4 onward.

## Code rules

- `src/ginrl/`, imports absolute, `from __future__ import annotations`.
- Type hints on every public function. `ruff` clean. No `# type: ignore` without
  a reason on the same line.
- No magic numbers: engine constants come from `spiel_facts` or `pyspiel.gin_rummy`;
  experiment constants come from a config dataclass in `configs/`.
- Every RNG is seeded from a single `Seeds` object. No bare `random.random()`,
  no unseeded `np.random`.
- Every artifact written to `runs/` carries the git SHA, the config hash, and the
  `spiel_facts` hash. An artifact without provenance is deleted, not debugged.
- Tests are fast (`make test` under 60s). Anything slower is a gate, not a test.

## Definition of done for any change

1. `make lint test` passes.
2. `make facts-check` passes.
3. The gate for the current phase passes.
4. The commit message states **what was verified, not what was written** — and
   records any non-obvious choice, any pick between two reasonable designs, and
   any new landmine. There is no separate decision log; `git log` is it, which is
   why the message carries the reasoning rather than a summary of the diff.
5. A new landmine also goes into the Landmines list below, where the next
   session will actually read it. The commit says why; this file says what.

## Session protocol

Start: `make status`, then `git log --oneline -10` and the current phase section
of `IMPLEMENTATION_PLAN.md`. Do not read the whole plan every session.

End: `make lint test`, then commit with a message that records what you verified
and any decision a future session would otherwise re-litigate.

## Do not

- Do not reimplement meld/deadwood logic (landmine: `GinRummyUtils` exists).
- Do not port R-NaD from JAX. See METHODOLOGY.md §3 for why regularised policy
  gradient is the chosen family; a JAX toolchain on this machine is a side quest.
- Do not add a dependency without a commit message that states what capability
  is missing and why nothing already installed provides it. Especially not `jax`,
  `tensorflow`, `ray`, `hydra`, or a second experiment tracker.
- Do not "improve" the reward function in the main training path. Reward
  shaping changes the game; shaped agents are population members evaluated on
  the true payoff (see METHODOLOGY.md §6).
- Do not report a head-to-head win rate without a confidence interval and the
  number of duplicate deals behind it.
- Do not report Elo without its CI and the current `ratings/cyclic_fraction`.
- **Do not add an evaluation metric without removing or demoting one.** Name the
  trade in the same commit message. This stack already accreted once and was trimmed;
  five ways to rank the same agents is four ways to pick the answer you like.
  The tiering in METHODOLOGY.md section 5 is the contract: Tier 1 is gated,
  Tier 2 is one figure for the write-up, Tier 3 is off by default.
- Do not add an incremental Elo update loop. Ratings are refit from
  `runs/game_record.jsonl`, which makes them reproducible from disk.
