# Observability

What gets logged, where it goes, and which numbers are allowed to be believed.

## Stack

| Layer | Tool | Role |
|---|---|---|
| Source of truth | `runs/<run>/metrics.jsonl` | Everything gates and `docs/RESULTS.md` read. Always on, no dependencies. |
| Dashboard | **Trackio** | Local-first, no account, SQLite-backed, `trackio show`. The only one. |
| Ratings | `ginrl.eval.ratings` | Anchored Bradley-Terry over the whole game record + a transitivity diagnostic. |

**Two sinks, not two dashboards.** The JSONL and Trackio do genuinely different
jobs — one is the record decisions are made from, the other is what a human
looks at while a run is going. A second dashboard would just be a second place
for the truth to live.

**Reading a run from SQLite.** Trackio's `metrics` column is a hex-encoded JSON
blob, so raw SQL needs decoding — `trackio get` is the readable path, and
`runs/<run>/metrics.jsonl` is the one gates read regardless.

**Why Trackio.** Local-first, no account, runs stored in SQLite, and a CLI
(`trackio list`, `trackio get`, `trackio query --sql`). That last point decides
it for this repo: a coding agent can interrogate its own training run rather
than asking a human to look at a chart. It also has `trackio.Histogram` for real
distribution views, and reports Apple Silicon GPU utilisation via
`log_system()` — which is the number that tells you whether MPS is actually
being used or the learner has quietly fallen back to CPU.

**Why not TensorBoard as well.** An earlier draft kept it for histogram views.
That was based on a wrong assumption: Trackio has `trackio.Histogram`, so there
was no capability gap, and the `TrackioSink` had been degrading histograms to
three scalars to work around a limitation that does not exist. TensorBoard also
brings `grpcio` and `tensorboard-data-server` for a capability already present.
If TensorBoard event files ever need importing, `trackio.import_tf_events`
handles it — the migration path runs the right way.

**Why JSONL underneath both.** Dashboards are for humans watching a run;
artifacts are for decisions. No gate, table or claim in the write-up may be
scraped from a dashboard. If a sink dies mid-run the run continues — every
optional sink is wrapped and degrades to a warning.

```bash
make board          # trackio show --project ginrl
make elo            # refit ratings from the game record, print the ladder
```

## Run identity

Every run is `{phase}-{git_sha}-{config_hash}`, and the same triple is the first
line of the JSONL. This is the existing provenance rule from CLAUDE.md, not a
new one: an artifact without provenance is deleted, not debugged.

## What is logged

### `loss/` and `reg/` — is the optimiser healthy
`loss/policy`, `loss/value`, `loss/entropy`, `loss/belief_bce`, `loss/total`;
`reg/alpha` (magnet coefficient), `reg/kl_to_magnet`, `reg/kl_step`,
`reg/clip_frac`, `reg/magnet_age`.

### `opt/` — is the optimiser being driven correctly
`opt/lr`, `opt/grad_norm`, `opt/grad_norm_clipped_frac`, `opt/param_norm`,
`opt/update_ratio` (update norm over parameter norm). Distinct from `loss/`:
these describe the optimiser's own state, not the objective. Per-head gradient
norms (`opt/grad_norm/<head>`, raw and weighted losses per head under `loss/`)
show an auxiliary head capturing the shared encoder before the aggregate does.
`opt/is_ratio_p50/p90/p99` track the importance-correction distribution when
consuming stale trajectories; mass clipped nearly everywhere is the policy-lag
tripwire, not a clipping constant to raise.

### `policy/` — is it still a policy
`policy/entropy`, `policy/entropy_normalised` (divided by `log(n_legal)`, so it
is comparable across states with different legal-action counts),
`policy/effective_action_count` (perplexity), `policy/max_action_share`.
Entropy is also logged per phase (`policy/entropy_<draw,discard,knock,...>`):
a two-action draw head and a 52-way discard head have different maximum
entropies, and one collapsing while the aggregate looks healthy is invisible
otherwise.

### `value/` — is the critic learning
`value/explained_variance`, `value/mean`, `value/std`, `value/return_std`.
From Phase 5 the match head adds `value/match_nll` and `value/match_brier`
over win/draw/loss forecasts; audit both by score bucket, since low global
error with poor values near 100 is the failure that loses matches.

### `belief/` — is the history reaching the network
`belief/auc`, `belief/brier`, `belief/top1_precision` — how often the card the
belief head is most confident about really is in the opponent's hand. A flat AUC
means `BeliefTracker` features are not being used, which is a tripwire.
Calibration is scored only over legally possible unseen cards. The joint
decoder adds `belief/hand_nll` (complete-hand negative log likelihood),
`belief/recall_at_k`, `belief/posterior_ess` (effective sample size of the
particle weights) and `belief/decision_value` — paired score with learned
versus uniform beliefs, which is the ablation that says whether dependencies
buy decisions rather than likelihood.

### `ratings/` — is it getting better

The headline progress metric is **`ratings/points_per_hand_vs_anchor`**: mean
paired score per hand against `SimpleGinRummyBot`, with a CI. It is in game
units, needs no model, no anchor convention and no tuning parameter, and it
cannot be misread. Watch this one.

Elo is secondary and exists for the *pool*, not the training curve: once there
are twenty checkpoints, a heuristic family and an ISMCTS baseline that have not
all played each other, a single comparable scale is genuinely useful.
`ratings/current_elo` with `ratings/current_elo_ci95`, `ratings/elo/<agent>` per
ladder member, `ratings/cyclic_fraction`, `ratings/first_player_edge_elo`.

**Match-level, from Phase 5:** `ratings/match_win_rate_vs_anchor` with its CI,
over duplicate matches to 100. Points per hand stays the headline training
curve — it is far lower variance — but the match is the objective, so both are
always reported. Where they disagree, the match is right and the difference is
worth understanding: it usually means the agent is winning hands it did not need
to win. Note that ratings themselves are still fit from *paired hands*; a match
is a correlated sequence and must not be fed to the Bradley-Terry fit as if its
hands were independent observations.

### `population/` — is the pool transitive, and in how many dimensions
`population/transitive_energy`, `population/cyclic_energy`,
`population/suggested_k`, `population/cycle_plane_share/{0..3}`, and
`population/zero_sum_residual` (`A[i,j] + A[j,i]`, a free check that the harness
really mirrors seats). Written by the Phase 6 tournament and by the checkpoint
league's spinning-top check — not by the training loop. See the glossary below
for how `population/cyclic_energy` differs from `ratings/cyclic_fraction`.

### `style/` — what kind of player is emerging
`style/gin_rate`, `style/knock_rate`, `style/undercut_rate`, `style/wall_rate`,
`style/mean_turns_to_knock`, `style/mean_deadwood_at_knock`,
`style/pile_draw_rate`, `style/danger_discard_rate`.

**Bucketed by match score differential**, from Phase 5 onward:
`style/mean_deadwood_at_knock_by_score/{behind,level,ahead}` and
`style/gin_rate_by_score/{behind,level,ahead}`. The score-conditional version is
the research question; the aggregate is its average, and an average over a
score-dependent policy hides exactly the effect we are looking for. A flat curve
across buckets means the policy is ignoring the score — see `gate-p5`.

This block is not diagnostics — it is the research question, live. Watching
`style/gin_rate` and `style/mean_turns_to_knock` move as regularisation anneals
shows the equilibrium style *emerging* rather than being read off at the end.
Plot `style/gin_rate` against `ratings/current_elo` and the knock-early-versus-gin
answer draws itself.

Phase 5+ profiler extensions: `style/pickup_precision` (accepted upcards that
enter the eventual best meld group) and match-level stats beside the rates
(match length, comeback rate, win probability by starting score). Defensive
discard quality is evaluated against posterior opponent hands, so it lives
with the belief metrics, not here.

### `perf/` — is the machine being used
`perf/env_steps_per_sec`, `perf/decisions_per_sec`, `perf/updates_per_sec`,
`perf/actor_wait_frac`, `perf/learner_wait_frac`, `perf/update_ms`,
`perf/inference_latency_p50/p95/p99`, `perf/policy_lag`,
`perf/host_ram_gb`, `perf/device_mem_gb`, `perf/logging_overhead_frac`.
Search work, when it exists, reports `perf/search_fallback_rate` (share of
positions solved by the fallback policy) beside its traversal/particle
budgets; strength is always quoted at a fixed latency budget.

### `tripwire/` — how close to the edge
Every tripwire from IMPLEMENTATION_PLAN.md is logged as *two* series: its current
value and its threshold. You want to watch entropy approach the floor over an
hour, not discover it hit the floor.

### Histograms (every `histogram_every` updates)
Deadwood at knock, turns to knock, action distribution, value predictions,
advantages, per-layer gradient norms.

### Tables and figures (every evaluation)
League payoff heatmap; Elo over training with a CI band; Nash-averaging mass
and rank over the pool; belief-head reliability diagram; and one **annotated
sample game**
— hand, deadwood, chosen action, policy entropy, belief top-5 — per evaluation.
The annotated game is the cheapest debugging artifact in the project and the one
most likely to reveal that the agent is doing something stupid that no scalar
captures.

## Overhead discipline

Logging must not perturb what it measures.

- Scalars accumulate in memory and flush every `scalar_every` updates.
- Histograms emit every `histogram_every` updates.
- No `.item()` per element in the hot loop; accumulate on device, sync once per
  interval.
- Rating evaluations run outside the learner step so a 20 000-deal tournament
  does not stall training.
- `perf/logging_overhead_frac` is itself logged, and Phase 5's gate caps it at
  2% of a *real* training run. (In a synthetic loop that does no work, this
  number is meaningless and will read high — the gate applies to real runs.)

## Ratings

### Elo, fit properly — not an alternative to Elo

To be explicit, because the naming invites confusion: the logistic Elo formula
`E_A = 1/(1 + 10^((R_B - R_A)/400))` *is* the Bradley-Terry model with a scale
constant. `fit_ratings` is not a different rating system; it produces ordinary
Elo numbers on the ordinary Elo scale. The only difference is estimation. Classic
Elo is online stochastic gradient ascent on the Bradley-Terry likelihood with
step size K; this is batch maximum likelihood on the same likelihood.

That difference is not cosmetic. On 3,000 synthetic games with known true
ratings, reshuffling the game order alone moves online Elo (K=32) by up to 290
points per agent, while the batch fit is bit-identical across orderings. Varying
K from 8 to 64 on a fixed ordering moves one agent from +15 to −119. RMSE against
truth: 101 Elo online, 38 Elo batch.

The underlying reason is that K is a *tracking* parameter — Elo was designed for
humans whose strength drifts, and K sets how fast old evidence is forgotten. Our
agents are frozen checkpoints with a fixed true strength, so there is nothing to
track and the maximum-likelihood estimate is simply the right answer. Computer
chess reached the same conclusion long ago; BayesElo and Ordo are batch fits for
exactly this reason. Batch estimation also yields standard errors from the
Hessian at no extra cost, and lets late evidence revise early checkpoints'
ratings, neither of which an online update can do.

### The trap

Elo assumes transitivity: that strengths lie on a line, so beating a strong
agent implies beating a weak one. Two-player zero-sum imperfect-information
games routinely violate this, and gin rummy *styles* are a natural candidate —
a patient gin-seeker may beat a cautious knocker who beats an aggressive knocker
who beats the gin-seeker. Under self-play against a growing pool of one's own
past checkpoints, Elo can rise monotonically while the agent walks in a circle.

This is not hypothetical. In our unit tests, a pure rock-paper-scissors
population produces three Elo ratings within 2 points of each other: Elo cannot
see the cycle at all, and a naive "current versus past checkpoints" ladder would
have reported steady improvement.

### What we do instead

**Anchored Bradley-Terry, refit over the whole record.** Checkpoints are frozen,
so their true strength does not change; the right estimator is a static
maximum-likelihood fit over every game ever played, not an incremental update.
`SimpleGinRummyBot` is pinned at 0 Elo so the scale means the same thing in week
one and week six. An L2 prior keeps an undefeated agent's rating finite —
without it the MLE diverges, which is both numerically fatal and, after thirty
games, not what we believe. Standard errors come from the observed information
matrix, so every Elo is reported with a CI. An Elo without error bars, in a game
this luck-heavy, is a decoration.

**Paired duplicate deals.** Each deal is played twice with the seats swapped.
The two games are strongly correlated, so they are collapsed into one paired
observation: the scores are summed, which cancels the deal's luck, and the sign
is the outcome. In our test, a 4-point-per-hand skill edge buried in ±30 points
of deal variance shows as a 55% raw win rate and a 100% paired win rate over the
same games.

**A first-player edge term.** Under correct mirroring this should come out near
zero. If it does not, the harness is not actually mirroring — it is a free
correctness check on the evaluation code, logged every time.

**Points per hand, alongside Elo.** Elo is computed from win/loss and discards
the margin, but in gin rummy the margin *is* the payoff. An agent that wins 51%
of hands by large margins is far better than its Elo suggests. Both are always
reported.

**`ratings/cyclic_fraction`.** The Helmholtz-Hodge decomposition of the pairwise
advantage matrix into a transitive gradient part and a cyclic residual, reported
as residual energy over total energy in [0, 1]. Zero means the pool is
transitive and Elo is fully descriptive. One means pure rock-paper-scissors and
every rating is a fiction. It is a logged scalar and a tripwire: past the
configured threshold, Elo stops being the headline number and Nash averaging
over the pool takes over.
This is the bridge between the training dashboard and the Phase 6 style study —
the same intransitivity that would break Elo is exactly what makes the
knock-early-versus-gin question interesting.

### mElo, and where it fits

`cyclic_fraction` says *how much* structure Elo is missing. mElo says *what it
is*. They are complements, and Elo is literally the k = 0 case of mElo, so
nothing is being replaced.

**mElo is Tier 2: one run, not gated, not a training-time series.** It is fit
once over the Phase 6 population to draw the write-up figure, and it reports
`ratings/melo_cycle_explained` (fractional log-loss reduction from the cyclic
term — 0 means one number per agent suffices) and `ratings/melo_cyclic_strength`
(largest cyclic logit between any pair) into that run's artifacts only. During
training, `ratings/cyclic_fraction` is the per-evaluation scalar and it is
sufficient; adding the mElo series to the training loop is how the evaluation
stack regrew last time.

The figure is the k = 1 **style wheel**: each agent plotted at its `c_i`, angle
giving its cyclic character and radius giving how strongly cyclic it is. An
agent near the origin has no cyclic character — it is simply strong or weak.

### Glossary: the four "cyclic" quantities

Four metric names contain the word *cyclic* and they are not interchangeable.
This table is authoritative; CLAUDE.md's landmine 7 points here. A fifth name
needs a commit message saying why.

| Metric | Question it answers | Tier | Cadence |
|---|---|---|---|
| `ratings/cyclic_fraction` | *How much* structure is a single Elo number missing? Helmholtz-Hodge residual energy over total, in [0, 1]. | 1, gated | every evaluation |
| `population/cyclic_energy` | Same decomposition over the *population* meta-game rather than the checkpoint ladder — the input to the spinning-top check. | 2 | Phase 6, plus the checkpoint league |
| `population/suggested_k` | How many independent *dimensions* that structure has, from `schur_spectrum`. Sets mElo's k. | 2 | Phase 6, once |
| `ratings/melo_cyclic_strength` | What those dimensions *look like*: the largest cyclic logit between any pair, the style wheel's radius. | 2 | Phase 6, once |

`ratings/melo_cycle_explained` is the companion to the last row — the fractional
log-loss reduction the cyclic term buys — and answers "was fitting k > 0 worth
it at all".

Do not report `‖C‖` for any of them: the embedding is identified only up to
rotation, and a set of parallel vectors can have large norm while contributing
nothing.

### Choosing k, and the spinning-top check

Logged: `population/transitive_energy`, `population/cyclic_energy`,
`population/suggested_k`, and `population/cycle_plane_share/{0..3}`.

The spinning-top check runs over the checkpoint league: compute `cyclic_energy`
separately over mid-training checkpoints and over near-champion checkpoints.
The prediction is that the first exceeds the second. If it holds, style choice
matters at intermediate skill and washes out near equilibrium.

### The ladder

Fixed, and it never trains: random, `SimpleGinRummyBot` (the anchor), the
heuristic family across knock thresholds, ISMCTS at a fixed budget, and a
selection of past champions. Every `elo_every` steps the current policy plays a
fixed number of duplicate deals against each, results append to
`runs/game_record.jsonl`, and the whole model is refit. Past ratings are revised
as evidence accumulates, which is the point of refitting rather than updating.
