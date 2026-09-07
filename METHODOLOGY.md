# Methodology

## 1. The problem

Gin Rummy is a two-player, zero-sum, imperfect-information, stochastic card game.
Each player holds ten cards from a 52-card deck, draws from either the face-down
stock or the face-up discard pile, discards one card, and may knock once their
deadwood (unmelded card value) falls to ten or below, or declare gin at zero
deadwood. The knocker scores the difference in deadwood; a defender whose
deadwood is strictly below the knocker's undercuts and scores instead, with a
bonus. A tie scores zero for the knocker (verified: a 9-9 tie returns 0-0,
not undercut+25 — the engine's undercut condition is strict). In
OpenSpiel's implementation the gin bonus and undercut bonus are both 25 and the
knock card is 10, all configurable.

**The unit of play is the match, not the hand.** OpenSpiel's `gin_rummy` is a
single hand, but gin rummy is played as successive hands to a target score —
100 points under the North American rules that OpenSpiel's defaults implement,
and the same target used by the EAAI Gin Rummy Undergraduate Research Challenge,
the field's standing benchmark for this game. This matters to the project's own
question. Whether to knock early or hold for gin is *score-dependent*: a player
30 points from winning takes the sure knock, and a player 90 points behind has
every reason to chase a gin they would never chase level. A single-hand analysis
cannot see that axis at all, and would answer a narrower question than the one
we are asking.

So we wrap the engine: `MatchEnv` plays hands to a target score, carries the
running score into the state, and the **match** is the episode. Hand scores are
intermediate rewards. Phases 3 and 4 stay at hand level, where runs are minutes
and bugs are visible; Phase 5 onward trains and evaluates on matches. Every
metric is reported at both levels — points per hand *and* match win rate — since
the first is the low-variance training signal and the second is the thing being
optimised.

Four properties shape everything below.

**Imperfect information with a live inference channel.** The opponent's hand is
hidden, but their behaviour leaks: taking a card from the discard pile reveals
almost exactly what they are building; declining an upcard is informative
negative evidence; their discards constrain what they hold. Unlike poker, where
the private information is dealt once, in gin rummy information accumulates every
turn. An agent that does not model this is playing a different, easier game.

**High variance.** Returns range over ±123 and much of the outcome is the deal.
Naive head-to-head evaluation needs enormous sample sizes to resolve the
differences that matter, which is why the evaluation harness (§5) is built before
the agent.

**A large but structured action space.** OpenSpiel exposes 241 actions, of which
185 are meld declarations used only after a knock. Those 185 are not strategic
choices in any meaningful sense — see §4.

**Strategy is score-dependent.** The optimal knock threshold is a function of
the match score, not a constant. This is the fourth property because it is the
one most easily lost: it is invisible inside a single hand, and a project that
trains and evaluates hand-by-hand will produce a confident answer to a question
nobody asked. It is also, conveniently, where the interesting version of the
knock-early-versus-gin question lives — see §6.

## 2. Solution concept: why Nash, and what "self-play" means here

In a two-player zero-sum game, a Nash equilibrium strategy is a maximin strategy:
its expected value cannot be driven below the game value by *any* opponent. That
gives a target that does not depend on modelling a particular adversary, and a
scalar measure of failure — exploitability, the amount a best-responding opponent
can gain.

**Is self-play relevant?** Yes, and it is the central mechanism, with one
qualification.

The historical worry was that naive self-play policy gradient does not converge
in imperfect-information games: the last iterate cycles around the equilibrium
rather than settling on it, which is why a decade of research produced
fictitious-play, double-oracle and CFR-based deep algorithms (NFSP, PSRO, Deep
CFR, DREAM, ESCHER). The modern picture is different. Regularising the self-play
update — pulling the policy toward a fixed "magnet" or reference policy —
restores last-iterate convergence, and empirically the resulting generic methods
are as strong as or stronger than the specialised machinery. R-NaD with this idea
mastered Stratego (a game with a vastly larger tree than gin rummy) model-free,
via self-play, without search. Magnetic mirror descent (MMD) proved linear
convergence to quantal-response equilibria in extensive-form games and later
reached superhuman Stratego play using annealed regularisation schedules. And a
large controlled study (over 5 000 runs across NFSP, PSRO, ESCHER, R-NaD, MMD,
PPO and PPG) found that the fictitious-play, double-oracle and CFR-based
approaches failed to outperform generic regularised policy gradient.

So: self-play is the training engine, and regularisation is what makes it
converge rather than cycle. Unregularised self-play is the failure mode, not
self-play itself.

## 3. Algorithm: one regularised policy-gradient trainer, three magnets

The update we implement is mirror descent with proximal regularisation toward a
magnet policy ρ:

```
π_{t+1}(·|s) = argmax_π  ⟨π, q_t(s,·)⟩  −  (1/η)·KL(π ‖ π_t)  −  α·KL(π ‖ ρ)
```

The first KL controls the step size; the second is the magnet term that damps
cycling. Three settings of ρ give three published algorithms from one codebase:

| Mode | ρ | Equivalent to |
|---|---|---|
| `uniform` | uniform over legal actions | PPO / PPG with an entropy bonus — the strong simple baseline |
| `snapshot` | policy snapshot refreshed every *k* updates | MMD as published |
| `ema` | EMA of the learner's weights | a moving-magnet variant |

α is annealed (linear and power-law schedules are both implemented; the
power-law schedule is what the superhuman Stratego result used, and it avoids
premature entropy collapse while allowing stronger convergence late).

This is a deliberate scope decision. We are **not** porting R-NaD from JAX, and
we are **not** implementing Deep CFR/ESCHER. The evidence says the extra
machinery does not buy performance, and a JAX toolchain on Apple Silicon is a
side quest. NFSP is kept only as a sanity baseline if time allows.

**Network.** A shared torso feeding three heads.

*Why these heads.* Policy and value are the actor-critic minimum. The belief
head is the one real choice, and it is argued for below. That much is settled.

*Why this torso — and why that is a weaker claim than it looks.* The residual
MLP is not an independent decision. It falls out of the representation choice in
§4: once the features are a hand-engineered flat vector, an MLP is simply the
default consumer of a flat vector. The representation choice itself rests on one
cited ablation, in a different rule set (RLCard), reporting that learned state
embeddings failed to help where explicit structure did. That is suggestive, not
decisive, and §4 is by our own description "the decision that sets the ceiling".
A ceiling-setting decision resting on one transferred result should be measured,
not assumed.

So the torso is an **experimental dimension, not a default**. Phase 4 runs a
bake-off on the reduced game, where a run is minutes, over at least:

| | torso | what it tests |
|---|---|---|
| **A** | residual MLP over flat features | the baseline |
| **B** | set encoder over card embeddings (hand, pile) | does permutation structure help? |
| **C** | sequence encoder over the action history | can the net *learn* the inference channel it is currently handed? |
| **D** | MLP on the raw 644-dim observation | the control: is the hand-engineering earning its keep at all? |

D is the important one and the one a plan like this usually omits. The
history-necessity gate in Phase 1 tests the feature set with a *probe
classifier*; only D tests it end-to-end, on the actual objective. If D is
competitive, §4's central premise is wrong and we should know that in Phase 4,
not in the write-up.

The comparison is only worth running if it is fair: equal gradient-step budget,
at least three seeds each, duplicate deals, CIs on every difference, and
parameter count and throughput reported alongside — otherwise the winner is
whichever architecture got tuned most. `gate-p4` requires the table; it does not
require a particular winner.

The heads:
- **policy** over the reduced action set (draw upcard, draw stock, pass, knock,
  52 discards), with the legal mask applied to logits before the softmax;
- **value**, on scaled returns;
- **opponent-hand belief**, a 52-way multilabel head trained with BCE against the
  true hidden hand.

The belief head is an auxiliary loss, not a plan component. It earns its place
three ways: it regularises the torso toward representations that encode the
inference channel; it gives a cheap, honest diagnostic (a flat AUC means the
history features are not reaching the network — a tripwire); and it makes the
agent's reasoning inspectable, which matters for the style study in §6.

## 4. State representation — the decision that sets the ceiling

Two facts about OpenSpiel's gin rummy drive the design, both verified directly
against the engine rather than assumed.

**There is no information-state tensor.** `information_state_tensor()` raises
`InformationStateTensorShape unimplemented`. Only `observation_tensor()` exists,
and it is not a sufficient statistic for the information state: a single
observation does not record which cards the opponent took from the discard pile,
nor which upcards they declined. Those are precisely the game's information
channels. An agent trained on raw per-step observations is blind to the thing
that makes gin rummy interesting.

**The raw tensor is sparse and awkward.** Of 644 dimensions, roughly 390 are
identically zero throughout normal play (they encode laid melds, populated only
after a knock).

We therefore do not consume the raw tensor. OpenSpiel 2.0 exposes typed JSON
structs (`state.to_observation_struct(p).to_dict()`), and we build features from
those plus an explicit per-seat history accumulator, the `BeliefTracker`:

- own hand, best meld decomposition, deadwood, per-card discard cost and
  meld-adjacency;
- discard pile contents and order, top card, stock size, turn index;
- cards **known** to be in the opponent's hand — taken from the pile, not yet
  discarded;
- cards the opponent **declined** — passed on the first upcard, or left available;
- cards provably out of play;
- knock legality and distance to knock.

This is deliberately hand-engineered rather than learned. The published gin rummy
ablation study found that learned state embeddings *failed* to help while
explicit structure did, which matches the intuition that the useful abstractions
here (deadwood, meld distance, danger) are cheap to compute exactly and expensive
to learn approximately.

**Information hygiene is a test, not a convention.** Because `BeliefTracker`
sees the full state object while building features, a leak is easy to write and
invisible at training time — it just makes the agent mysteriously strong in
self-play and weak against anything else. The guard: resample the hidden state
with `state.resample_from_infostate()` 32 times and assert the feature vector is
bit-identical. Any variation is a leak.

**The meld phase is solved, not learned.** After a knock, both players must
declare melds and lay off. Choosing the meld group that minimises deadwood is a
deterministic optimisation, and OpenSpiel ships it:
`pyspiel.gin_rummy.GinRummyUtils` provides `min_deadwood`, `best_meld_group`,
`all_melds`, `legal_melds`, `all_layoffs`. Hard-coding this policy removes 185 of
241 actions from the learned problem. (The one strategic wrinkle — laying a
sub-maximal meld set to deny the opponent a layoff — is tested as an ablation in
Phase 6 rather than assumed away.)

## 5. Evaluation

### 5.1 Which tools are load-bearing, and which is a figure

This stack accreted answer-by-answer and was trimmed once deliberately. The
tiering below is the trim; keep it.

**Tier 1 — gated, always on.** These decide whether work proceeds.

| tool | question | where |
|---|---|---|
| `points_per_hand` vs fixed anchor | is training working? | every eval, Phase 3+ |
| `fit_ratings` (Elo + CI) | how do agents compare when they have not all played? | Phase 5 league |
| `cyclic_fraction` | is that Elo meaningful, or is the pool cycling? | logged beside every Elo |
| `nash_averaging` | which style is actually best? | Phase 6 gate |
| RL-BR / ISMCTS-BR | how exploitable is the champion? | Phase 5 gate |

**Tier 2 — the write-up figure. One run, not gated.** `schur_spectrum` chooses
k, `fit_melo` draws the style wheel. They are one unit; neither is useful
without the other. This exists because it answers the project's motivating
question pictorially — the axes along which styles beat one another — and for
no other reason. If it is not in the paper, it should not be run.

**Tier 3 — optional, off by default.** `alpha_rank`. It was specified at the
outset and remains available via `scripts/tournament.py --alpha-rank`, but it is
no longer part of the pipeline or any gate. It is not clone invariant, which is
disqualifying for a population built from parameter sweeps; at high alpha it
reads only the sign of each matchup and discards the margin, which in gin rummy
is the payoff; and the evolutionary question it answers is largely covered by
the Nash averaging support set, clone-invariantly. Run it as a one-off if the
evolutionary framing is wanted in the write-up, with its clone-drift number
attached.

**The rule that keeps this from regrowing:** adding an evaluation metric
requires naming which existing one it replaces or demoting one to a lower tier,
in the same commit message. Five ways to rank the same agents is four ways to pick
whichever answer you like best.


### 5.2 The four measures

No single one is sufficient, and each answers a question the others cannot.

**Head-to-head with duplicate deals.** Every deck is played twice with the seats
swapped and the results paired. This is common random numbers, and in a game
where the deal dominates single-hand outcomes it is the difference between
needing thousands and needing hundreds of thousands of hands. Every reported
win rate carries a bootstrap CI and the deal count behind it; the harness refuses
to report a comparison below the power threshold for the effect size claimed.

**Approximate exploitability, as a lower bound.** Head-to-head results alone are
a poor proxy for equilibrium quality — the standard illustration is that in
rock-paper-scissors the exact equilibrium ties an always-rock strategy, though
the two are worlds apart in exploitability. Exact exploitability requires a full
tree traversal and is out of reach for gin rummy at *any* configuration: our
probe had `nash_conv` killed by the OOM reaper even on the smallest legal
reduced deck. So we use **RL best response** — freeze the champion, treat it as
the environment, train a fresh agent against it, and report the value achieved.
That is a lower bound on exploitability, and it is the right shape of measurement:
"how much can a dedicated adversary extract", not "did it win". An ISMCTS-based
best response gives a second, independent bound.

Exact exploitability *is* used, on Kuhn and Leduc poker, to validate the trainer
before it ever touches gin rummy (Phase 3). Anchors verified in our probe:
uniform random policy NashConv is 0.9167 on Kuhn and 4.7472 on Leduc.

**Ratings over time, with a lie detector attached.** Head-to-head numbers answer
"is A better than B"; during a long training run we also want "is it getting
better", as a single tracked scalar. The obvious answer, Elo against past
checkpoints, has a specific failure mode here: Elo is a one-dimensional model
and assumes transitivity, while gin rummy styles need not be transitive — a
patient gin-seeker may beat a cautious knocker who beats an aggressive knocker
who beats the gin-seeker. Under self-play against a growing pool of one's own
past checkpoints, Elo can climb steadily while the agent walks in a circle.

Four choices make the rating trustworthy: anchored Bradley-Terry refit over the
whole game record rather than an incremental update; paired duplicate deals
collapsed into one observation; points per hand reported alongside Elo, because
Elo is fit to win/loss and in gin rummy the margin is the payoff; and a
cyclic-structure diagnostic, `ratings/cyclic_fraction`, logged beside every
rating. Past its threshold, Elo stops being the headline number and Nash
averaging over the pool takes over.

**`docs/OBSERVABILITY.md` § Ratings is canonical for all four** — the estimator,
why batch beats online here, the measured anchors, and what each metric means.
It is not restated in this document; the point here is only *why* a rating in
this game needs a lie detector attached at all.

This is also the bridge to §6: the intransitivity that would quietly break Elo
is precisely what makes the knock-early-versus-gin question interesting.

**Nash averaging over the population, with α-Rank as a second lens.** We use
Nash averaging as defined by Balduzzi et al., *Re-evaluating Evaluation*
(NeurIPS 2018) — the agent-vs-agent variant, not agent-vs-task. We build the
empirical payoff matrix of mean paired score and take the **maximum-entropy
Nash of that meta-game**. The
meta-game is two-player, symmetric and zero-sum, so its payoff matrix is
antisymmetric — precisely the setting Nash averaging was designed for, and the
setting where Nash is a linear program over a ~16×16 matrix rather than the
PPAD-complete problem that motivates alternatives.

Two quantities come out of it and they answer different questions. The
**rating** is `A @ p`: each strategy's payoff against the equilibrium mixture,
which is equivalent to ranking by Nash-equilibrium regret. The **mass** `p` is
how heavily the equilibrium leans on a strategy. Rank by the former, report the
latter as the support. Ranking by mass is a tempting error: on a transitive
ladder it returns `{1.0, 0, 0, 0}` and supplies no ordering below the top, while
`A @ p` returns `{0, −2, −5, −9}` and orders correctly.

Note what the rating does *not* do: every strategy in the equilibrium support
scores exactly 0, because that is the equilibrium condition. Nash averaging
therefore identifies a *set* of maximally general strategies the data cannot
separate, and ranks everything outside it by how many points per hand it loses
to the mixture. That is a feature — forcing a total order on a cyclic population
is what got Elo into trouble in the first place — but it means the headline
Phase 6 result may be "these three styles are jointly unseparable" rather than
"style X wins".

The property that decides this is **clone invariance**. Splitting a strategy
into k near-identical copies leaves every other strategy's mass unchanged and
preserves the family total. Our population is built from parameter sweeps —
five `gin_seeker(λ)`, several `knock_rusher(κ)`, five `heuristic(T)` — so it is
full of near-clones by construction. Without invariance, the winning style would
be whichever family happened to be swept most finely, which would make the
headline result an artifact of the experiment design rather than a fact about
gin rummy.

Measured on a synthetic style meta-game with an intransitive cycle
(`gin_seeker > patient_knock > fast_knock > gin_seeker`), cloning one strategy
into four seeds moved `gin_seeker`'s α-Rank mass from 0.332 to 0.110 and
inflated the clone family from 0.332 to 0.454, while every Nash mass was
unchanged to four decimal places. The same example shows a second issue: the
cycle has unequal weights (3, 5, 4) and Nash returns 0.417 / 0.333 / 0.250,
while α-Rank at high α returns 0.332 / 0.332 / 0.332 — it reads only the sign of
each matchup from the response graph and discards the margin. In gin rummy the
margin is the payoff.

α-Rank is also **not multidimensional** in mElo's sense. It returns one scalar
per strategy, the stationary mass. It is cycle-*aware* — it will not force a
false ordering on a cycle, and on rock-paper-scissors correctly returns
1/3, 1/3, 1/3 — but it is not cycle-*descriptive*: those three numbers say
nothing about the structure of the cycle, whereas mElo places the three
strategies 120° apart on a plane. Its richer intermediate objects (the Markov
transition matrix, the response graph) are the data re-expressed at n×n, not a
learned low-dimensional embedding, and the α sweep is a one-parameter family
rather than a second dimension.

**α-Rank is available but off by default** (Tier 3). The evolutionary question
it answers — which strategies persist under selection — is largely answered by
the Nash averaging support set, which does it clone-invariantly. Run it as a
one-off if the evolutionary framing earns its place in the write-up, with its
clone-drift number attached; do not wire it into the pipeline.

Handled explicitly rather than in a footnote:
- **Numerics.** `alpharank.compute` overflows `np.exp` at `alpha=1e2` on real
  payoff scales. We use the infinite-α model with an ε noise term, or sweep α
  with `sweep_pi_vs_alpha` and report the plateau.
- **Clone sensitivity.** Quantified rather than warned about:
  `clone_invariance_error` duplicates a real strategy from the real payoff
  matrix and reports the drift of both methods, so the reader sees how much the
  α-Rank ordering depends on the sweep design.
- **Payoff noise.** The matrix is estimated. We bootstrap from per-pair variance,
  recompute both rankings 1 000 times, and report stability, not a point
  ordering.
- **Zero-sum residual.** `A[i,j] + A[j,i]` should be zero and will not be
  exactly; the residual is logged as a free check that the harness really
  mirrors seats.

## 6. The style study: knock early, aim for gin, or balance?

The population is built by controlled intervention, and every member is scored
on the **true** game payoff regardless of what it was trained on:

- `nash` — the unshaped champion.
- `gin_seeker(λ)` — trained with an additional gin bonus λ ∈ {0, 12.5, 25, 50, 75}.
- `knock_rusher(κ)` — trained with a per-turn penalty.
- `undercut_averse(μ)` — trained with an extra undercut penalty.
- `heuristic(T)` — rule-based, knock threshold T ∈ {0 (gin only), 2, 5, 8, 10}.
- `nash@temp(τ)` — the champion at sampling temperatures {0.5, 1.0, 2.0}.

The λ, κ, μ and T values above are a *starting grid, not a requirement*. Every
cell is a trained agent, so this is the largest compute commitment in the
project. Run the endpoints and the midpoint of each family first, look at where
the behavioural profile actually moves, and refine only there. A four-point sweep
that brackets the interesting region beats an evenly-spaced five-point sweep that
does not. What the gate requires is a complete payoff matrix over whatever
population was built — not a particular population.

Reward shaping changes the game, so shaped agents are *population members*, never
the main training objective. Their behaviour is the point; their training reward
is the instrument.

Alongside the ranking, each agent gets a behavioural profile: gin rate, knock
rate, undercut rate, mean turns to knock, mean deadwood at knock, deadwood
trajectory, pile-draw rate, and rate of discarding a card the opponent is known
to want. The deliverable is the cross-tabulation of Nash-averaging rank
(`A @ p`, §5.2) against behaviour — not just *which* style wins but *what
winning looks like*.

**The prior, stated as a hypothesis.** A recent controlled study of gin rummy
agents (in RLCard's rule set) reported that a strong fixed expert gins in roughly
0.7–1.7% of hands and wins by knocking early with low deadwood, and — more
strikingly — that reward shaping could not induce gin-chasing: paying three times
more for a gin left the gin rate under one percent. If that carries over, "aim
for gin" is not a viable style at equilibrium, and the interesting axis is *how*
early to knock rather than whether to chase gin. But OpenSpiel's scoring differs
(gin bonus 25, undercut bonus 25, configurable knock card), so we test it here
rather than inheriting it.

### 6.1 The score-dependent version of the question

"Knock early, hold for gin, or balance" has no single answer, because the answer
is a function of the match score. That is the version worth reporting: not a
number but a **policy surface** — mean deadwood at knock, and gin rate, against
score differential.

The hypotheses are cheap to state and each is falsifiable:

- *Trailing badly, chase gin.* A player 90 points behind gains little from a
  6-point knock and much from a 25-point gin bonus, so the gin rate should rise
  as the deficit grows.
- *Near the target, take the sure thing.* A player 20 points from winning should
  knock at a higher deadwood than they would level.
- *Undercut risk is score-dependent too.* Being undercut near the target is much
  worse than being undercut at 0-0, so knock thresholds should tighten as the
  opponent approaches the target.

If the trained policy shows all three, the agent has learned match strategy and
the surface is the paper's central figure. If it shows none, either the score
features are not reaching the network — check them the way the belief head is
checked — or the effect is genuinely small, which is itself worth reporting
against a strong prior that it is not.

This is also why the population in §6 is scored on matches: a `gin_seeker(λ)`
that looks weak per hand may be the right policy from 90 points behind, and a
hand-level evaluation would never see it.

### 6.2 Watching the style emerge

Because the behavioural profile is logged continuously rather than measured once
at the end, the style question can be watched forming. `style/gin_rate`,
`style/mean_turns_to_knock` and `style/mean_deadwood_at_knock` plotted against
training step — and against `ratings/current_elo` — show what the agent gives up
as it gets stronger. If gin-chasing is a beginner's mistake in this rule set,
the gin rate should fall as Elo rises, and the crossover point is itself a
result. Full metric list in `docs/OBSERVABILITY.md`.

## 7. Rules sweep

Phase 7 asks the natural follow-up: the balance between knocking and ginning is a
property of the *scoring rules*, not of gin rummy in the abstract. Retraining a
`nash` agent per rule set produces a curve: how large must the gin bonus be
before equilibrium play chases gin? That converts an anecdote about style into a
statement about the game.

**The deliverable is the curve, not the grid.** `gin_bonus` is the axis that
matters; `undercut_bonus`, `knock_card` and the Oklahoma variant are secondary.
Start at the ends and the middle — `gin_bonus` ∈ {0, 25, 100} — find the bracket
containing the crossover, and spend the remaining budget refining *there*. An
evenly-spaced grid spends most of its runs confirming the flat parts of a curve.
Report the cells you ran and why; a curve with three well-chosen points and a
located crossover is a better result than eleven evenly-spaced ones without.

## 8. Hardware notes

Apple M4, unified memory, PyTorch MPS backend. Three consequences:

- MPS does not support float64. All torch paths are float32 or bfloat16, with a
  cast at the NumPy boundary.
- MPS is not automatically faster. For models of this size, per-kernel dispatch
  overhead can make CPU quicker at small batches. The device is chosen by a
  microbenchmark of the real torso at the real batch size, recorded in
  `docs/ENV_FACTS.md`, not by assumption.
- Environment stepping is not the bottleneck. Measured at roughly 36 000
  steps/s/core with observation and mask extraction, against ~34 decisions per
  hand, a single core can generate far more experience than the learner can
  consume. Optimisation effort belongs in inference batching.

## References

- Pérolat et al. (2022), *Mastering the Game of Stratego with Model-Free
  Multiagent Reinforcement Learning* (R-NaD / DeepNash). arXiv:2206.15378
- Sokota et al. (2023), *A Unified Approach to Reinforcement Learning, Quantal
  Response Equilibria, and Two-Player Zero-Sum Games* (MMD). ICLR 2023
- Sokota et al. (2025), superhuman Stratego via MMD with annealed regularisation
- Rudolph et al. (2025), *Reevaluating Policy Gradient Methods for
  Imperfect-Information Games*. arXiv:2502.08938
- Lisý & Bowling (2017), local best response; Timbers et al. (2022),
  *Approximate Exploitability: Learning a Best Response* (ISMCTS-BR),
  arXiv:2004.09677
- Omidshafiei et al. (2019), *α-Rank: Multi-Agent Evaluation by Evolution*
- Lanctot et al. (2019), *OpenSpiel: A Framework for Reinforcement Learning in
  Games*. arXiv:1908.09453
- *A Gold-Standard Study of What Makes a Lightweight Game-Playing Agent Strong*
  (2026), arXiv:2607.06854 — the gin rate and reward-shaping findings in §6
