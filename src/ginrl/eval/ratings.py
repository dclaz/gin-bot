"""Anchored Bradley-Terry ratings refit from the game record.

Elo *is* Bradley-Terry on the 400 scale; `fit_ratings` is batch maximum
likelihood on that likelihood, not a different system. Frozen checkpoints have
fixed strength, so refitting over every game ever played beats online updates
(order-invariance, Hessian standard errors, late evidence revises early
ratings).

Two stages, matching docs/OBSERVABILITY.md:
1. Ratings come from *paired* duplicate-deal outcomes: each deal contributes
   one observation (sign of the summed scores, ties split half-half), which
   cancels the deal's luck. An L2 prior keeps an undefeated agent finite.
2. The first-player edge is fit jointly with leg-scale ratings on decisive
   legs (ties carry no seat information and are dropped). Under correct seat
   mirroring it sits within its CI of zero, which makes it a free
   correctness check on the harness. The joint fit matters: paired ratings
   saturate on undefeated records, and at fixed saturated ratings a single
   leg-level upset has unbounded leverage on the edge.

`cyclic_fraction` is the Helmholtz-Hodge diagnostic: least-squares fit of a
transitive gradient (r_i - r_j) to the pairwise log-odds advantage matrix,
reported as residual energy over total energy in [0, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ELO_SCALE = 400.0 / math.log(10)  # one logit unit in Elo
L2_PRIOR_SIGMA = 2.0  # logit units; gates.yaml records this choice
SMOOTHING_HALF_COUNT = 0.5  # win-count smoothing for the advantage matrix


@dataclass(frozen=True)
class LegRecord:
    """One leg: who sat first and whether the first seat won.

    `first_won` (not a winner name) so self-play legs, where both names are
    the same agent, still carry the seat outcome the edge fit needs.
    """

    a: str
    b: str
    a_first: bool
    first_won: bool | None  # None on an exact-score tie


@dataclass(frozen=True)
class DealPair:
    """One duplicate deal: the paired score plus its two legs."""

    a: str
    b: str
    score_a: float  # leg1 + leg2 returns for A
    legs: tuple[LegRecord, LegRecord]


@dataclass(frozen=True)
class RatingResult:
    names: list[str]
    rating: dict[str, float]  # logits, anchor pinned at 0
    se: dict[str, float]  # standard errors from the observed information
    edge_logit: float
    edge_se: float
    cyclic_fraction: float
    n_deals: int
    n_legs: int
    n_tied_deals: int
    prior_sigma: float = L2_PRIOR_SIGMA

    def elo(self, name: str) -> float:
        return self.rating[name] * ELO_SCALE

    def elo_ci(self, name: str, z: float = 1.96) -> tuple[float, float]:
        mid = self.elo(name)
        half = z * self.se[name] * ELO_SCALE
        return mid - half, mid + half

    @property
    def edge_elo(self) -> float:
        return self.edge_logit * ELO_SCALE

    def edge_elo_ci(self, z: float = 1.96) -> tuple[float, float]:
        half = z * self.edge_se * ELO_SCALE
        return self.edge_elo - half, self.edge_elo + half

    def win_prob(self, a: str, b: str) -> float:
        """Neutral-seat probability that a beats b."""
        return _sigmoid(self.rating[a] - self.rating[b])


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def fit_ratings(deals: list[DealPair], anchor: str) -> RatingResult:
    """Batch Bradley-Terry fit over paired deals. Anchor rating is pinned at 0."""
    names = sorted({n for d in deals for n in (d.a, d.b)})
    if anchor not in names:
        raise ValueError(f"anchor {anchor!r} played no games")
    idx = {n: i for i, n in enumerate(names)}
    free = [n for n in names if n != anchor]

    # Stage 1: paired outcomes. y = P(a wins the deal); ties split half-half.
    # Sorted into canonical order so the fit is bit-identical across input
    # orderings (online Elo can move 290 points on a reshuffle; this cannot).
    obs, n_tied = _paired_observations(deals, idx)
    if free:
        r, ses = _solve_paired_bt(names, free, idx, anchor, obs)
        pos = {n: k for k, n in enumerate(free)}
        rating = {n: float(r[idx[n]]) for n in names}
        se = {n: (0.0 if n == anchor else float(ses[pos[n]])) for n in names}
    else:
        # Self-play: no pairs to rate; the anchor rating is trivially zero.
        rating = {names[0]: 0.0}
        se = {names[0]: 0.0}

    # Stage 2: first-player edge, fit jointly with leg-scale ratings. The
    # paired ratings above saturate on undefeated records (150-0 deals), and
    # a leg-level upset is then infinitely surprising at fixed ratings —
    # two such legs moved the old fixed-rating edge by 200 Elo. Leg-scale
    # ratings stay finite, so the edge stays honest.
    legs = _leg_observations(deals, idx)
    edge, edge_se = _fit_edge_joint(names, free, idx, anchor, legs)

    cyclic = _cyclic_fraction(names, idx, deals)
    return RatingResult(
        names=names,
        rating=rating,
        se=se,
        edge_logit=edge,
        edge_se=edge_se,
        cyclic_fraction=cyclic,
        n_deals=len(deals),
        n_legs=sum(len(d.legs) for d in deals),
        n_tied_deals=n_tied,
    )


def _logistic_bt_hessian(
    n_free: int,
    pos: dict[int, int],
    anchor_i: int,
    obs: list[tuple[int, int, float]],
    margins: list[float],
    sigma: float,
    edge: bool = False,
) -> np.ndarray:
    """Closed-form Hessian of an L2-penalized logistic pairwise likelihood.

    Observation (i, j) with margin m contributes w=s(m)(1-s(m)) to the (i,i)
    and (j,j) blocks and -w to the cross terms, plus the prior diagonal.
    With edge=True there is one extra parameter added to every margin; each
    observation then also loads w onto the edge diagonal and the edge row.
    Exactness matters: on large records the numeric Hessian deflects the
    final step just enough to land at gnorm ~= 1e-4, where the absolute
    Armijo decrease (~1e-15) is below float resolution of the ~1e4-magnitude
    objective at every alpha, so the line search fails and the fit raises
    over 0.07 Elo. The exact Hessian converges quadratically straight past
    that band.
    """
    n = n_free + (1 if edge else 0)
    h = np.eye(n) / (sigma**2)
    for (i, j, _y), m in zip(obs, margins, strict=True):
        s = _sigmoid(m)
        w = s * (1.0 - s)
        ii = pos.get(i)
        jj = pos.get(j)
        if ii is not None:
            h[ii, ii] += w
        if jj is not None:
            h[jj, jj] += w
        if ii is not None and jj is not None:
            h[ii, jj] -= w
            h[jj, ii] -= w
        if edge:
            h[n_free, n_free] += w
            if ii is not None:
                h[n_free, ii] += w
                h[ii, n_free] += w
            if jj is not None:
                h[n_free, jj] -= w
                h[jj, n_free] -= w
    return h


def _solve_paired_bt(
    names: list[str],
    free: list[str],
    idx: dict[str, int],
    anchor: str,
    obs: list[tuple[int, int, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Penalized MLE for paired outcomes; anchor pinned at 0. Returns (r, se)."""
    cols = [idx[n] for n in free]
    pos = {c: k for k, c in enumerate(cols)}
    anchor_i = idx[anchor]

    def ratings_of(theta: np.ndarray) -> np.ndarray:
        r = np.zeros(len(names))
        r[cols] = theta
        return r

    def nll(theta: np.ndarray) -> float:
        r = ratings_of(theta)
        total = float(np.sum(theta * theta) / (2 * L2_PRIOR_SIGMA**2))
        for i, j, y in obs:
            m = r[i] - r[j]
            # -[y log s(m) + (1-y) log s(-m)], branch-free via softplus
            total += (
                (1.0 - y) * m + math.log1p(math.exp(-m))
                if m >= 0
                else -y * m + math.log1p(math.exp(m))
            )
        return total

    def grad(theta: np.ndarray) -> np.ndarray:
        r = ratings_of(theta)
        g = theta / (L2_PRIOR_SIGMA**2)
        for i, j, y in obs:
            residual = _sigmoid(r[i] - r[j]) - y
            if i != anchor_i:
                g[pos[i]] += residual
            if j != anchor_i:
                g[pos[j]] -= residual
        return g

    def hess(theta: np.ndarray) -> np.ndarray:
        r = ratings_of(theta)
        return _logistic_bt_hessian(
            len(free), pos, anchor_i, obs, [r[i] - r[j] for i, j, _ in obs], L2_PRIOR_SIGMA
        )

    theta = _newton(nll, grad, np.zeros(len(free)), "rating fit", hess=hess)
    try:
        cov = np.linalg.inv(_numeric_hessian(grad, theta))
        ses = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        ses = np.full(len(free), float("nan"))
    return ratings_of(theta), ses


def _paired_observations(
    deals: list[DealPair], idx: dict[str, int]
) -> tuple[list[tuple[int, int, float]], int]:
    """One (a, b, P(a wins)) row per deal, canonically sorted. Ties split."""
    obs = []
    n_tied = 0
    for d in deals:
        y = 1.0 if d.score_a > 0 else (0.0 if d.score_a < 0 else 0.5)
        n_tied += y == 0.5
        obs.append((idx[d.a], idx[d.b], y))
    obs.sort()
    return obs, n_tied


def _leg_observations(deals: list[DealPair], idx: dict[str, int]) -> list[tuple[int, int, float]]:
    """One (first, second, P(first wins)) row per decisive leg, sorted.

    Tied legs are dropped: with no winner they carry no seat information
    (the sign-test logic), and under saturated ratings a y=0.5 row has
    unbounded leverage on the edge.
    """
    legs = []
    for d in deals:
        for leg in d.legs:
            if leg.first_won is None:
                continue
            first, second = (leg.a, leg.b) if leg.a_first else (leg.b, leg.a)
            legs.append((idx[first], idx[second], 1.0 if leg.first_won else 0.0))
    legs.sort()
    return legs


def _numeric_hessian(grad: object, theta: np.ndarray) -> np.ndarray:
    """Central-difference Hessian of a scalar objective from its gradient."""
    eps = 1e-5
    k = len(theta)
    hess = np.zeros((k, k))
    for c in range(k):
        step = np.zeros(k)
        step[c] = eps
        hess[:, c] = (grad(theta + step) - grad(theta - step)) / (2 * eps)  # type: ignore[operator]
    return (hess + hess.T) / 2


def _newton(
    nll: object,
    grad: object,
    theta0: np.ndarray,
    what: str,
    hess: object = None,
) -> np.ndarray:
    """Damped Newton with backtracking for convex penalized likelihoods.

    Deterministic and boring on purpose: quasi-Newton line searches proved
    flaky on benign inputs, and a gate dependency must not roll dice.
    Pass the closed-form Hessian when one exists (see _paired_bt_hessian for
    why the numeric fallback can strand the search at the stall floor).
    """
    theta = np.array(theta0, dtype=float)
    val = float(nll(theta))  # type: ignore[operator]
    # Gradient 1e-7 is ~1e-9 logits past the optimum: far below any reporting
    # precision, and reachable given numeric-Hessian noise. The stall limit
    # below is 1e-4 (≈0.07 Elo worst case through the L2 floor): returning
    # there is harmless, raising there breaks gates over nothing.
    for _ in range(100):
        g = np.asarray(grad(theta), dtype=float)  # type: ignore[operator]
        gnorm = float(np.max(np.abs(g)))
        if gnorm < 1e-7:
            return theta
        hess_mat = hess(theta) if hess is not None else _numeric_hessian(grad, theta)  # type: ignore[operator]
        try:
            step = np.linalg.solve(hess_mat + 1e-9 * np.eye(len(theta)), g)
        except np.linalg.LinAlgError:
            step = g
        cand: np.ndarray | None = None
        alpha = 1.0
        while alpha >= 1e-4:
            trial = theta - alpha * step
            if float(nll(trial)) <= val - 1e-4 * alpha * float(g @ step):  # type: ignore[operator]
                cand = trial
                break
            alpha *= 0.5
        if cand is None:
            if gnorm < 1e-4:
                return theta
            raise RuntimeError(f"{what} stalled above tolerance")
        new_val = float(nll(cand))  # type: ignore[operator]
        if new_val >= val:
            # Accepted by Armijo yet no float progress: rounding ate the
            # decrease, so this is the numeric bottom.
            if gnorm < 1e-4:
                return theta
            raise RuntimeError(f"{what} stalled above tolerance")
        theta, val = cand, new_val
    raise RuntimeError(f"{what} did not converge")


def _fit_edge_joint(
    names: list[str],
    free: list[str],
    idx: dict[str, int],
    anchor: str,
    legs: list[tuple[int, int, float]],
) -> tuple[float, float]:
    """Joint penalized MLE of leg-scale ratings and the first-player edge.

    P(first wins) = sigmoid(r_first - r_second + e), anchor pinned at 0.
    Only the edge is reported; the leg-scale ratings are internal (the
    headline ratings come from paired deals in stage 1). Returns (edge, se).
    """
    cols = [idx[n] for n in free]
    pos = {c: k for k, c in enumerate(cols)}
    anchor_i = idx[anchor]

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, float]:
        r = np.zeros(len(names))
        r[cols] = theta[:-1]
        return r, float(theta[-1])

    def nll(theta: np.ndarray) -> float:
        r, e = unpack(theta)
        total = float(np.sum(theta * theta) / (2 * L2_PRIOR_SIGMA**2))
        for i, j, y in legs:
            m = r[i] - r[j] + e
            total += (
                (1.0 - y) * m + math.log1p(math.exp(-m))
                if m >= 0
                else -y * m + math.log1p(math.exp(m))
            )
        return total

    def grad(theta: np.ndarray) -> np.ndarray:
        r, e = unpack(theta)
        g = theta / (L2_PRIOR_SIGMA**2)
        for i, j, y in legs:
            residual = _sigmoid(r[i] - r[j] + e) - y
            if i != anchor_i:
                g[pos[i]] += residual
            if j != anchor_i:
                g[pos[j]] -= residual
            g[-1] += residual
        return g

    def hess(theta: np.ndarray) -> np.ndarray:
        r, e = unpack(theta)
        return _logistic_bt_hessian(
            len(free),
            pos,
            anchor_i,
            legs,
            [r[i] - r[j] + e for i, j, _ in legs],
            L2_PRIOR_SIGMA,
            edge=True,
        )

    theta = _newton(nll, grad, np.zeros(len(free) + 1), "edge fit", hess=hess)
    try:
        cov = np.linalg.inv(_numeric_hessian(grad, theta))
        se = float(np.sqrt(max(cov[-1, -1], 0.0)))
    except np.linalg.LinAlgError:
        se = float("nan")
    return float(theta[-1]), se


def _cyclic_fraction(names: list[str], idx: dict[str, int], deals: list[DealPair]) -> float:
    """Helmholtz-Hodge residual energy over total energy of log-odds matrix."""
    design, target = _gradient_design(names, idx, deals)
    if design is None or target is None:
        return 0.0
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ coef
    total = float(target @ target)
    if total == 0.0:
        return 0.0
    return float(((target - fitted) @ (target - fitted)) / total)


def _gradient_design(
    names: list[str], idx: dict[str, int], deals: list[DealPair]
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Least-squares design for the transitive gradient part of log-odds."""
    k = len(names)
    wins = np.zeros((k, k))
    for d in deals:
        for leg in d.legs:
            i, j = idx[leg.a], idx[leg.b]
            if leg.first_won is None:
                wins[i, j] += 0.5
                wins[j, i] += 0.5
            else:
                won_a = leg.first_won == leg.a_first
                wins[i, j] += 1.0 if won_a else 0.0
                wins[j, i] += 0.0 if won_a else 1.0
    adv = np.log((wins + SMOOTHING_HALF_COUNT) / (wins.T + SMOOTHING_HALF_COUNT))
    rows = []
    targets = []
    for i in range(k):
        for j in range(i + 1, k):
            if wins[i, j] + wins[j, i] == 0:
                continue
            row = np.zeros(k - 1)
            # Pin names[0] at 0; columns index names[1:].
            if i > 0:
                row[i - 1] = 1.0
            if j > 0:
                row[j - 1] = -1.0
            rows.append(row)
            targets.append(adv[i, j])
    if not rows:
        return None, None
    return np.stack(rows), np.array(targets)


def elo_duel_prob(elo_a: float, elo_b: float) -> float:
    """Classic 400-scale expected score, for synthetic game generation."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def synthetic_deals(
    elos: dict[str, float],
    deals_per_pair: int,
    seed: int,
    edge_logit: float = 0.0,
    tie_prob: float = 0.0,
) -> list[DealPair]:
    """Seeded synthetic duplicate deals from known Elos (tests and docs).

    Legs are drawn from the Bradley-Terry model the fit assumes, so this is a
    calibration check of the estimator — not evidence the model fits gin.
    """
    rng = np.random.RandomState(seed)
    names = sorted(elos)
    deals = []
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            a, b = names[x], names[y]
            for _ in range(deals_per_pair):
                legs = []
                score_a = 0.0
                for leg in range(2):
                    a_first = leg == 0
                    first, second = (a, b) if a_first else (b, a)
                    m = (elos[first] - elos[second]) / ELO_SCALE + edge_logit
                    u = rng.rand()
                    if u < tie_prob:
                        first_won: bool | None = None
                    else:
                        first_won = rng.rand() < _sigmoid(m)
                    legs.append(LegRecord(a=a, b=b, a_first=a_first, first_won=first_won))
                    if first_won is None:
                        continue
                    won_a = first_won == a_first
                    score_a += 1.0 if won_a else -1.0
                deals.append(DealPair(a=a, b=b, score_a=score_a, legs=(legs[0], legs[1])))
    return deals
