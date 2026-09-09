"""One trainer, three magnet modes (METHODOLOGY.md section 3).

Loss per minibatch (means over rows):

    L = -PG_clip + vf_coef * MSE(V, ret) + aux_coef * AUX(aux, opp)
        + reg_coef * KL(pi || magnet)

AUX is single-label cross-entropy on small games (gate-p3 calibration) and
multilabel BCE over the joint opponent hand on gin (Phase 4 belief path).

`magnet=uniform` makes the KL term an entropy bonus (up to log L); that is
the Rudolph et al. baseline, not a separate entropy knob. `snapshot`
freezes a copy of the net every `snapshot_every` updates (MMD proper);
`ema` tracks the learner's parameters. The ablation control is
`reg_coef=0`, same code path.

Advantages are GAE or Monte-Carlo returns (`advantage`), decided by the
estimator guard on Leduc. Learning rate and reg_coef anneal linearly to
zero when `anneal="linear"`.

Determinism: every stochastic choice (action sampling, minibatch order)
draws from a caller-owned `torch.Generator`. Same seed + same thread count
on CPU gives bit-identical parameters.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from ginrl.config import MAGNET_EMA, MAGNET_SNAPSHOT, MAGNET_UNIFORM, TrainerConfig
from ginrl.nets.actor_critic import MaskedActorCritic
from ginrl.train.driver import RolloutStep, SmallGameVecEnv


@dataclass
class Batch:
    """One rollout round as stacked tensors (CPU or learner device)."""

    obs: torch.Tensor  # [N, D] float32
    mask: torch.Tensor  # [N, A] bool
    action: torch.Tensor  # [N] long
    logp_old: torch.Tensor  # [N] float32, behaviour log-probs
    value_old: torch.Tensor  # [N] float32, behaviour values
    adv: torch.Tensor  # [N] float32, standardised advantages
    ret: torch.Tensor  # [N] float32, returns (value targets)
    # Aux belief target: [N] long single-label (legacy small-game path) or
    # [N, D] float32 multi-hot joint hand (gin path; see RolloutStep).
    opp: torch.Tensor


def linear_factor(steps_done: int, total_steps: int) -> float:
    """Annealing multiplier: 1 -> 0 over the run, floored at 0."""
    if total_steps <= 0:
        return 1.0
    return max(0.0, 1.0 - steps_done / total_steps)


def advantages(
    steps: list[RolloutStep],
    values: list[float],
    gamma: float,
    lam: float,
    mode: str,
) -> tuple[list[float], list[float]]:
    """Per-episode GAE (or Monte-Carlo) advantages and returns.

    Rows arrive round-major; grouping by `table` recovers each table's
    chronological stream, and `done` splits it into episodes. Standardising
    happens in `make_batch`, not here, so these are raw.

    Zero-sum perspective rule (two-player games only): the critic predicts
    the ACTING seat's return, so a value or reward from the other seat
    estimates the negation. Everything is converted to the seat-0
    perspective, accumulated there, and converted back — bootstrapping or
    summing raw across a seat change silently flips the sign of the
    objective (this exact bug once drove self-play to NashConv 1.0 while
    single-seat best-response training worked fine).
    """
    if mode not in ("gae", "mc"):
        raise ValueError(f"advantage mode must be 'gae' or 'mc', got {mode!r}")
    streams: dict[int, list[int]] = {}
    for row, step in enumerate(steps):
        if step.seat not in (0, 1):
            raise ValueError(f"advantages need a 2-player zero-sum stream, got seat {step.seat}")
        streams.setdefault(step.table, []).append(row)
    adv = [0.0] * len(steps)
    ret = [0.0] * len(steps)
    for rows in streams.values():
        gae0, mc0 = 0.0, 0.0  # seat-0-perspective accumulators
        for pos in range(len(rows) - 1, -1, -1):
            row = rows[pos]
            sign = 1.0 if steps[row].seat == 0 else -1.0
            reward0 = sign * steps[row].reward
            value0 = sign * values[row]
            nonterminal = 0.0 if steps[row].done else 1.0
            # Same table re-deals only after done, so the next row in the
            # stream is the same episode unless the buffer cut it mid-hand
            # (then bootstrap 0: the standard fixed-batch truncation bias).
            if steps[row].done or pos + 1 == len(rows):
                next0 = 0.0
            else:
                nxt = rows[pos + 1]
                next_sign = 1.0 if steps[nxt].seat == 0 else -1.0
                next0 = next_sign * values[nxt]
            mc0 = reward0 + gamma * nonterminal * mc0
            if mode == "mc":
                adv0, ret0 = mc0 - value0, mc0
            else:
                delta0 = reward0 + gamma * nonterminal * next0 - value0
                gae0 = delta0 + gamma * lam * nonterminal * gae0
                adv0, ret0 = gae0, gae0 + value0
            adv[row], ret[row] = sign * adv0, sign * ret0
    return adv, ret


def _stack_opp_labels(steps: list[RolloutStep], device: torch.device) -> torch.Tensor:
    """Stack aux belief targets: legacy single-label or joint multi-hot."""
    if any(s.opp_hand is not None for s in steps):
        missing = [i for i, s in enumerate(steps) if s.opp_hand is None]
        if missing:
            raise ValueError(f"gin batch mixes labelled and unlabelled rows: {missing[:5]}")
        width = {len(s.opp_hand or ()) for s in steps}
        if len(width) != 1:
            raise ValueError(f"gin batch mixes opp-hand widths: {sorted(width)}")
        return torch.tensor([s.opp_hand or () for s in steps], dtype=torch.float32, device=device)
    return torch.tensor([s.opp_card for s in steps], dtype=torch.long, device=device)


def make_batch(
    steps: list[RolloutStep],
    logps: list[float],
    values: list[float],
    cfg: TrainerConfig,
    device: torch.device,
) -> Batch:
    """Stack a rollout round into tensors with standardised advantages."""
    adv_raw, rets = advantages(steps, values, cfg.gamma, cfg.gae_lambda, cfg.advantage)
    adv_t = torch.tensor(adv_raw, dtype=torch.float32)
    if len(adv_t) > 1:
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    return Batch(
        obs=torch.tensor([s.obs for s in steps], dtype=torch.float32, device=device),
        mask=torch.tensor([s.mask for s in steps], dtype=torch.bool, device=device),
        action=torch.tensor([s.action for s in steps], dtype=torch.long, device=device),
        logp_old=torch.tensor(logps, dtype=torch.float32, device=device),
        value_old=torch.tensor(values, dtype=torch.float32, device=device),
        adv=adv_t.to(device),
        ret=torch.tensor(rets, dtype=torch.float32, device=device),
        opp=_stack_opp_labels(steps, device),
    )


class Magnet:
    """Reference policy for the KL term: uniform, snapshot, or EMA."""

    def __init__(self, net: MaskedActorCritic, cfg: TrainerConfig) -> None:
        if cfg.magnet_mode not in (MAGNET_UNIFORM, MAGNET_SNAPSHOT, MAGNET_EMA):
            raise ValueError(f"unknown magnet mode {cfg.magnet_mode!r}")
        self.mode = cfg.magnet_mode
        self.decay = cfg.magnet_ema_decay
        self.every = cfg.snapshot_every
        self.ref: MaskedActorCritic | None = None
        if cfg.magnet_mode in (MAGNET_SNAPSHOT, MAGNET_EMA):
            self.ref = copy.deepcopy(net).requires_grad_(False)
        self.n_updates = 0

    def kl(
        self, logits: torch.Tensor, mask: torch.Tensor, net: MaskedActorCritic, obs: torch.Tensor
    ) -> torch.Tensor:
        """Per-row KL(pi || magnet) over legal actions.

        Differentiable in the online `logits` (the KL gradient is the
        regulariser). The snapshot/EMA reference forward is detached —
        the magnet is a target, not a co-learner.
        """
        logp = torch.log_softmax(logits, dim=-1)
        pi = logp.exp()
        if self.mode == MAGNET_UNIFORM:
            # KL(pi || u) = -H(pi) + log L over the L legal actions.
            n_legal = mask.sum(dim=-1, keepdim=True).clamp_min(1).to(logits.dtype)
            return ((pi * logp).sum(dim=-1, keepdim=True) + n_legal.log()).squeeze(-1)
        assert self.ref is not None
        with torch.no_grad():
            ref_logits, _, _ = self.ref(obs, mask)
        ref_logp = torch.log_softmax(ref_logits, dim=-1)
        return (pi * (logp - ref_logp)).sum(dim=-1)

    def snapshot_state(self) -> dict[str, object]:
        """Opaque magnet state for exact resume (update count + reference)."""
        return {
            "n_updates": self.n_updates,
            "ref": self.ref.state_dict() if self.ref is not None else None,
        }

    def restore_state(self, state: dict[str, object], net: MaskedActorCritic) -> None:
        """Restore magnet state saved by `snapshot_state`.

        Cross-mode resume keeps the constructor's reference: a uniform
        checkpoint carries no ref, so resuming it under snapshot/EMA starts
        from a fresh anchor at the resumed weights (not a null magnet);
        resuming a snapshot checkpoint under uniform drops the stale
        reference (a uniform magnet has no anchor by definition).
        """
        self.n_updates = int(state["n_updates"])  # type: ignore[arg-type]
        ref = state["ref"]
        if self.ref is not None and ref is not None:
            self.ref.load_state_dict(ref)  # type: ignore[arg-type]

    @torch.no_grad()
    def post_update(self, net: MaskedActorCritic) -> None:
        """Refresh the reference after one PPO update: snapshot or EMA step."""
        self.n_updates += 1
        if self.ref is None:
            return
        if self.mode == MAGNET_SNAPSHOT:
            if self.n_updates % self.every == 0:
                self.ref.load_state_dict(net.state_dict())
        else:  # EMA: every update drifts the magnet toward the learner
            for ref_p, net_p in zip(self.ref.parameters(), net.parameters(), strict=True):
                ref_p.mul_(self.decay).add_(net_p.detach(), alpha=1.0 - self.decay)


def sample_actions(logits: torch.Tensor, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one action per row from the masked policy (returns actions, logps)."""
    probs = torch.softmax(logits, dim=-1)
    actions = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
    logps = torch.log_softmax(logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return actions, logps


@torch.no_grad()
def collect(
    env: SmallGameVecEnv,
    net: MaskedActorCritic,
    n_rows: int,
    gen: torch.Generator,
    device: torch.device,
    base_seed: int,
    redeal_seed: int,
) -> tuple[list[RolloutStep], list[float], list[float], int]:
    """Self-play rollout: both seats share `net`.

    Each call re-deals every table from `base_seed` (table i's stream is
    base_seed + i), so batches are independent given the seeds and a
    checkpoint needs no table state — only counters and RNG states.
    Returns (steps, behaviour logps, behaviour values, next redeal seed) so
    the caller can chain collections without reusing a chance stream.
    """
    env.reset_all(base_seed=base_seed)
    steps: list[RolloutStep] = []
    logps: list[float] = []
    values: list[float] = []
    net.eval()
    while len(steps) < n_rows:
        live = [i for i in range(env.n_envs) if env.pending_returns(i) is None]
        if not live:
            break
        seen = [env.observe(i) for i in live]
        obs = torch.tensor([o for o, _ in seen], dtype=torch.float32, device=device)
        masks = torch.tensor([m for _, m in seen], dtype=torch.bool, device=device)
        logits, value, _ = net(obs, masks)
        actions, lp = sample_actions(logits.cpu(), gen)
        at = {table: k for k, table in enumerate(live)}
        rollout = env.act([int(actions[at[i]]) if i in at else None for i in range(env.n_envs)])
        for step in rollout.steps:
            k = at[step.table]
            logps.append(float(lp[k]))
            values.append(float(value[k].cpu()))
        steps.extend(rollout.steps)
        for i in range(env.n_envs):
            if env.pending_returns(i) is not None:
                redeal_seed += 1
                env.reset_table(i, redeal_seed)
    net.train()
    return steps, logps, values, redeal_seed


def ppo_update(
    net: MaskedActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    cfg: TrainerConfig,
    magnet: Magnet,
    gen: torch.Generator,
    steps_done: int,
) -> dict[str, float]:
    """Epochs x minibatches of clipped PPO + magnet KL + aux head. Returns scalars."""
    factor = linear_factor(steps_done, cfg.total_steps) if cfg.anneal == "linear" else 1.0
    lr = cfg.lr * factor
    for group in optimizer.param_groups:
        group["lr"] = lr
    reg = cfg.reg_coef * factor
    stats: dict[str, float] = {}
    n = batch.obs.shape[0]
    clip_fracs: list[float] = []
    grad_norms: list[float] = []
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, cfg.minibatch_size):
            idx = perm[start : start + cfg.minibatch_size]
            logits, value, aux = net(batch.obs[idx], batch.mask[idx])
            logp = (
                torch.log_softmax(logits, dim=-1)
                .gather(-1, batch.action[idx].unsqueeze(-1))
                .squeeze(-1)
            )
            ratio = (logp - batch.logp_old[idx]).exp()
            pg = torch.min(
                ratio * batch.adv[idx],
                ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * batch.adv[idx],
            ).mean()
            vf = functional.mse_loss(value, batch.ret[idx])
            if batch.opp.dtype == torch.long:
                aux_loss = functional.cross_entropy(aux, batch.opp[idx])
            else:
                aux_loss = functional.binary_cross_entropy_with_logits(aux, batch.opp[idx])
            kl = magnet.kl(logits, batch.mask[idx], net, batch.obs[idx]).mean()
            loss = -pg + cfg.vf_coef * vf + cfg.aux_coef * aux_loss + reg * kl
            optimizer.zero_grad()
            loss.backward()
            # Norm measurement only: max_norm=1e9 disables actual clipping.
            # Phase 3 does not clip gradients; grad_norm is a health signal.
            grad_norms.append(
                float(torch.nn.utils.clip_grad_norm_([p for p in net.parameters()], 1e9))
            )
            optimizer.step()
            clip_fracs.append(
                float(((ratio < 1.0 - cfg.clip_eps) | (ratio > 1.0 + cfg.clip_eps)).float().mean())
            )
            stats = {
                "pg": -float(pg.detach()),
                "vf": float(vf.detach()),
                "aux": float(aux_loss.detach()),
                "kl": float(kl.detach()),
                "loss": float(loss.detach()),
            }
    magnet.post_update(net)
    # Diagnostics on the updated net (one extra forward; small games only).
    net.eval()
    with torch.no_grad():
        final_logits, _, _ = net(batch.obs, batch.mask)
        stats["entropy"] = float(MaskedActorCritic.entropy(final_logits).mean())
    net.train()
    ret_var = float(batch.ret.var())
    stats["explained_variance"] = (
        1.0 - float((batch.ret - batch.value_old).var()) / ret_var if ret_var > 0 else float("nan")
    )
    stats["clip_frac"] = sum(clip_fracs) / len(clip_fracs)
    stats["grad_norm"] = sum(grad_norms) / len(grad_norms)
    stats["lr"] = lr
    stats["reg_alpha"] = reg
    return stats
