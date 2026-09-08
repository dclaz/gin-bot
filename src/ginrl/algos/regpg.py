"""One trainer, three magnet modes (METHODOLOGY.md section 3).

Loss per minibatch (means over rows):

    L = -PG_clip + vf_coef * MSE(V, ret) + aux_coef * BCE(aux, opp_card)
        + reg_coef * KL(pi || magnet)

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
    opp: torch.Tensor  # [N] long, opponent-card aux labels


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
    """
    if mode not in ("gae", "mc"):
        raise ValueError(f"advantage mode must be 'gae' or 'mc', got {mode!r}")
    streams: dict[int, list[int]] = {}
    for row, step in enumerate(steps):
        streams.setdefault(step.table, []).append(row)
    adv = [0.0] * len(steps)
    ret = [0.0] * len(steps)
    for rows in streams.values():
        gae, mc = 0.0, 0.0
        for pos in range(len(rows) - 1, -1, -1):
            row = rows[pos]
            reward = steps[row].reward
            nonterminal = 0.0 if steps[row].done else 1.0
            # Same table re-deals only after done, so the next row in the
            # stream is the same episode unless the buffer cut it mid-hand
            # (then bootstrap 0: the standard fixed-batch truncation bias).
            if steps[row].done or pos + 1 == len(rows):
                next_value = 0.0
            else:
                next_value = values[rows[pos + 1]]
            mc = reward + gamma * nonterminal * mc
            if mode == "mc":
                adv[row], ret[row] = mc - values[row], mc
            else:
                delta = reward + gamma * nonterminal * next_value - values[row]
                gae = delta + gamma * lam * nonterminal * gae
                adv[row], ret[row] = gae, gae + values[row]
    return adv, ret


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
        opp=torch.tensor([s.opp_card for s in steps], dtype=torch.long, device=device),
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

    @torch.no_grad()
    def kl(
        self, logits: torch.Tensor, mask: torch.Tensor, net: MaskedActorCritic, obs: torch.Tensor
    ) -> torch.Tensor:
        """Per-row KL(pi || magnet) over legal actions."""
        logp = torch.log_softmax(logits, dim=-1)
        pi = logp.exp()
        if self.mode == MAGNET_UNIFORM:
            # KL(pi || u) = -H(pi) + log L over the L legal actions.
            n_legal = mask.sum(dim=-1, keepdim=True).clamp_min(1).to(logits.dtype)
            return ((pi * logp).sum(dim=-1, keepdim=True) + n_legal.log()).squeeze(-1)
        assert self.ref is not None
        ref_logits, _, _ = self.ref(obs, mask)
        ref_logp = torch.log_softmax(ref_logits, dim=-1)
        return (pi * (logp - ref_logp)).sum(dim=-1)

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
            aux_loss = functional.cross_entropy(aux, batch.opp[idx])
            kl = magnet.kl(logits.detach(), batch.mask[idx], net, batch.obs[idx]).mean()
            loss = -pg + cfg.vf_coef * vf + cfg.aux_coef * aux_loss + reg * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            stats = {
                "pg": -float(pg),
                "vf": float(vf),
                "aux": float(aux_loss),
                "kl": float(kl),
                "loss": float(loss),
            }
    magnet.post_update(net)
    return stats
