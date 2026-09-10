"""Full-game match training (Phase 5).

The match is the episode: hand returns are intermediate rewards (scaled),
the signed match bonus is terminal, and the running score is in the state
(MatchEnv pushes it to the trackers, so score dependence is learnable).
Both seats share one GinNet; the learner's seat is recorded per match and
only learner-seat rows carry gradients (independent-PPO style).

Opponent mix per match (~plan §11 starting values): 35% current self,
30% historical snapshot, 20% heuristic specialists, 10% simple bot,
5% uniform. Snapshots accumulate under run/snapshots and are sampled
uniformly. The PPO update, GAE, magnet, and Recorder machinery are shared
verbatim with the hand-level path.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ginrl.agents.baselines import HeuristicAgent, HeuristicParams, RandomAgent
from ginrl.agents.learned import GinNetAgent
from ginrl.agents.spiel_bots import SimpleGinRummyAgent
from ginrl.algos.regpg import Magnet, make_batch, ppo_update, sample_actions
from ginrl.config import HandConfig, MatchConfig, Seeds, TrainerConfig
from ginrl.env.features import feature_dim
from ginrl.env.game import HandEnv
from ginrl.env.match import MatchEnv
from ginrl.eval.arena import Arena
from ginrl.eval.record import ensure_header
from ginrl.eval.stats import bootstrap_ci
from ginrl.nets.gin import GinNet, count_params
from ginrl.telemetry.recorder import Recorder, RecorderConfig
from ginrl.train.driver import RolloutStep
from ginrl.train.gin import GIN_STAT_TO_METRIC, gin_observation_row
from ginrl.train.loop import load_checkpoint, save_checkpoint

ANCHOR = "simple_bot"

HEURISTIC_FAMILY: tuple[tuple[str, HeuristicParams], ...] = (
    ("heur", HeuristicParams()),
    ("heur-aggro", HeuristicParams(knock_threshold=3, pile_draw_aggression=0.8)),
    ("heur-gin", HeuristicParams(knock_threshold=10, discard_danger_weight=1.5)),
    ("heur-safe", HeuristicParams(discard_danger_weight=3.0)),
)


@dataclass
class MatchTrainResult:
    run_dir: Path
    final_score: float = 0.0
    final_entropy: float = 0.0
    params: int = 0
    best_ckpt: str = ""
    best_score: float = float("-inf")


def _snapshot_files(run: Path) -> list[Path]:
    return sorted((run / "snapshots").glob("round_*.pt"))


def _load_snapshot(path: Path, feat_dim: int, deck: int, device: torch.device) -> GinNetAgent:
    net = GinNet("mlp", feat_dim=feat_dim, deck=deck, hidden=128)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    agent = GinNetAgent(net.to(device), feat_dim, device, seed=0)
    agent.name = "snapshot"
    return agent


def pick_opponent(
    rng: random.Random,
    learner: GinNetAgent,
    run: Path,
    feat_dim: int,
    deck: int,
    device: torch.device,
) -> object:
    """Sample one opponent agent for a match (plan §11 mix)."""
    roll = rng.random()
    if roll < 0.35:
        return learner
    if roll < 0.65:
        snaps = _snapshot_files(run)
        if snaps:
            return _load_snapshot(rng.choice(snaps), feat_dim, deck, device)
        return learner
    if roll < 0.85:
        name, params = rng.choice(list(HEURISTIC_FAMILY))
        return HeuristicAgent(params, name=name)
    if roll < 0.95:
        # The C++ reference bot only supports the full deck (its knock-meld
        # declarations break on reduced configs); fall back to heuristic.
        if deck == HandConfig().deck_size:
            return SimpleGinRummyAgent()
        name, params = rng.choice(list(HEURISTIC_FAMILY))
        return HeuristicAgent(params, name=name)
    return RandomAgent(seed=rng.randrange(1 << 30))


@dataclass
class _Streams:
    """Per-env match streams: seats, opponents, reward attachment cursors."""

    envs: list
    seats: list[int]
    opps: list
    teams: list
    last_row: list
    seed_counter: list[int]
    rows: list
    stats: dict
    rng: random.Random
    learner: GinNetAgent
    run: Path
    feat_dim: int
    deck: int
    device: torch.device
    reward_scale: float
    match_bonus: float
    drain_guard: list = field(default_factory=list)

    def new_match(self, i: int) -> None:
        # Begin BEFORE the reset: the deal informs fire during reset and the
        # fresh bots must already be listening or they miss their own deal.
        self.seats[i] = self.rng.randrange(2)
        self.opps[i] = pick_opponent(
            self.rng, self.learner, self.run, self.feat_dim, self.deck, self.device
        )
        opp = self.opps[i]
        self.teams[i] = (self.learner, opp) if self.seats[i] == 0 else (opp, self.learner)
        wire(self.envs[i], self.teams[i])
        self.envs[i].reset(seed=self.seed_counter[0])
        self.seed_counter[0] += 1
        self.last_row[i] = None

    def absorb(self, i: int, result) -> None:
        """Attach one step's consequences to the learner stream."""
        seat = self.seats[i]
        if result.hand_returns is not None:
            self.drain_guard[i] = 0
        if result.hand_returns is not None and self.last_row[i] is not None:
            self.stats["hands"] += 1
            self.rows[self.last_row[i]]["reward"] += result.hand_returns[seat] * self.reward_scale
        if result.new_hand_pending:
            wire(self.envs[i], self.teams[i])  # begin BEFORE the deal informs
            self.envs[i].next_hand()
        if result.done:
            self.stats["matches"] += 1
            self.stats["caps"] += 1 if result.capped else 0
            self.stats["learner_wins"] += 1 if result.winner == seat else 0
            if self.last_row[i] is not None:
                self.rows[self.last_row[i]]["reward"] += (
                    self.match_bonus if result.winner == seat else -self.match_bonus
                )
                self.rows[self.last_row[i]]["done"] = True
            self.new_match(i)


def collect_match(
    envs: list[MatchEnv],
    net: GinNet,
    n_rows: int,
    gen: torch.Generator,
    device: torch.device,
    feat_dim: int,
    reward_scale: float,
    match_bonus: float,
    run: Path,
    rng: random.Random,
    first_seed: int,
) -> tuple[list[RolloutStep], list[float], list[float], int, dict[str, float]]:
    """Match self-play: rows for the learner seat only, match is the episode.

    Hand returns (seat-relative, scaled) attach to the learner row that
    closed the hand; the signed match bonus attaches to the row that closed
    the match. Every env.step result — learner or opponent move — flows
    through _absorb, so no hand/match boundary is ever dropped. Returns
    (steps, logps, values, next_seed, stats).
    """
    deck = envs[0].config.hand.deck_size
    learner = GinNetAgent(net.to(device).eval(), feat_dim, device, seed=0)
    learner.name = "learner"
    seats = [rng.randrange(2) for _ in envs]
    opps = [pick_opponent(rng, learner, run, feat_dim, deck, device) for _ in envs]
    teams = [
        ((learner, opp) if seat == 0 else (opp, learner))
        for opp, seat in zip(opps, seats, strict=True)
    ]
    for i, env in enumerate(envs):
        wire(env, teams[i])
        env.reset(seed=first_seed + i)
    st = _Streams(
        envs=envs,
        seats=seats,
        opps=opps,
        teams=teams,
        last_row=[None] * len(envs),
        seed_counter=[first_seed + len(envs)],
        rows=[],
        stats={"hands": 0, "matches": 0, "caps": 0, "learner_wins": 0},
        rng=rng,
        learner=learner,
        run=run,
        feat_dim=feat_dim,
        deck=deck,
        device=device,
        reward_scale=reward_scale,
        match_bonus=match_bonus,
        drain_guard=[0] * len(envs),
    )
    rows, logps, values = st.rows, [], []
    net.eval()
    while len(rows) < n_rows:
        # Phase 1: gather learner decisions, one batched forward.
        feats, masks, opps_hot, order = _gather(envs, st.seats, feat_dim, deck)
        if order:
            with torch.no_grad():
                logits, value, _ = net(
                    torch.tensor(feats, dtype=torch.float32, device=device),
                    torch.tensor(masks, dtype=torch.bool, device=device),
                )
            actions, blogp = sample_actions(logits.cpu(), gen)
            for k, i in enumerate(order):
                result = envs[i].step(actions[k])
                rows.append(
                    {
                        "table": i,
                        "obs": feats[k],
                        "mask": masks[k],
                        "action": actions[k],
                        "reward": 0.0,
                        "done": False,
                        "seat": st.seats[i],
                        "opp": opps_hot[k],
                    }
                )
                st.last_row[i] = len(rows) - 1
                st.absorb(i, result)
                logps.append(float(blogp[k]))
                values.append(float(value[k].cpu()))
        # Phase 2: drain opponent turns until every env wants the learner.
        _drain_streams(st)
    net.train()
    steps = [
        RolloutStep(
            table=r["table"],
            obs=r["obs"],
            mask=r["mask"],
            action=r["action"],
            reward=r["reward"],
            done=r["done"],
            seat=r["seat"],
            opp_card=-1,
            opp_hand=tuple(r["opp"]),
        )
        for r in rows[:n_rows]
    ]
    return steps, logps[:n_rows], values[:n_rows], st.seed_counter[0], st.stats


MANUAL_PHASES = frozenset({"Knock", "Layoff"})


def manual_seats_for(agents: tuple) -> frozenset:
    return frozenset(s for s, a in enumerate(agents) if getattr(a, "manual_phases", False))


def wire(env: MatchEnv, agents: tuple) -> None:
    """Begin a hand for both agents: fresh bot state + inform stream.

    Manual-phase seats (engine bots) need a new bot per hand (landmine 3);
    the listener republishes every action, including chance, to non-acting
    agents. Re-run on every hand boundary, not just match start.
    """

    def listener(acting_seat: int, state, player: int, action: int) -> None:
        # Inform-all (OpenSpiel bot protocol): the C++ bots track their own
        # moves through inform_action; our agents ignore informs. See arena.
        for agent in agents:
            agent.inform(state, player, action)

    env.hand.listener = listener
    env.hand.manual_phases = set(manual_seats_for(agents))
    agents[0].begin_game(0)
    agents[1].begin_game(1)


def drive(env: MatchEnv, agents: tuple):
    """One acting turn with Knock/Layoff manual routing (mirrors arena)."""
    hand = env.hand
    seat = hand.state().current_player()
    agent = agents[seat]
    if str(hand.state().to_dict()["phase"]) in MANUAL_PHASES and seat in manual_seats_for(agents):
        return env.raw_step(agent.choose_raw(hand.state()))
    return env.step(agent.choose(hand, seat))


def _opp_multi_hot(hand: HandEnv, seat: int, deck: int) -> list[float]:
    vec = [0.0] * deck
    for card in hand.hidden_opp_hand(seat):
        vec[card] = 1.0
    return vec


def _drain_streams(st: _Streams) -> None:
    """Step opponent turns until every env is waiting on the learner.

    The guard resets on hand boundaries: a turn-1 knock can pass a whole
    hand (or match) with no learner turn, which is legitimate — only a
    single endless hand trips.
    """
    for i, env in enumerate(st.envs):
        while env.hand.state().current_player() != st.seats[i]:
            st.absorb(i, drive(env, st.teams[i]))
            st.drain_guard[i] += 1
            assert st.drain_guard[i] < 10000, "opponent drain stuck inside one hand"


def _gather(
    envs: list[MatchEnv], seats: list[int], feat_dim: int, deck: int
) -> tuple[list, list, list, list[int]]:
    """Learner-seat observations (with pre-move opp labels) for one sweep."""
    feats, masks, opps_hot, order = [], [], [], []
    for i, env in enumerate(envs):
        seat = env.hand.state().current_player()
        if seat == seats[i]:
            obs, mask, _ = gin_observation_row(env.hand, seat, feat_dim)
            feats.append(obs)
            masks.append(mask)
            opps_hot.append(_opp_multi_hot(env.hand, seat, deck))
            order.append(i)
    return feats, masks, opps_hot, order


def _learner_agent(
    net: GinNet, feat_dim: int, device: torch.device, name: str, seed: int
) -> GinNetAgent:
    agent = GinNetAgent(net.to(device).eval(), feat_dim, device, seed=seed)
    agent.name = name
    return agent


def ladder_members(hand_config: HandConfig, seed: int) -> list:
    """Eval ladder: random, reference bot, heuristic specialists."""
    members: list = [RandomAgent(seed=seed)]
    members.append(SimpleGinRummyAgent(config=hand_config))
    for name, params in HEURISTIC_FAMILY:
        members.append(HeuristicAgent(params, name=name))
    return members


def ladder_summaries(
    net: GinNet,
    feat_dim: int,
    device: torch.device,
    agent_name: str,
    hand_config: HandConfig,
    seed: int,
    n_deals: int,
):
    """Hand-level duplicate deals vs every ladder member."""
    arena = Arena(config=hand_config, seeds=Seeds(seed))
    learner = _learner_agent(net, feat_dim, device, agent_name, seed)
    out = {}
    for member in ladder_members(hand_config, seed):
        out[member.name] = arena.duplicate_summary(learner, member, seed, n_deals)
    return out


def match_win_rate(
    net: GinNet,
    feat_dim: int,
    device: torch.device,
    match_config: MatchConfig,
    seed: int,
    n_matches: int,
) -> tuple[float, float, float]:
    """Duplicate matches to the target vs the simple bot: (wins, n, rate)."""
    import dataclasses

    # Manual boundaries are mandatory here: per-hand begins keep engine-bot
    # state exact, and auto-advance can never interleave them.
    match_config = dataclasses.replace(match_config, auto_advance=False)
    learner = _learner_agent(net, feat_dim, device, "learner", seed)
    bot = SimpleGinRummyAgent(config=match_config.hand)
    won = []
    for m in range(n_matches):
        for swap in (False, True):
            env = MatchEnv(config=match_config, seeds=Seeds(seed))
            agents = (learner, bot) if not swap else (bot, learner)
            mine = 0 if not swap else 1
            wire(env, agents)
            r = env.reset(seed=seed * 1_000_003 + m)
            while not r.done:
                r = drive(env, agents)
                if r.new_hand_pending:
                    wire(env, agents)  # begin BEFORE the deal informs
                    r = env.next_hand()
            won.append(1.0 if r.winner == mine else 0.0)
    s = bootstrap_ci(won)
    return s.mean, s.lo, s.hi


def train_match(
    hand_config: HandConfig | None = None,
    torso: str = "mlp",
    cfg: TrainerConfig | None = None,
    master_seed: int = 0,
    device: torch.device | None = None,
    run_dir: Path | None = None,
    target_score: int = 100,
    max_hands: int = 50,
    reward_scale: float = 0.02,
    match_bonus: float = 1.0,
    eval_every: int = 20,
    eval_deals: int = 100,
    snapshot_every: int = 10,
    match_probe_every: int = 100,
    probe_matches: int = 20,
    belief_states_n: int = 500,
    agent_name: str = "p5",
    resume: Path | None = None,
) -> MatchTrainResult:
    """Match-episode self-play on the full game with a snapshot matchmaker."""
    cfg = cfg or TrainerConfig()
    dev = device or torch.device("cpu")
    run = run_dir or Path(f"runs/match-{torso}-{master_seed}")
    run.mkdir(parents=True, exist_ok=True)
    (run / "snapshots").mkdir(exist_ok=True)
    config_hash = f"{master_seed}:{cfg.total_steps}:{reward_scale}:{match_bonus}"
    record_path = run / "game_record.jsonl"
    ensure_header(record_path, config_hash)
    recorder = Recorder(RecorderConfig(run_name=run.name, run_dir=run))
    torch.manual_seed(master_seed)
    torch.set_num_threads(1)
    match_config = MatchConfig(
        hand=hand_config or HandConfig(),
        target_score=target_score,
        max_hands=max_hands,
        auto_advance=False,  # drivers begin agents before each hand's deal
    )
    seeds = Seeds(master=master_seed)
    deck = match_config.hand.deck_size
    feat_dim = feature_dim(deck)
    envs = [MatchEnv(config=match_config, seeds=seeds) for _ in range(cfg.n_envs)]
    net = GinNet(torso, feat_dim=feat_dim, deck=deck, hidden=128).to(dev)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    magnet = Magnet(net, cfg)
    gen = torch.Generator().manual_seed(master_seed + 1)
    rng = random.Random(master_seed + 999)
    steps_done, deal_round, match_seed = _resume_state(
        resume, run, net, optimizer, magnet, gen, dev, master_seed
    )
    result = MatchTrainResult(run_dir=run, params=count_params(net))
    last_stats: dict[str, float] = {}
    learner_time = eval_time = 0.0
    workers: dict[int, object] = {}
    evals_dir = run / "evals"
    scanned = (
        {int(p.stem.split("_")[1]) for p in evals_dir.glob("round_*.json")}
        if evals_dir.exists()
        else set()
    )
    eval_failed = 0
    while steps_done < cfg.total_steps:
        t0 = time.perf_counter()
        steps, logps, values, match_seed, mstats = collect_match(
            envs,
            net,
            cfg.batch_size,
            gen,
            dev,
            feat_dim,
            reward_scale,
            match_bonus,
            run,
            rng,
            match_seed,
        )
        (run / "match_state.json").write_text(json.dumps({"match_seed": match_seed}))
        batch = make_batch(steps, logps, values, cfg, dev)
        last_stats = ppo_update(net, optimizer, batch, cfg, magnet, gen, steps_done)
        steps_done += len(steps)
        deal_round += 1
        learner_time += time.perf_counter() - t0
        rows_per_sec = len(steps) / max(time.perf_counter() - t0, 1e-9)
        for name, metric in GIN_STAT_TO_METRIC.items():
            if name in last_stats:
                recorder.log_scalar(steps_done, metric, last_stats[name])
        recorder.log_scalar(steps_done, "perf/rows_per_sec", rows_per_sec)
        recorder.log_scalar(steps_done, "match/hands", float(mstats["hands"]))
        recorder.log_scalar(steps_done, "match/matches", float(mstats["matches"]))
        recorder.log_scalar(steps_done, "match/caps", float(mstats["caps"]))
        recorder.log_scalar(
            steps_done, "match/learner_win_frac", mstats["learner_wins"] / max(mstats["matches"], 1)
        )
        if deal_round % snapshot_every == 0:
            torch.save(net.state_dict(), run / "snapshots" / f"round_{deal_round}.pt")
        eval_failed += reap_workers(workers, run)
        if deal_round % eval_every == 0 or steps_done >= cfg.total_steps:
            eval_time += _eval_block(
                deal_round=deal_round,
                steps_done=steps_done,
                workers=workers,
                run=run,
                agent_name=agent_name,
                master_seed=master_seed,
                eval_deals=eval_deals,
                belief_n=belief_states_n,
                probe_matches=probe_matches,
                match_probe_every=match_probe_every,
                final=steps_done >= cfg.total_steps,
                target_score=target_score,
                scanned=scanned,
                recorder=recorder,
                result=result,
                learner_time=learner_time,
                eval_time=eval_time,
            )
        save_checkpoint(
            run / "checkpoint.pt",
            net,
            optimizer,
            magnet,
            gen,
            steps_done,
            deal_round,
            match_seed,
        )
    torch.save(net.state_dict(), run / "final.pt")
    for _ in range(180):  # drain eval workers (<=15 min) for final selection
        eval_failed += reap_workers(workers, run)
        scan_evals(run, scanned, recorder, steps_done, result)
        if not workers:
            break
        time.sleep(5)
    if workers:
        print(f"stopping with {len(workers)} eval workers unfinished", flush=True)
    scan_evals(run, scanned, recorder, steps_done, result)
    recorder.log_scalar(steps_done, "match/eval_workers_failed", float(eval_failed))
    if result.best_ckpt:
        shutil.copy(run / result.best_ckpt, run / "best.pt")
    result.final_entropy = last_stats.get("entropy", 0.0)
    recorder.close()
    return result


EVAL_SCALARS = (
    ("anchor_pph", "ratings/paired_score"),
    ("anchor_pph", "ratings/points_per_hand_vs_anchor"),
    ("elo", "ratings/current_elo"),
    ("cyclic_fraction", "ratings/cyclic_fraction"),
    ("first_player_edge_elo", "ratings/first_player_edge_elo"),
    ("gin_rate", "style/gin_rate"),
    ("knock_rate", "style/knock_rate"),
    ("undercut_rate", "style/undercut_rate"),
    ("wall_rate", "style/wall_rate"),
    ("mean_turns_to_knock", "style/mean_turns_to_knock"),
    ("mean_deadwood_at_knock", "style/mean_deadwood_at_knock"),
    ("pile_draw_rate", "style/pile_draw_rate"),
    ("danger_discard_rate", "style/danger_discard_rate"),
    ("belief_auc", "belief/auc"),
    ("match_win_mean", "match/win_rate_simplebot"),
    ("match_win_lo", "match/win_rate_lo"),
    ("match_win_hi", "match/win_rate_hi"),
)


def spawn_eval(
    workers: dict[int, object],
    run: Path,
    deal_round: int,
    agent_name: str,
    eval_seed: int,
    eval_deals: int,
    belief_n: int,
    probe_matches: int,
    target_score: int,
    max_workers: int = 1,
) -> None:
    """Fork a background ladder eval; the learner never blocks on it."""
    import subprocess

    if len(workers) >= max_workers:
        print(f"eval worker still busy, skipping round {deal_round}", flush=True)
        return
    ckpt = run / "snapshots" / f"round_{deal_round}.pt"
    if not ckpt.exists():
        # snapshots/ must hold raw state dicts only: the matchmaker's
        # _load_snapshot and the eval worker both load_state_dict directly.
        # checkpoint.pt is a training envelope — unwrap the weights.
        envelope = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
        torch.save(envelope["net"], ckpt)
    workers[deal_round] = subprocess.Popen(
        [
            sys.executable,
            str(Path("scripts/eval_snapshot.py").absolute()),
            "--run",
            str(run),
            "--round",
            str(deal_round),
            "--ckpt",
            str(ckpt),
            "--agent",
            agent_name,
            "--seed",
            str(eval_seed),
            "--deals",
            str(eval_deals),
            "--belief-n",
            str(belief_n),
            "--probe-matches",
            str(probe_matches),
            "--target",
            str(target_score),
        ],
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        stdout=open(run / f"eval_r{deal_round}.log", "w"),
        stderr=subprocess.STDOUT,
    )


def reap_workers(workers: dict[int, object], run: Path) -> int:
    """Reap finished eval workers; returns new failures (never raises)."""
    failed = 0
    for rnd in [r for r, p in workers.items() if p.poll() is not None]:
        proc = workers.pop(rnd)
        if proc.returncode != 0:
            failed += 1
            print(f"eval worker round {rnd} failed rc={proc.returncode}", flush=True)
    return failed


def last_eval_round(scanned: set[int]) -> int:
    return max(scanned) if scanned else 0


def _resume_state(
    resume: Path | None,
    run: Path,
    net: GinNet,
    optimizer: torch.optim.Optimizer,
    magnet: Magnet,
    gen: torch.Generator,
    dev: torch.device,
    master_seed: int,
) -> tuple[int, int, int]:
    """Restore (steps_done, deal_round, match_seed), tolerating absence."""
    if resume is None:
        return 0, 0, master_seed * 1_000_000 + 7
    steps_done, deal_round, match_seed = load_checkpoint(resume, net, optimizer, magnet, gen, dev)
    state_path = run / "match_state.json"
    if state_path.exists():
        match_seed = int(json.loads(state_path.read_text())["match_seed"])
    return steps_done, deal_round, match_seed


def _eval_block(
    *,
    deal_round: int,
    steps_done: int,
    workers: dict[int, object],
    run: Path,
    agent_name: str,
    master_seed: int,
    eval_deals: int,
    belief_n: int,
    probe_matches: int,
    match_probe_every: int,
    final: bool,
    target_score: int,
    scanned: set[int],
    recorder: Recorder,
    result: MatchTrainResult,
    learner_time: float,
    eval_time: float,
) -> float:
    """Spawn one background eval, scan finished ones, log overhead."""
    te = time.perf_counter()
    # Round-scoped eval seed: deal numbers repeat across rounds, and
    # load_deals groups legs by (a, b, deal) — one seed per run would
    # stack 4+ legs per key and void every pair. deal_round persists
    # across resume, so resumed runs cannot collide either.
    eval_seed = master_seed * 1_000_003 + deal_round
    spawn_eval(
        workers,
        run,
        deal_round,
        agent_name,
        eval_seed,
        eval_deals,
        belief_n,
        probe_matches if (deal_round % match_probe_every == 0 or final) else 0,
        target_score,
    )
    scan_evals(run, scanned, recorder, steps_done, result)
    spent = time.perf_counter() - te
    recorder.log_scalar(
        steps_done,
        "perf/logging_overhead_frac",
        (eval_time + spent) / max(learner_time + eval_time + spent, 1e-9),
    )
    recorder.log_scalar(
        steps_done,
        "match/eval_lag_rounds",
        float(deal_round - last_eval_round(scanned)),
    )
    return spent


def scan_evals(
    run: Path,
    scanned: set[int],
    recorder: Recorder,
    steps_done: int,
    result: MatchTrainResult,
) -> None:
    """Log newly finished worker summaries; track best by anchor pph."""
    eval_dir = run / "evals"
    if not eval_dir.exists():
        return
    for path in sorted(eval_dir.glob("round_*.json")):
        rnd = int(path.stem.split("_")[1])
        if rnd in scanned:
            continue
        try:
            out = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue  # worker still writing; pick it up next round
        scanned.add(rnd)
        for key, metric in EVAL_SCALARS:
            if key in out:
                recorder.log_scalar(steps_done, metric, float(out[key]))
        lo, hi = out.get("elo_ci", [0.0, 0.0])
        recorder.log_scalar(steps_done, "ratings/elo_ci_width", float(hi) - float(lo))
        if out.get("anchor_pph", float("-inf")) > result.best_score:
            result.best_score = float(out["anchor_pph"])
            result.best_ckpt = f"snapshots/round_{rnd}.pt"
