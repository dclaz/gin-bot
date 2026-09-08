"""Networks: shared torso, three fixed heads (policy, value, opponent cards)."""

from ginrl.nets.actor_critic import LOGIT_FLOOR, MaskedActorCritic, NetConfig, ResidualBlock

__all__ = ["LOGIT_FLOOR", "MaskedActorCritic", "NetConfig", "ResidualBlock"]
