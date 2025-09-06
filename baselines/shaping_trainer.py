"""Shaping-based reward trainer for Pokemon Red Gym environments.

This module defines :class:`ShapingTrainer` which computes shaped rewards from
environment ``info`` dictionaries. Rewards for individual behaviours (shaping
units) decay once the behaviour has been repeatedly mastered and recover if the
behaviour is forgotten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Union, Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback if yaml isn't installed
    yaml = None  # type: ignore


@dataclass
class ShapingUnit:
    """Configuration and state for a single shaping target.

    Parameters
    ----------
    name:
        Key expected in the environment ``info`` dict.
    base_reward:
        Initial reward for successfully performing the behaviour.
    decay_trigger:
        Number of consecutive successes before reward decays. ``None`` disables
        decay.
    recovery_trigger:
        Number of consecutive failures before the behaviour's reward is
        temporarily increased. ``None`` disables recovery.
    decay_rate:
        Multiplier applied to ``current_reward`` once ``decay_trigger`` is met.
    recovery_multiplier:
        Multiplier applied to ``base_reward`` when recovery is triggered.
    min_reward:
        Lower bound for ``current_reward`` after decays.
    """

    name: str
    base_reward: float
    decay_trigger: Optional[int] = None
    recovery_trigger: Optional[int] = None
    decay_rate: float = 1.0
    recovery_multiplier: float = 1.0
    min_reward: float = 0.0

    # Runtime state
    current_reward: float = field(init=False)
    success_count: int = field(default=0, init=False)
    failure_count: int = field(default=0, init=False)
    _temp_reward: Optional[float] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.current_reward = self.base_reward

    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        """Reset per-episode counters while keeping learned reward values."""
        self.success_count = 0
        self.failure_count = 0
        self._temp_reward = None

    # ------------------------------------------------------------------
    def _apply_decay(self) -> None:
        if self.decay_trigger is not None and self.success_count >= self.decay_trigger:
            self.current_reward = max(self.min_reward, self.current_reward * self.decay_rate)
            self.success_count = 0

    # ------------------------------------------------------------------
    def _apply_recovery(self) -> None:
        if self.recovery_trigger is not None and self.failure_count >= self.recovery_trigger:
            self._temp_reward = self.base_reward * self.recovery_multiplier
            self.failure_count = 0

    # ------------------------------------------------------------------
    def update(self, active: bool) -> float:
        """Update success/failure counters and return reward contribution.

        Parameters
        ----------
        active:
            ``True`` if the environment ``info`` contains this unit's ``name``
            and the value is truthy.
        """

        if active:
            reward = self._temp_reward if self._temp_reward is not None else self.current_reward
            # success handling
            self.success_count += 1
            self.failure_count = 0
            self._temp_reward = None
            self._apply_decay()
            return reward
        else:
            # failure handling
            self.failure_count += 1
            self._apply_recovery()
            return 0.0


class ShapingTrainer:
    """Applies shaping logic to environment ``info`` dictionaries."""

    def __init__(self, units: Iterable[ShapingUnit]):
        self.units: Dict[str, ShapingUnit] = {u.name: u for u in units}

    # ------------------------------------------------------------------
    def reset_episode(self) -> None:
        """Reset per-episode counters for all shaping units."""
        for unit in self.units.values():
            unit.reset_episode()

    # ------------------------------------------------------------------
    def get_shaped_reward(self, info: Mapping[str, Any]) -> float:
        """Compute reward from the provided environment ``info`` dictionary."""
        total = 0.0
        for name, unit in self.units.items():
            active = bool(info.get(name))
            total += unit.update(active)
        return total

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: Union[str, Path, Mapping[str, Any]]) -> "ShapingTrainer":
        """Create a trainer from a YAML file path or a configuration dict."""

        if isinstance(config, (str, Path)):
            if yaml is None:
                raise ImportError("PyYAML is required to load configuration from a file")
            with open(config, "r", encoding="utf-8") as fh:
                cfg_dict = yaml.safe_load(fh)
        elif isinstance(config, Mapping):
            cfg_dict = config
        else:  # pragma: no cover - defensive programming
            raise TypeError("config must be a path or mapping")

        units_cfg = cfg_dict.get("shaping_units", [])
        units = []
        for unit_cfg in units_cfg:
            base_reward = unit_cfg.get("base_reward")
            if base_reward is None:
                base_reward = unit_cfg.get("reward")  # backward compatibility
            unit = ShapingUnit(
                name=unit_cfg["name"],
                base_reward=float(base_reward),
                decay_trigger=unit_cfg.get("decay_trigger"),
                recovery_trigger=unit_cfg.get("recovery_trigger"),
                decay_rate=unit_cfg.get("decay_rate", 1.0),
                recovery_multiplier=unit_cfg.get("recovery_multiplier", 1.0),
                min_reward=unit_cfg.get("min_reward", 0.0),
            )
            units.append(unit)

        return cls(units)


__all__ = ["ShapingTrainer", "ShapingUnit"]
