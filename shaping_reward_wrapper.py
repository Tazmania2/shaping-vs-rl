"""Reward shaping wrapper for Pokémon Red Gym environments.

This module defines :class:`ShapingRewardWrapper`, a thin :mod:`gymnasium`
wrapper that augments or replaces the environment reward using a
:class:`baselines.shaping_trainer.ShapingTrainer` instance.  The trainer looks at
information provided by the environment via the ``info`` dictionary and computes
additional reward components with optional decay and recovery dynamics.

Usage
-----
>>> from baselines.shaping_trainer import ShapingTrainer
>>> from shaping_reward_wrapper import ShapingRewardWrapper
>>> trainer = ShapingTrainer.from_config("shaping_config.yaml")
>>> env = ShapingRewardWrapper(gym.make("SomeEnv"), trainer)

The wrapper calls ``trainer.reset_episode()`` on ``reset`` and combines the
returned shaped reward with the environment's reward on each ``step``.
"""
from __future__ import annotations

from typing import Any, Tuple

import gymnasium as gym

from baselines.shaping_trainer import ShapingTrainer


class ShapingRewardWrapper(gym.Wrapper):
    """Apply :class:`ShapingTrainer` rewards to an environment.

    Parameters
    ----------
    env:
        The environment to wrap.
    trainer:
        Instance of :class:`ShapingTrainer` providing shaped reward values.
    replace_reward:
        If ``True`` the environment's reward is ignored and the shaped reward is
        returned instead.  Otherwise the shaped reward is added to the original
        reward (default behaviour).
    """

    def __init__(self, env: gym.Env, trainer: ShapingTrainer, *, replace_reward: bool = False) -> None:
        super().__init__(env)
        self.trainer = trainer
        self.replace_reward = replace_reward

    # ------------------------------------------------------------------
    def reset(self, **kwargs: Any) -> Tuple[Any, dict]:
        """Reset the environment and internal trainer state."""
        self.trainer.reset_episode()
        return self.env.reset(**kwargs)

    # ------------------------------------------------------------------
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        """Augment rewards with shaped values on every environment step."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = self.trainer.get_shaped_reward(info)
        reward = shaped if self.replace_reward else reward + shaped
        return obs, reward, terminated, truncated, info


__all__ = ["ShapingRewardWrapper"]
