"""Dual-agent training script comparing baseline and reward shaped PPO agents.

The script trains two agents on the Pokémon Red Gym environment.  Agent A uses
only the sparse environment reward while Agent B receives additional shaped
reward via :class:`ShapingTrainer`.  Episode statistics (return, length and
success flag) are written to CSV files in the specified results directory.

Example
-------
$ python dual_train.py --total_timesteps 10000 --seed 42 \
    --shaping_config shaping.yaml --output results
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from baselines.shaping_trainer import ShapingTrainer
from shaping_reward_wrapper import ShapingRewardWrapper

# ---------------------------------------------------------------------------
class EpisodeLogger(BaseCallback):
    """Collect per-episode statistics and write them to a CSV file."""

    def __init__(self, csv_path: Path):
        super().__init__()
        self.csv_path = csv_path
        self.rows = []

    def _on_step(self) -> bool:  # type: ignore[override]
        info = self.locals.get("infos", [{}])[0]
        if "episode" in info:
            ep = info["episode"]
            row = {
                "reward": ep.get("r", 0.0),
                "length": ep.get("l", 0),
                "success": int(info.get("badge_obtained", False)),
            }
            self.rows.append(row)
        return True

    def _on_training_end(self) -> None:  # type: ignore[override]
        if self.rows:
            df = pd.DataFrame(self.rows)
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.csv_path, index=False)


# ---------------------------------------------------------------------------
def make_env_fn(env_id: str, *, seed: int, trainer: Optional[ShapingTrainer] = None) -> Callable[[], gym.Env]:
    """Create an environment factory respecting the given seed and wrapper."""

    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        if trainer is not None:
            env = ShapingRewardWrapper(env, trainer)
        return Monitor(env)

    return _init


# ---------------------------------------------------------------------------
def train_single(env_fn: Callable[[], gym.Env], total_timesteps: int, seed: int, csv_file: Path) -> None:
    """Train a PPO agent and log per-episode statistics."""

    env = make_vec_env(env_fn, n_envs=1, seed=seed)
    model = PPO("CnnPolicy", env, seed=seed, verbose=0)
    callback = EpisodeLogger(csv_file)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    env.close()


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--total_timesteps", type=int, default=1000, help="Training steps per agent")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Directory to store results")
    parser.add_argument("--shaping_config", type=Path, help="YAML config for reward shaping")
    parser.add_argument(
        "--reward_type",
        choices=["baseline", "shaping", "both"],
        default="both",
        help="Which agents to train",
    )
    parser.add_argument("--env_id", default="PokemonRed-v0", help="Gym environment id")
    args = parser.parse_args()

    np.random.seed(args.seed)

    baseline_csv = args.output / "baseline_rewards.csv"
    shaping_csv = args.output / "shaping_rewards.csv"

    if args.reward_type in {"baseline", "both"}:
        env_fn = make_env_fn(args.env_id, seed=args.seed)
        train_single(env_fn, args.total_timesteps, args.seed, baseline_csv)

    if args.reward_type in {"shaping", "both"}:
        trainer = ShapingTrainer.from_config(args.shaping_config)
        env_fn = make_env_fn(args.env_id, seed=args.seed, trainer=trainer)
        train_single(env_fn, args.total_timesteps, args.seed, shaping_csv)


if __name__ == "__main__":
    main()
