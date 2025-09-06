"""Visualise baseline vs reward-shaped training results.

Reads the CSV logs produced by :mod:`dual_train` and creates comparison plots
showing raw episode returns, rolling average returns and success rates.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.index.name = "episode"
    return df


def plot(df_base: pd.DataFrame, df_shape: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Total reward per episode
    plt.figure()
    plt.plot(df_base.index, df_base["reward"], label="Baseline")
    plt.plot(df_shape.index, df_shape["reward"], label="Shaping")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "episode_rewards.png")
    plt.close()

    # Rolling average reward
    roll_base = df_base["reward"].rolling(100, min_periods=1).mean()
    roll_shape = df_shape["reward"].rolling(100, min_periods=1).mean()
    plt.figure()
    plt.plot(roll_base, label="Baseline")
    plt.plot(roll_shape, label="Shaping")
    plt.xlabel("Episode")
    plt.ylabel("Rolling avg reward (100 ep)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "rolling_reward.png")
    plt.close()

    # Success rate
    succ_base = df_base["success"].rolling(100, min_periods=1).mean()
    succ_shape = df_shape["success"].rolling(100, min_periods=1).mean()
    plt.figure()
    plt.plot(succ_base, label="Baseline")
    plt.plot(succ_shape, label="Shaping")
    plt.xlabel("Episode")
    plt.ylabel("Success rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "success_rate.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"), help="Directory containing CSV logs")
    args = parser.parse_args()

    base_csv = args.results / "baseline_rewards.csv"
    shape_csv = args.results / "shaping_rewards.csv"
    df_base = load_csv(base_csv)
    df_shape = load_csv(shape_csv)
    plot(df_base, df_shape, args.results / "plots")


if __name__ == "__main__":
    main()
