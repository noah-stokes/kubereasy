import argparse
from pathlib import Path
from statistics import fmean, pstdev

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from agents.baselines import BASELINES
from env.DevOpsEnv import DevOpsEnv
from training.train import make_ppo_env, normalization_path


DEFAULT_MODEL_PATH = Path("models/ppo_devops")
DEFAULT_EPISODES = 10
DEFAULT_SEED = 10_000


def episode_rewards(policy, episodes, seed, flatten_observation=False):
    env = DevOpsEnv()
    if flatten_observation:
        env = gym.wrappers.FlattenObservation(env)

    rewards = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            total_reward = 0.0

            while True:
                action = policy(observation)
                observation, reward, terminated, truncated, _ = env.step(
                    action
                )
                total_reward += reward
                if terminated or truncated:
                    break

            rewards.append(total_reward)
    finally:
        env.close()

    return rewards


def compare(
    model_path=DEFAULT_MODEL_PATH,
    episodes=DEFAULT_EPISODES,
    seed=DEFAULT_SEED,
):
    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    model = PPO.load(model_path)
    normalizer = VecNormalize.load(
        normalization_path(model_path), make_ppo_env()
    )
    normalizer.training = False

    def ppo_policy(observation):
        observation = normalizer.normalize_obs(observation)
        action, _ = model.predict(observation, deterministic=True)
        return action

    try:
        results = {
            "ppo": episode_rewards(
                ppo_policy,
                episodes,
                seed,
                flatten_observation=True,
            )
        }
        for name, policy in BASELINES.items():
            results[name] = episode_rewards(policy, episodes, seed)
        return results
    finally:
        normalizer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare PPO with deterministic autoscalers"
    )
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH
    )
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    results = compare(args.model, args.episodes, args.seed)
    print(f"{'policy':<10} {'mean reward':>14} {'std dev':>12}")
    for name, rewards in sorted(
        results.items(), key=lambda item: fmean(item[1]), reverse=True
    ):
        print(f"{name:<10} {fmean(rewards):>14.2f} {pstdev(rewards):>12.2f}")


if __name__ == "__main__":
    main()
