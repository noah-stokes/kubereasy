import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.DevOpsEnv import DevOpsEnv


DEFAULT_TIMESTEPS = 100_000
DEFAULT_MODEL_PATH = Path("models/ppo_devops")


def normalization_path(model_path):
    model_path = Path(model_path)
    if model_path.suffix == ".zip":
        model_path = model_path.with_suffix("")
    return Path(f"{model_path}.vecnormalize.pkl")


def make_ppo_env():
    return DummyVecEnv(
        [lambda: Monitor(gym.wrappers.FlattenObservation(DevOpsEnv()))]
    )


def train(
    total_timesteps=DEFAULT_TIMESTEPS,
    seed=0,
    model_path=DEFAULT_MODEL_PATH,
):
    env = VecNormalize(
        make_ppo_env(),
        norm_obs=True,
        norm_reward=False,
    )
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={
            "net_arch": {
                "pi": [64, 64],
                "vf": [64, 64],
            }
        },
        n_steps=256,
        batch_size=64,
        seed=seed,
        tensorboard_log="logs/tensorboard",
        verbose=1,
    )

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.learn(total_timesteps=total_timesteps)
        model.save(model_path)
        env.save(normalization_path(model_path))
    finally:
        env.close()

    return model


def main():
    parser = argparse.ArgumentParser(description="Train PPO on DevOpsEnv")
    parser.add_argument(
        "--timesteps", type=int, default=DEFAULT_TIMESTEPS
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_MODEL_PATH
    )
    args = parser.parse_args()

    train(args.timesteps, args.seed, args.output)
    print(f"Saved model to {args.output}.zip")


if __name__ == "__main__":
    main()
