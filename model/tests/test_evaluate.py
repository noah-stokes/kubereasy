import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

from training.evaluate import compare
from training.train import make_ppo_env, normalization_path, train


class NoOpModel:
    def __init__(self):
        self.observations = []

    def predict(self, observation, deterministic):
        self.observations.append(observation.copy())
        return 0, None


class EvaluateTests(unittest.TestCase):
    @patch("training.evaluate.PPO.load", return_value=NoOpModel())
    def test_compare_uses_matching_traffic(self, load):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"
            normalizer = VecNormalize(
                make_ppo_env(), norm_obs=True, norm_reward=False
            )
            normalizer.reset()
            normalizer.save(normalization_path(model_path))
            normalizer.close()

            results = compare(model_path, episodes=1, seed=123)

        load.assert_called_once_with(model_path)
        self.assertEqual(
            set(results), {"ppo", "cpu", "traffic", "no_op"}
        )
        self.assertEqual(results["ppo"], results["no_op"])
        self.assertGreater(results["traffic"][0], results["no_op"][0])
        self.assertLessEqual(
            np.abs(load.return_value.observations[0]).max(), 10.0
        )

    def test_compare_requires_an_episode(self):
        with self.assertRaises(ValueError):
            compare("model", episodes=0)

    @patch("training.train.PPO")
    def test_train_normalizes_observations_and_saves_statistics(self, ppo):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"

            train(total_timesteps=1, model_path=model_path)

            env = ppo.call_args.args[1]
            self.assertIsInstance(env, VecNormalize)
            self.assertTrue(env.norm_obs)
            self.assertFalse(env.norm_reward)
            self.assertEqual(
                ppo.call_args.kwargs["tensorboard_log"],
                "logs/tensorboard",
            )
            self.assertTrue(normalization_path(model_path).is_file())


if __name__ == "__main__":
    unittest.main()
