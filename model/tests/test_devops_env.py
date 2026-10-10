import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from env.DevOpsEnv import DevOpsEnv


class DevOpsEnvTests(unittest.TestCase):
    def test_env_checker_observations_and_seeded_workload(self):
        env = DevOpsEnv()
        check_env(env, skip_render_check=True)

        first, _ = env.reset(seed=123)
        second, _ = env.reset(seed=123)

        self.assertTrue(env.observation_space.contains(first))
        for key in first:
            self.assertEqual(first[key], second[key])

    def test_scaling_is_delayed_and_respects_replica_bounds(self):
        env = DevOpsEnv(traffic=np.zeros(13))
        observation, _ = env.reset()
        self.assertEqual(observation["active_replicas"], 1)

        for step in range(6):
            action = env.SCALE_UP if step == 0 else env.NO_OP
            observation, reward, _, _, info = env.step(action)
            if step == 0:
                self.assertEqual(reward, -1.0)
                self.assertEqual(observation["active_replicas"], 1)
                self.assertEqual(observation["pending_replica_delta"], 1)
                self.assertEqual(
                    observation["provisioning_time_remaining"], 5
                )

        self.assertTrue(info["action_accepted"])
        self.assertEqual(observation["active_replicas"], 2)
        self.assertEqual(observation["pending_replica_delta"], 0)

        for step in range(6):
            action = env.SCALE_DOWN if step == 0 else env.NO_OP
            observation, _, _, _, _ = env.step(action)

        self.assertEqual(observation["active_replicas"], 1)
        self.assertEqual(observation["pending_replica_delta"], 0)

        _, reward, _, truncated, info = env.step(env.SCALE_DOWN)
        self.assertTrue(truncated)
        self.assertFalse(info["action_accepted"])
        self.assertEqual(reward, -11.0)

    def test_queue_reward_truncation_and_invalid_action(self):
        env = DevOpsEnv(traffic=[150, 150])
        observation, _ = env.reset()
        self.assertTrue(env.observation_space.contains(observation))

        observation, reward, terminated, truncated, _ = env.step(env.NO_OP)
        self.assertEqual(observation["queue_depth"], 50)
        self.assertEqual(reward, -31.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

        observation, reward, terminated, truncated, _ = env.step(env.NO_OP)
        self.assertEqual(observation["queue_depth"], 100)
        self.assertEqual(observation["traffic_rps"], 0)
        self.assertEqual(reward, -36.0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)

        env.reset()
        with self.assertRaises(ValueError):
            env.step(3)

    def test_latency_severity_penalty_is_capped(self):
        env = DevOpsEnv(traffic=[10_000])
        env.reset()

        _, reward, _, truncated, _ = env.step(env.NO_OP)

        self.assertEqual(reward, -36.0)
        self.assertTrue(truncated)


if __name__ == "__main__":
    unittest.main()
