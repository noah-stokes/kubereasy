import unittest

from agents.baselines import (
    BASELINES,
    cpu_autoscaler,
    no_op_autoscaler,
    traffic_autoscaler,
)
from env.DevOpsEnv import DevOpsEnv


def observation(**overrides):
    values = {
        "traffic_rps": 200,
        "cpu_utilization": 0.5,
        "active_replicas": 3,
        "pending_replica_delta": 0,
    }
    values.update(overrides)
    return values


class BaselineTests(unittest.TestCase):
    def test_cpu_autoscaler_thresholds_pending_actions_and_bounds(self):
        self.assertEqual(
            cpu_autoscaler(observation(cpu_utilization=0.75)),
            DevOpsEnv.SCALE_UP,
        )
        self.assertEqual(
            cpu_autoscaler(observation(cpu_utilization=0.35)),
            DevOpsEnv.SCALE_DOWN,
        )
        self.assertEqual(cpu_autoscaler(observation()), DevOpsEnv.NO_OP)
        self.assertEqual(
            cpu_autoscaler(
                observation(cpu_utilization=1.0, pending_replica_delta=1)
            ),
            DevOpsEnv.NO_OP,
        )
        self.assertEqual(
            cpu_autoscaler(
                observation(cpu_utilization=1.0, active_replicas=20)
            ),
            DevOpsEnv.NO_OP,
        )
        self.assertEqual(
            cpu_autoscaler(
                observation(cpu_utilization=0.0, active_replicas=1)
            ),
            DevOpsEnv.NO_OP,
        )

    def test_traffic_autoscaler_targets_projected_capacity(self):
        self.assertEqual(
            traffic_autoscaler(
                observation(
                    traffic_rps=250,
                    active_replicas=2,
                    pending_replica_delta=1,
                )
            ),
            DevOpsEnv.SCALE_UP,
        )
        self.assertEqual(
            traffic_autoscaler(
                observation(
                    traffic_rps=250,
                    active_replicas=5,
                    pending_replica_delta=-1,
                )
            ),
            DevOpsEnv.NO_OP,
        )
        self.assertEqual(
            traffic_autoscaler(
                observation(traffic_rps=250, active_replicas=5)
            ),
            DevOpsEnv.SCALE_DOWN,
        )
        self.assertEqual(
            traffic_autoscaler(
                observation(traffic_rps=0, active_replicas=1)
            ),
            DevOpsEnv.NO_OP,
        )
        self.assertEqual(
            traffic_autoscaler(
                observation(traffic_rps=1_000_000, active_replicas=19)
            ),
            DevOpsEnv.SCALE_UP,
        )
        self.assertEqual(
            traffic_autoscaler(
                observation(
                    traffic_rps=1_000_000,
                    active_replicas=19,
                    pending_replica_delta=1,
                )
            ),
            DevOpsEnv.NO_OP,
        )

    def test_no_op_and_registry(self):
        self.assertEqual(no_op_autoscaler(observation()), DevOpsEnv.NO_OP)
        self.assertEqual(
            BASELINES,
            {
                "cpu": cpu_autoscaler,
                "traffic": traffic_autoscaler,
                "no_op": no_op_autoscaler,
            },
        )


if __name__ == "__main__":
    unittest.main()
