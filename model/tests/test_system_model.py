import unittest

from env.system_model import Simulation


class CalculateMetricsTests(unittest.TestCase):
    def test_queue_is_carried_between_timesteps_and_drained_by_capacity(self):
        simulation = Simulation(100, 6, 6, 1, 20, 1.0, 200, 10.0)

        first = simulation.calculate_metrics(150, 1)
        second = simulation.calculate_metrics(
            150, 1, previous_queue_depth=first.queue_depth
        )
        drained = simulation.calculate_metrics(
            20, 1, previous_queue_depth=first.queue_depth
        )

        self.assertEqual(first.queue_depth, 50)
        self.assertEqual(second.queue_depth, 100)
        self.assertEqual(second.p95_latency, 1200)
        self.assertEqual(drained.queue_depth, 0)
        self.assertEqual(drained.cpu_utilization, 0.7)

    def test_zero_capacity_preserves_all_demand(self):
        simulation = Simulation(100, 6, 6, 1, 20, 1.0, 200, 10.0)

        metrics = simulation.calculate_metrics(
            25, 0, previous_queue_depth=10
        )

        self.assertEqual(metrics.queue_depth, 35)
        self.assertEqual(metrics.cpu_utilization, 1.0)
        self.assertEqual(metrics.p95_latency, 10_000)


if __name__ == "__main__":
    unittest.main()
