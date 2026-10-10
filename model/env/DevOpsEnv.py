import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.system_model import Metrics, Simulation, config as system_config
from env.workload import (
    WorkloadConfig,
    WorkloadGenerator,
    config as workload_config,
)


class DevOpsEnv(gym.Env):
    metadata = {"render_modes": []}

    NO_OP = 0
    SCALE_UP = 1
    SCALE_DOWN = 2

    def __init__(self, traffic=None):
        super().__init__()

        simulation_config = dict(system_config)
        simulation_config.setdefault(
            "scale_down_delay",
            simulation_config["provisioning_delay"],
        )
        self.simulation = Simulation(**simulation_config)
        self.workload_config = WorkloadConfig(**workload_config)
        self._provided_traffic = self._validate_traffic(traffic)

        self.action_space = spaces.Discrete(3)
        max_delay = max(
            self.simulation.provisioning_delay,
            self.simulation.scale_down_delay,
        )
        replica_range = (
            self.simulation.max_replicas - self.simulation.min_replicas
        )
        max_float = np.finfo(np.float32).max
        self.observation_space = spaces.Dict(
            {
                "traffic_rps": spaces.Box(
                    0.0, max_float, shape=(), dtype=np.float32
                ),
                "p95_latency_ms": spaces.Box(
                    0.0, max_float, shape=(), dtype=np.float32
                ),
                "cpu_utilization": spaces.Box(
                    0.0, 1.0, shape=(), dtype=np.float32
                ),
                "queue_depth": spaces.Box(
                    0.0, max_float, shape=(), dtype=np.float32
                ),
                "active_replicas": spaces.Box(
                    self.simulation.min_replicas,
                    self.simulation.max_replicas,
                    shape=(),
                    dtype=np.int32,
                ),
                "pending_replica_delta": spaces.Box(
                    -replica_range,
                    replica_range,
                    shape=(),
                    dtype=np.int32,
                ),
                "provisioning_time_remaining": spaces.Box(
                    0, max_delay, shape=(), dtype=np.int32
                ),
            }
        )

        self._traffic = None
        self._step_index = 0
        self._active_replicas = self.simulation.min_replicas
        self._pending_actions = []
        self._queue_depth = 0.0
        self._metrics = None

    @staticmethod
    def _validate_traffic(traffic):
        if traffic is None:
            return None

        values = np.asarray(traffic, dtype=float)
        if (
            values.ndim != 1
            or values.size == 0
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
        ):
            raise ValueError(
                "traffic must be a non-empty sequence of nonnegative numbers"
            )
        return values.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self._provided_traffic is None:
            generator = WorkloadGenerator(
                self.workload_config, seed=self.np_random
            )
            self._traffic = generator.add_noise(generator.periodic())
        else:
            self._traffic = self._provided_traffic.copy()

        self._step_index = 0
        self._active_replicas = self.simulation.min_replicas
        self._pending_actions = []
        self._queue_depth = 0.0
        self._metrics = Metrics(
            requests_per_second=self._traffic[0],
            p95_latency=0,
            cpu_utilization=0.0,
            queue_depth=0,
            replicas_active=self._active_replicas,
            replicas_pending=0,
            provisioning_time_remaining=0,
        )

        return self._observation(self._traffic[0]), {}

    def step(self, action):
        if self._metrics is None:
            raise RuntimeError("reset() must be called before step()")
        if self._step_index >= len(self._traffic):
            raise RuntimeError("step() called after the episode ended")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action!r}")

        action = int(action)
        action_accepted = self._queue_action(action)
        pending_delta = sum(delta for _, delta in self._pending_actions)
        provisioning_time = min(
            (remaining for remaining, _ in self._pending_actions),
            default=0,
        )

        self._metrics = self.simulation.calculate_metrics(
            requests_per_second=self._traffic[self._step_index],
            replicas_active=self._active_replicas,
            replicas_pending=pending_delta,
            provisioning_time_remaining=provisioning_time,
            previous_queue_depth=self._queue_depth,
        )
        self._queue_depth = self._metrics.queue_depth

        reward = -(
            self._metrics.replicas_active * self.simulation.replica_cost
        )
        if self._metrics.p95_latency > self.simulation.latency_limit_ms:
            latency_excess = (
                self._metrics.p95_latency
                - self.simulation.latency_limit_ms
            ) / self.simulation.latency_limit_ms
            reward -= self.simulation.latency_penalty
            reward -= self.simulation.latency_severity_penalty * min(
                latency_excess,
                self.simulation.latency_severity_cap,
            )
        if action != self.NO_OP and not action_accepted:
            reward -= self.simulation.invalid_action_penalty

        self._advance_provisioning()
        self._step_index += 1
        truncated = self._step_index >= len(self._traffic)
        next_traffic = 0.0 if truncated else self._traffic[self._step_index]

        return (
            self._observation(next_traffic),
            float(reward),
            False,
            truncated,
            {"action_accepted": action_accepted},
        )

    def _queue_action(self, action):
        if action == self.NO_OP:
            return True

        delta = 1 if action == self.SCALE_UP else -1
        projected_replicas = self._active_replicas + sum(
            pending_delta for _, pending_delta in self._pending_actions
        )
        if not (
            self.simulation.min_replicas
            <= projected_replicas + delta
            <= self.simulation.max_replicas
        ):
            return False

        delay = (
            self.simulation.provisioning_delay
            if delta > 0
            else self.simulation.scale_down_delay
        )
        self._pending_actions.append((delay, delta))
        return True

    def _advance_provisioning(self):
        remaining_actions = []
        for remaining, delta in self._pending_actions:
            remaining -= 1
            if remaining <= 0:
                self._active_replicas += delta
            else:
                remaining_actions.append((remaining, delta))
        self._pending_actions = remaining_actions

    def _observation(self, traffic_rps):
        pending_delta = sum(delta for _, delta in self._pending_actions)
        provisioning_time = min(
            (remaining for remaining, _ in self._pending_actions),
            default=0,
        )
        return {
            "traffic_rps": np.asarray(traffic_rps, dtype=np.float32),
            "p95_latency_ms": np.asarray(
                self._metrics.p95_latency, dtype=np.float32
            ),
            "cpu_utilization": np.asarray(
                self._metrics.cpu_utilization, dtype=np.float32
            ),
            "queue_depth": np.asarray(
                self._metrics.queue_depth, dtype=np.float32
            ),
            "active_replicas": np.asarray(
                self._active_replicas, dtype=np.int32
            ),
            "pending_replica_delta": np.asarray(
                pending_delta, dtype=np.int32
            ),
            "provisioning_time_remaining": np.asarray(
                provisioning_time, dtype=np.int32
            ),
        }
