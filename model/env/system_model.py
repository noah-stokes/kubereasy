# models the simulation infra behaviour
from dataclasses import dataclass

import yaml

with open("configs/system.yaml", "r") as f:
    config = yaml.safe_load(f)


@dataclass
class Metrics:
    requests_per_second: int
    p95_latency: int
    cpu_utilization: float
    queue_depth: int
    replicas_active: int
    replicas_pending: int
    provisioning_time_remaining: int


@dataclass
class Simulation:
    replica_capacity: int
    provisioning_delay: int
    scale_down_delay: int
    min_replicas: int
    max_replicas: int
    replica_cost: float
    latency_limit_ms: int
    latency_penalty: float
    latency_severity_penalty: float = 2.0
    latency_severity_cap: float = 5.0
    invalid_action_penalty: float = 10.0

    def calculate_metrics(
        self,
        requests_per_second: int,
        replicas_active: int,
        replicas_pending: int = 0,
        provisioning_time_remaining: int = 0,
        previous_queue_depth: int = 0,
    ) -> Metrics:

        total_capacity = replicas_active * self.replica_capacity
        total_demand = previous_queue_depth + requests_per_second

        # Avoid division by zero
        if total_capacity <= 0:
            cpu_utilization = 1.0
            queue_depth = total_demand
            p95_latency = 10_000

        else:
            demand_ratio = total_demand / total_capacity

            # CPU represented from 0.0 -> 1.0
            cpu_utilization = min(demand_ratio, 1.0)

            # Requests that cannot be handled this timestep
            queue_depth = max(
                0,
                total_demand - total_capacity
            )

            # Base latency rises as the system becomes more utilized
            base_latency = 50

            utilization_latency = 150 * (cpu_utilization ** 2)

            # Queueing causes a much larger latency increase
            queue_latency = (
                1000 * queue_depth / total_capacity
            )

            p95_latency = int(
                base_latency
                + utilization_latency
                + queue_latency
            )

        return Metrics(
            requests_per_second=requests_per_second,
            p95_latency=p95_latency,
            cpu_utilization=cpu_utilization,
            queue_depth=queue_depth,
            replicas_active=replicas_active,
            replicas_pending=replicas_pending,
            provisioning_time_remaining=provisioning_time_remaining,
        )
