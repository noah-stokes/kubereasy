from math import ceil

from env.DevOpsEnv import DevOpsEnv
from env.system_model import config


CPU_SCALE_UP_THRESHOLD = 0.75
CPU_SCALE_DOWN_THRESHOLD = 0.35
TARGET_UTILIZATION = 0.70


def cpu_autoscaler(observation):
    active = int(observation["active_replicas"])
    pending = int(observation["pending_replica_delta"])
    if pending:
        return DevOpsEnv.NO_OP

    cpu = float(observation["cpu_utilization"])
    if cpu >= CPU_SCALE_UP_THRESHOLD and active < config["max_replicas"]:
        return DevOpsEnv.SCALE_UP
    if cpu <= CPU_SCALE_DOWN_THRESHOLD and active > config["min_replicas"]:
        return DevOpsEnv.SCALE_DOWN
    return DevOpsEnv.NO_OP


def traffic_autoscaler(observation):
    desired = ceil(
        float(observation["traffic_rps"])
        / (config["replica_capacity"] * TARGET_UTILIZATION)
    )
    desired = min(
        config["max_replicas"], max(config["min_replicas"], desired)
    )
    projected = (
        int(observation["active_replicas"])
        + int(observation["pending_replica_delta"])
    )

    if projected < desired:
        return DevOpsEnv.SCALE_UP
    if projected > desired:
        return DevOpsEnv.SCALE_DOWN
    return DevOpsEnv.NO_OP


def no_op_autoscaler(observation):
    return DevOpsEnv.NO_OP


BASELINES = {
    "cpu": cpu_autoscaler,
    "traffic": traffic_autoscaler,
    "no_op": no_op_autoscaler,
}
