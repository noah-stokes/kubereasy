from dataclasses import dataclass

import numpy as np
import yaml

with open("configs/workload.yaml", "r") as f:
    config = yaml.safe_load(f)

@dataclass
class WorkloadConfig:
    episode_length: int
    base_rps: float
    peak_rps: float
    period: int 
    noise_std: float 

class WorkloadGenerator:
    def __init__(self, config: WorkloadConfig, seed=None):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def constant(self):
        return np.full(
            self.config.episode_length,
            self.config.base_rps,
            dtype=float
        )

    def ramp(self):
        return np.linspace(
            self.config.base_rps,
            self.config.peak_rps,
            self.config.episode_length
        )

    def periodic(self):
        t = np.arange(self.config.episode_length)

        amplitude = (
            self.config.peak_rps - self.config.base_rps
        ) / 2

        workload = (
            self.config.base_rps
            + amplitude * np.sin(
                2 * np.pi * t / self.config.period
            )
        )

        return np.maximum(workload, 0)

    def add_noise(self, workload):
        noise = self.rng.normal(
            0,
            self.config.noise_std,
            len(workload)
        )

        return np.maximum(workload + noise, 0)
