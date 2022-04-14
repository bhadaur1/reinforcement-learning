import uuid
import numpy as np


class TestBed(object):
    def __init__(self, n, seed):
        self.n = n
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.mu = self.rng.normal(0, 1.0, self.n)
        self.sig = np.ones(self.n, dtype=np.float32)
        self.id = uuid.uuid1()

    def get_dist(self, index, size=1):
        return self.rng.normal(self.mu[index], self.sig[index], size=size)

    def __str__(self) -> str:
        return (
            f"{self.n}-armed Testbed\n"
            f"Id: {self.id.hex}\n"
            f"Random Seed: {self.seed}\n"
            f"Max. Expected Value: {np.max(self.mu)}\n"
        )
