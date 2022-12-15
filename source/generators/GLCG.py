from abstract_generator import AbstractGenerator
from datetime import datetime
import numpy as np
import numpy.typing as npt
from typing import Optional

class GeneralizedLinearCongruentialGenerator(AbstractGenerator):
    def __init__(self, m: int = 2 ** 18, a_seq: Optional[npt.NDArray[int]] = None, seed: Optional[npt.NDArray[int]] = None):
        now = datetime.now()
        super().__init__(m)
        shape = 10

        if a_seq is not None and seed is not None:
            assert a_seq.shape == seed.shape, "a_seq and seed must be of the same shape"
            shape = a_seq.shape if a_seq is not None else seed.shape

        self.a_seq = a_seq if a_seq is not None else np.array([(now.microsecond // (i + 1)) % self.M for i in range(shape)])
        self.seed = seed if seed is not None else np.array([(now.microsecond // (i + 1)) % (self.M // 3) for i in range(shape)])

    def send(self, ignored_arg: None = None):
        value = np.sum(self.a_seq * self.seed) % self.M
        self.seed = np.concatenate([np.roll(self.seed, 1)[:-1], np.array([value])])
        return value


if __name__ == "__main__":
    generator = GeneralizedLinearCongruentialGenerator()
    print(generator.sample(10))
