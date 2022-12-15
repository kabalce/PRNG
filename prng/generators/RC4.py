from prng.generators.abstract_generator import AbstractGenerator
from datetime import datetime
import numpy as np
import numpy.typing as npt
from typing import Optional


class RivestCipher4Generator(AbstractGenerator):
    def __init__(self, m: int = 2 ** 18, k_seq: Optional[npt.NDArray[int]] = None):
        now = datetime.now()
        super().__init__(m)

        self.k_seq = k_seq if k_seq is not None else np.array(
            [(now.microsecond // (i + 1)) % self.M for i in range(10)])
        self.seed = self._ksa()
        self.i, self.j = 0, 0

    def _ksa(self) -> npt.NDArray[int]:
        l = len(self.k_seq)
        perm = np.arange(self.M)
        j = 0
        for i in range(self.M):
            j = (j + perm[i] + self.k_seq[i % l]) % self.M
            perm[i], perm[j] = perm[j], perm[i]
        return perm

    def send(self, ignored_arg: None = None):
        self.i = (self.i + 1) % self.M
        self.j = (self.j + self.seed[self.i]) % self.M
        self.seed[self.i], self.seed[self.j] = self.seed[self.j], self.seed[self.i]
        value = self.seed[(self.seed[self.i] + self.seed[self.j]) % self.M]
        return value


if __name__ == "__main__":
    generator = RivestCipher4Generator()
    print(generator.sample(10))
