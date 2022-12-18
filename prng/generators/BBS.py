from prng.generators.abstract_generator import AbstractGenerator
from datetime import datetime
from typing import Optional


class BlumBlumShubGenerator(AbstractGenerator):
    def __init__(self, m: int = 993319 * 23, seed: Optional[int] = None):
        """

        :param m: modulus
        :param a: multiplier
        :param c: summand
        :param seed: initial value
        """
        now = datetime.now()
        super().__init__(m)
        self.seed = seed if seed is not None else now.microsecond % self.M

    def send(self, ignored_arg: None = None) -> int:
        self.seed = (self.seed ** 2) % self.M
        return int(bin(self.seed)[-9:], 2)


if __name__ == "__main__":
    generator = BlumBlumShubGenerator()
    print(generator.sample(10))
