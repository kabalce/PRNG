from abstract_generator import AbstractGenerator
from datetime import datetime
from typing import Optional

class LinearCongruentialGenerator(AbstractGenerator):
    def __init__(self, m: int = 2 ** 18, a: Optional[int] = None, c: Optional[int] = None, seed: Optional[int] = None):
        """

        :param m: modulus
        :param a: multiplier
        :param c: summand
        :param seed: initial value
        """
        now = datetime.now()
        super().__init__(m)
        self.a = a if a is not None else now.minute % self.M
        self.c = c if c is not None else now.second % self.M
        self.seed = seed if a is not None else now.microsecond % self.M

    def send(self, ignored_arg: None = None):
        self.seed = (self.a * self.seed + self.c) % self.M
        return self.seed


if __name__ == "__main__":
    generator = LinearCongruentialGenerator()
    print(generator.sample(10))
