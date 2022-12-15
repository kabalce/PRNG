from collections.abc import Generator
import numpy as np


class AbstractGenerator(Generator):
    def __init__(self, m: int = 2 ** 18):
        self.M = m

    def send(self, ignored_arg: None = None):
        """
        Generate the next number and update needed parameters
        :param value:
        :return: random integer
        """
        return_value = 0
        return return_value

    def throw(self, typ=None, value=None, traceback=None):
        raise StopIteration

    def sample(self, length: int = 1):
        """
        Sample numbers from iterator
        :param length: length of sampled sequence
        :return: np array of shape (length, )
        """
        return np.array([next(self) for _ in range(length)])


if __name__ == "__main__":
    generator = AbstractGenerator()
    print(generator.sample(10))
