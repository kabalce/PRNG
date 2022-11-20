import numpy as np
# from collections.abc import Generator

def lcg_gen(M: int, a: int, c: int, seed: int)  # -> Generator[int, None, None]:
    """LCG - linear congruential generator"""
    while True:
        seed = (a * seed + c) % M
        yield seed

