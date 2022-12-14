import numpy as np
# from collections.abc import Generator

def lcg_gen(M: int, a: int, c: int, seed: int, length: int) -> list[float]:  # -> Generator[int, None, None]:
    """LCG - linear congruential generator"""
    res = []
    s = seed
    for _ in range(length):
        s = (a * s + c) % M
        res.append(s / M)
    return res


if __name__  == "__main__":
    print(lcg_gen(10000, 17, 1, 48,  100))
