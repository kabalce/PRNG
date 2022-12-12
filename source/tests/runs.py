import numpy as np
import numpy.typing as npt
import scipy.stats as stats
from scipy.special import erfc
from bitarray import bitarray
from typing import Tuple
from icecream import ic
"""
Runs test, 2.3  http://www.math.uni.wroc.pl/~lorek/teaching/files/nistspecialpublication800-22r1a.pdf
"""


def to_bytes(x: npt.NDArray[float]) -> bitarray:
    byt = bytearray(x)
    ba = bitarray()
    for b in bytes(byt):
        ba += bitarray(b)
    return ba


def runs_pre_test(ba: bitarray) -> bool:
    pi = len(ba.search(bitarray('1'))) / len(ba)
    tau = 2 / np.sqrt(len(ba))
    return pi >= tau, pi


def runs_stat(ba: bitarray) -> float:
    stat = len(ba.search(bitarray('01'))) + len(ba.search(bitarray('10'))) + 2
    return stat


def runs_pval(pi: float, t: float, n: int) -> float:
    p = erfc(abs(t - 2 * n * pi * (1 - pi)) / (2 * np.sqrt(2 * n) * pi * (1 - pi)))
    return p



def runs_test(x: npt.NDArray[float], alpha: float = 0.05) -> Tuple[float, float, bool]:
    ba = to_bytes(x)
    res_pre, pi = runs_pre_test(ba)
    if res_pre:
        t = runs_stat(ba)
        p = runs_pval(pi, t, len(ba))
        return p, t, p < alpha
    else:
        return None, None, False


if __name__ == "__main__":
    alpha = 0.05
    x = np.array([0.05, 0.02, 0.055, 0.99, 0.98, 1.00, 0.22, 0.576, .371])

    ic(runs_test(x, alpha))
