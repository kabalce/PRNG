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
    k = int(np.sqrt(x.shape[0]))
    y = (x.reshape(-1,  1) >= (np.arange(k) / k)[np.newaxis, :]).sum(axis=1) - 1
    byt = bytearray(y)
    # byt = bytearray(x)
    ba = bitarray()
    for b in bytes(byt):
        ba += bitarray(b)
    return ba


def runs_pre_test(ba: bitarray, n: int) -> bool:
    pi = len(ba.search(bitarray('1'))) / len(ba)
    tau = 2 / np.sqrt(n)
    return pi >= tau, pi
    # return True, pi


def runs_stat(ba: bitarray) -> float:
    stat = len(ba.search(bitarray('01'))) + len(ba.search(bitarray('10'))) + 2
    return stat


def runs_pval(pi: float, t: float, n: int) -> float:
    ic(abs(t - 2 * n * pi * (1 - pi)) / (2 * np.sqrt(2 * n) * pi * (1 - pi)))
    p = erfc(abs(t - 2 * n * pi * (1 - pi)) / (2 * np.sqrt(2 * n) * pi * (1 - pi)))
    return p



def runs_test(x: npt.NDArray[float], alpha: float = 0.05) -> Tuple[float, float, bool]:
    n = x.shape[0]
    ba = to_bytes(x)
    ic(ba)
    res_pre, pi = runs_pre_test(ba, n)
    if res_pre:
        t = runs_stat(ba)
        p = runs_pval(pi, t, n)
        return p, t, p < alpha
    else:
        return None, None, False





if __name__ == "__main__":
    alpha = 0.05
    x = np.array([0.05, 0.02, 0.055, 0.99, 0.98, 1.00, 0.22, 0.576, .371])
    x = np.random.uniform(size=100)
    # ic(runs_stat(x, k))
    # stat = runs_stat(x, k)
    # ic(runs_pval(l, stat))
    ic(runs_test(x, alpha))
