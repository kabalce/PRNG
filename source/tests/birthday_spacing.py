import numpy as np
import numpy.typing as npt
import scipy.stats as stats
from typing import Tuple
from icecream import ic
"""
Birthday Spacing Test
"""

def bds_stat(x: npt.NDArray[float], k: int) -> float:
    Y = (x.reshape(-1,  1) >= (np.arange(k) / k)[np.newaxis, :]).sum(axis=1) - 1
    Ys = np.sort(Y)
    S = Ys[1:] - Ys[:-1]
    K = np.bincount(S)
    stat = (K[K > 0] - 1).sum()
    return stat


def bds_pval(l: float, t: float) -> float:
    return 1 - stats.poisson(l).cdf(t)


def bds_test(x: npt.NDArray[float], alpha: float = 0.05) -> Tuple[float, float, bool]:
    n = x.shape[0]
    k = np.ceil(n ** 2.4)
    l = n ** 3 / (4 * k)
    t = bds_stat(x, k)
    p = bds_pval(l, t)
    return p, t, p < alpha


if __name__ == "__main__":
    alpha = 0.05
    x = np.array([0.05, 0.02, 0.055, 0.99, 0.98])
    n = x.shape[0]
    k = int(np.ceil(n ** 2.4))
    l = n ** 3 / (4 * k)

    ic(bds_stat(x, k))
    stat = bds_stat(x, k)
    ic(bds_pval(l, stat))
    ic(bds_test(x, alpha))
