import numpy as np
import numpy.typing as npt
import scipy.stats as stats
from typing import Tuple
from icecream import ic
"""
Birthday Spacing test
"""

def bd_stat(x: npt.NDArray[float],  k: int) -> float:
    Y = (x.reshape(-1,  1) >= (np.arange(k) / k)[np.newaxis, :]).sum(axis=1) - 1
    Ys = np.sort(Y)
    S = Ys[1:] - Ys[:-1]
    K = np.bincount(S)
    stat = 0
    return stat


def bd_pval(k: int, t: float) -> float:
    return 0


def bd_test(x: npt.NDArray[float], k: int, alpha: float = 0.05) -> Tuple[float, float, bool]:
    t = bd_stat(x, k)
    p = bd_pval(k, t)
    return p, t, p < alpha


if __name__ == "__main__":
    alpha = 0.05
    x = np.array([0.05, 0.02, 0.055, 0.99, 0.98])
    k = 10

    ic(bd_stat(x, k))
    stat = bd_stat(x, k)
    ic(bd_pval(k, stat))
    ic(bd_test(x, k, alpha))
