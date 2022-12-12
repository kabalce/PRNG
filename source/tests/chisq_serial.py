import numpy as np
import numpy.typing as npt
import scipy.stats as stats
from typing import Tuple
from icecream import ic
"""
Test serii
"""

def chisq_stat(x: npt.NDArray[float],  k: int) -> float:
    Y = np.flip(np.bincount((x.reshape(-1,  1) < (np.arange(k) / k)[np.newaxis, :]).sum(axis=1), minlength=k))
    n = x.shape[0]
    stat = (np.square(Y - n / k) / (n / k)).sum()
    return stat


def chisq_pval(k: int, t: float) -> float:
    stats.chi2(k-1).cdf(t)
    return 1 - stats.chi2(k-1).cdf(t)


def chisq_test(x: npt.NDArray[float], k: int, alpha: float = 0.05) -> Tuple[float, float, bool]:
    t = chisq_stat(x, k)
    p = chisq_pval(k, t)
    return p, t, p < alpha


if __name__ == "__main__":
    alpha = 0.05
    x = np.array([0.05, 0.02, 0.055, 0.99, 0.98])
    k = 10

    ic(chisq_stat(x, k))
    stat = chisq_stat(x, k)
    ic(chisq_pval(k, stat))
    ic(chisq_test(x, k, alpha))
