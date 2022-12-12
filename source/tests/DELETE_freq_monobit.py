import numpy as np
import numpy.typing as npt
import scipy.stats as stats
from typing import Tuple
from icecream import ic


def fmb_stat(x: npt.NDArray[float],  k: int) -> float:
    stat = 0
    return stat


def fmb_pval(k: int, t: float) -> float:
    return 0


def fmb_test(x: npt.NDArray[float], k: int, alpha: float = 0.05) -> Tuple[float, float, bool]:
    t = fmb_stat(x, k)
    p = fmb_pval(k, t)
    return p, t, p < alpha


if __name__ == "__main__":
    alpha = 0.05
    x = np.array([0.05, 0.02, 0.055, 0.99, 0.98])
    k = 10

    ic(fmb_stat(x, k))
    stat = fmb_stat(x, k)
    ic(fmb_pval(k, stat))
    ic(fmb_test(x, k, alpha))
