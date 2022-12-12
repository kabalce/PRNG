import numpy as np
import numpy.typing as npt
import scipy.stats as stats
from typing import Tuple
from icecream import ic


def colision_stat(x: npt.NDArray[float],  k: int) -> float:
    stat = 0
    return stat


def colision_pval(k: int, t: float) -> float:
    return 0


def colision_test(x: npt.NDArray[float], k: int, alpha: float = 0.05) -> Tuple[float, float, bool]:
    t = colision_stat(x, k)
    p = colision_pval(k, t)
    return p, t, p < alpha


if __name__ == "__main__":
    alpha = 0.05
    x = np.array([0.05, 0.02, 0.055, 0.99, 0.98])
    k = 10

    ic(colision_stat(x, k))
    stat = colision_stat(x, k)
    ic(colision_pval(k, stat))
    ic(colision_test(x, k, alpha))
