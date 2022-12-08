import numpy as np
import numpy.typing as npt
from typing import Tuple


# statystyka testowa
def ks_stat(x: npt.NDArray[float],  k: int) -> float:
    return 0


# p-wartosc
def ks_pval(n: int, t: float) -> float:
    return 0


# wynik testu
def ks_test(x: npt.NDArray[float], k: int, alpha: float = 0.05) -> Tuple[float, float, bool]:
    n = x.shape[0]  # TODO
    t = ks_stat(x, k)
    p = ks_pval(n, t)
    return (p, t, p < alpha)