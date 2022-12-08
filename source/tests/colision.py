import numpy as np
import numpy.typing as npt
from typing import Tuple


# statystyka testowa
def colision_stat(x: npt.NDArray[float],  k: int) -> float:
    return 0


# p-wartosc
def colision_pval(n: int, t: float) -> float:
    return 0


# wynik testu
def colision_test(x: npt.NDArray[float], k: int, alpha: float = 0.05) -> Tuple[float, float, bool]:
    n = x.shape[0]  # TODO
    t = colision_stat(x, k)
    p = colision_pval(n, t)
    return (p, t, p < alpha)