import numpy as np
import numpy.typing as npt
from typing import Tuple


# statystyka testowa
def birthday_stat(x: npt.NDArray[float],  k: int) -> float:
    return 0


# p-wartosc
def birthday_pval(n: int, t: float) -> float:
    return 0


# wynik testu
def birthday_test(x: npt.NDArray[float], k: int, alpha: float = 0.05) -> Tuple[float, float, bool]:
    n = x.shape[0]  # TODO
    t = birthday_stat(x, k)
    p = birthday_pval(n, t)
    return (p, t, p < alpha)