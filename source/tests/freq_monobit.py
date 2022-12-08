import numpy as np
import numpy.typing as npt
from typing import Tuple


# statystyka testowa
def freq_monobit_stat(x: npt.NDArray[float],  k: int) -> float:
    return 0


# p-wartosc
def freq_monobit_pval(n: int, t: float) -> float:
    return 0


# wynik testu
def freq_monobit_test(x: npt.NDArray[float], k: int, alpha: float = 0.05) -> Tuple[float, float, bool]:
    n = x.shape[0]  # TODO
    t = freq_monobit_stat(x, k)
    p = freq_monobit_pval(n, t)
    return (p, t, p < alpha)