import numpy as np
import numpy.typing as npt
import scipy.stats as stats
from typing import Tuple
from icecream import ic


def ks_test(x: npt.NDArray[float], alpha: float = 0.05) -> Tuple[float, float, bool]:
    res = stats.kstest(x, stats.uniform.cdf)
    return (res.pvalue, res.statistic, res.pvalue < alpha)


if __name__ == "__main__":
    alpha = 0.05
    x = np.array([0.05, 0.02, 0.055, 0.99, 0.98])

    tr = stats.kstest(x, stats.uniform.cdf)
    ic(tr.statistic)
    ic(tr.pvalue)
    ic(tr.pvalue < alpha)
