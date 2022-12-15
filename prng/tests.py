import numpy as np
import numpy.typing as npt
import scipy.stats as stats
from typing import Tuple, Optional
from abc import ABC
from scipy.special import erfc
from bitarray import bitarray


class Tester(ABC):
    def __init__(self, data: npt.NDArray[int], m: Optional[int] = None):
        self.M = data.max() if m is None else m
        self.integers = data
        self.uniform = None
        self.binaries = None
        self.n = self.integers.shape[0]

    def _provide_uniform(self):
        if self.uniform is None:
            self.uniform = self.integers / self.M

    def _provide_binary(self):
        if self.binaries is None:
            byt = bytearray(self.integers)
            ba = bitarray()
            for b in bytes(byt):
                ba += bitarray(b)
            self.binaries = ba

    def _chi2_stat(self, k: int = 1000) -> float:
        y = np.flip(np.bincount((self.uniform.reshape(-1, 1) < (np.arange(k) / k)[np.newaxis, :]).sum(axis=1), minlength=k))
        stat = (np.square(y - self.n / k) / (self.n / k)).sum()
        return stat

    @staticmethod
    def _chi2_pval(k: int, t: float) -> float:
        stats.chi2(k - 1).cdf(t)
        return 1 - stats.chi2(k - 1).cdf(t)

    def chi2_test(self, k: int = 1000, alpha: float = 0.05) -> Tuple[float, float, bool]:
        self._provide_uniform()
        t = self._chi2_stat(k)
        p = self._chi2_pval(k, t)
        return p, t, p < alpha

    def _bds_stat(self, k: int) -> float:
        y = (self.integers.reshape(-1, 1) >= (np.arange(k))[np.newaxis, :]).sum(axis=1) - 1
        ys = np.sort(y)
        s = ys[1:] - ys[:-1]
        k = np.bincount(s)
        stat = (k[k > 0] - 1).sum()
        return stat

    @staticmethod
    def _bds_pval(l: float, t: float) -> float:
        return 1 - stats.poisson(l).cdf(t)

    def bds_test(self, alpha: float = 0.05) -> Tuple[float, float, bool]:
        k = np.ceil(self.n ** 2.4)
        l = self.n ** 3 / (4 * k)
        t = self._bds_stat(k)
        p = self._bds_pval(l, t)
        return p, t, p < alpha

    def ks_test(self, alpha: float = 0.05) -> Tuple[float, float, bool]:
        self._provide_uniform()
        res = stats.kstest(self.uniform, stats.uniform().cdf)
        return res.pvalue, res.statistic, res.pvalue < alpha

    def _runs_pre_test(self) -> bool:
        pi = len(self.binaries.search(bitarray('1'))) / len(self.binaries)
        tau = 2 / np.sqrt(len(self.binaries))
        return pi >= tau, pi

    def _runs_stat(self) -> float:
        stat = len(self.binaries.search(bitarray('01'))) + len(self.binaries.search(bitarray('10'))) + 2
        return stat

    def _runs_pval(self, pi: float, t: float) -> float:
        p = erfc(abs(t - 2 * self.n * pi * (1 - pi)) / (2 * np.sqrt(2 * self.n) * pi * (1 - pi)))
        return p

    def runs_test(self, alpha: float = 0.05) -> Tuple[Optional[float], Optional[float], bool]:
        self._provide_binary()
        res_pre, pi = self._runs_pre_test()
        if res_pre:
            t = self._runs_stat()
            p = self._runs_pval(pi, t)
            return p, t, p < alpha
        else:
            return None, None, False
    
if __name__ == "__main__":
    from prng.generators.LCG import LinearCongruentialGenerator as LCG
    generator = LCG()
    data = generator.sample(100)
    tester = Tester(data)
    print(tester.runs_test())
    print(tester.chi2_test())
    print(tester.bds_test())
    print(tester.ks_test())
