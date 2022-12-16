import numpy as np
import numpy.typing as npt
import scipy.stats as stats
from typing import Tuple, Optional
from abc import ABC
from scipy.special import erfc
from bitarray import bitarray


class Tester(ABC):
    def __init__(self, data: npt.NDArray[int], m: Optional[int] = None, no_folds: int = 100):
        self.M = data.max() if m is None else m
        self.integers = data
        self.uniform = None
        self.binaries = None
        self.n = self.integers.shape[0]
        self.fold_index = [(self.n * i) // no_folds for i in range(no_folds + 1)]

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

    @staticmethod
    def _chi2_stat(uniform: npt.NDArray[float], k: int = 1000) -> float:
        n = uniform.shape[0]
        y = np.flip(np.bincount((uniform.reshape(-1, 1) < (np.arange(k) / k)[np.newaxis, :]).sum(axis=1), minlength=k))
        stat = (np.square(y - n / k) / (n / k)).sum()
        return stat

    @staticmethod
    def _chi2_pval(k: int, t: float) -> float:
        stats.chi2(k - 1).cdf(t)
        return 1 - stats.chi2(k - 1).cdf(t)

    def chi2_test(self, k: int = 1000, alpha: float = 0.05) -> Tuple[float, float, bool]:
        self._provide_uniform()
        t = self._chi2_stat(self.uniform, k)
        p = self._chi2_pval(k, t)
        return p, t, p < alpha

    def ch2_2nd_level_test(self, k: int = 1000, alpha: float = 0.05) -> Tuple[float, float, bool]:
        self._provide_uniform()
        p_values = np.array([self._chi2_pval(k, self._chi2_stat(self.uniform[self.fold_index[i]: self.fold_index[i + 1]], k)) for i in range(len(self.fold_index) - 1)])
        t = self._chi2_stat(p_values)
        p = self._chi2_pval(k, t)
        return p, t, p < alpha

    @staticmethod
    def _bds_stat(integers: npt.NDArray[int], k: int) -> float:
        # y = (integers.reshape(-1, 1) >= ).sum(axis=1) - 1
        indexes = np.arange(k)
        y = np.array([(i >= indexes).sum() - 1 for i in integers])
        ys = np.sort(y)
        s = ys[1:] - ys[:-1]
        k = np.bincount(s)
        stat = (k[k > 0] - 1).sum()
        return stat

    @staticmethod
    def _bds_pval(l: float, t: float) -> float:
        return 1 - stats.poisson(l).cdf(t)

    def bds_test(self, alpha: float = 0.05, k: int = 512) -> Tuple[float, float, bool]:
        l = self.n ** 3 / (4 * k)
        t = self._bds_stat(self.integers, k)
        p = self._bds_pval(l, t)
        return p, t, p < alpha

    def bds_2nd_level_test(self, k: int = 10, k_loc: int = 512, alpha: float = 0.05) -> Tuple[float, float, bool]:
        l = int(self.n ** 3 / (4 * k_loc))
        poi_stats = np.array([self._bds_stat(self.integers[self.fold_index[i]: self.fold_index[i + 1]], k_loc) for i in range(len(self.fold_index) - 1)])

        distr = stats.poisson(l)
        quantiles = np.flip(distr.ppf(np.arange(k) / k).astype(int))
        prob_i = distr.cdf(quantiles)
        prob_i = np.concatenate([np.array([1]), prob_i[:-1]]) - prob_i

        y = np.flip(np.bincount(k - (poi_stats.reshape(-1, 1) < quantiles[np.newaxis, :]).sum(axis=1), minlength=k))
        t = (np.square(y - prob_i) / prob_i).sum()
        p = 1 - stats.chi2(k - 1).cdf(t)
        return p, t, p < alpha

    def ks_test(self, alpha: float = 0.05) -> Tuple[float, float, bool]:
        self._provide_uniform()
        res = stats.kstest(self.uniform, stats.uniform().cdf)
        return res.pvalue, res.statistic, res.pvalue < alpha

    def ks_2nd_level_test(self, alpha: float = 0.05) -> Tuple[float, float, bool]:
        self._provide_uniform()
        p_values = np.array([stats.kstest(self.uniform[self.fold_index[i]: self.fold_index[i + 1]], stats.uniform().cdf).pvalue for i in range(len(self.fold_index) - 1)])
        res = stats.kstest(p_values, stats.uniform().cdf)
        return res.pvalue, res.statistic, res.pvalue < alpha

    @staticmethod
    def _runs_pre_test(binaries: bitarray) -> Tuple[bool, float]:
        pi = len(binaries.search(bitarray('1'))) / len(binaries)
        tau = 2 / np.sqrt(len(binaries))
        return pi >= tau, pi

    @staticmethod
    def _runs_stat(binaries: bitarray) -> float:
        stat = len(binaries.search(bitarray('01'))) + len(binaries.search(bitarray('10'))) + 2
        return stat

    @staticmethod
    def _runs_pval(n: int, pi: float, t: float) -> float:
        p = erfc(abs(t - 2 * n * pi * (1 - pi)) / (2 * np.sqrt(2 * n) * pi * (1 - pi)))
        return p

    def runs_test(self, alpha: float = 0.05) -> Tuple[Optional[float], Optional[float], bool]:
        self._provide_binary()
        res_pre, pi = self._runs_pre_test(self.binaries)
        if res_pre:
            t = self._runs_stat(self.binaries)
            p = self._runs_pval(self.n, pi, t)
            return p, t, p < alpha
        else:
            return None, None, False


if __name__ == "__main__":
    from prng.generators.LCG import LinearCongruentialGenerator as LCG
    generator = LCG()
    data = generator.sample(2 ** 15)
    tester = Tester(data)
    print(tester.runs_test())
    print(tester.chi2_test())
    print(tester.ch2_2nd_level_test())
    print(tester.bds_test())
    print(tester.bds_2nd_level_test())
    print(tester.ks_test())
    print(tester.ks_2nd_level_test())
