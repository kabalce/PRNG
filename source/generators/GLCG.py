import numpy as np
import numpy.typing as npt

def glcg_gen(M: int, a_seq: npt.NDArray[int], seed_seq: npt.NDArray[int], length: int) -> list[float]:  # add types
    """GLCG - generalized linear congruential generator"""
    res = []
    s = seed_seq.copy()
    for _ in range(length):
        x = np.sum(a_seq * s) % M
        res.append(x / M)
        s = np.concatenate([np.roll(s, 1)[:-1], np.array([x])])
    return res


if __name__ == "__main__":
    print(glcg_gen(10000, np.array([2, 3, 4, 6]), np.array([4, 13,  45, 566]), 100))
