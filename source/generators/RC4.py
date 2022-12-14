import numpy as np


def ksa(K: list[int], m: int) -> list[int]:
    L = len(K)
    S = np.arange(m)
    j = 0
    for i in range(m):
        j = (j + S[i] + K[i % L]) % m
        S[i], S[j] = S[j], S[i]
    return S


def rc4_gen(S: list[int], length: int) -> list[int]:
    i, j = 0, 0
    res = []
    m = len(S)
    for _ in range(length):
        i = (i + 1) % m
        j = (j + S[i]) % m
        S[i], S[j] = S[j], S[i]
        res.append(S[(S[i] + S[j]) % m] / m)
    return res


if __name__ == "__main__":
    S = ksa([2, 5, 19, 1], 1000)
    print(rc4_gen(S, 100))

