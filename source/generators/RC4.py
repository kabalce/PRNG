import numpy as np


def ksa(K, m):
    L = len(K)
    S = np.arange(m)
    j = 0
    for i in range(m):
        j = (j + S[i] + K[i % L]) % m
        S_i = S[i]
        S_j = S[j]
        S[i] = S_j
        S[j] = S_i
    i = 0
    j = 0
    return(S)


def rc4_gen(S, m):
    i = 0
    j = 0
    while True:
        i = (i + 1) % m
        j = (j + S[i]) % m
        S_i = S[i]
        S_j = S[j]
        S[i] = S_j
        S[j] = S_i
        y = S[(S[i] + S[j]) % m]
        yield y


