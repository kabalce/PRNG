import numpy as np


def glcg_gen(M: int, a_seq, seed_seq)  # add types
    """LCG - linear congruential generator"""
    while True:
        x = np.sum(a_seq * seed_seq) % M
        seed_seq[:] =  np.roll(seed_seq, 1)
        seed_seq[-1] = x
        yield x

