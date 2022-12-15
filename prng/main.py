from prng.generators.GLCG import glcg_gen
from prng.generators.LCG import lcg_gen
from prng.generators.RC4 import rc4_gen, ksa

from tests.birthday_spacing import bds_test
from tests.chisq import chisq_test
from tests.runs import runs_test
from tests.ks import ks_test

import numpy as np

from pathlib import Path
import datetime

GENERATORS = [glcg_gen, lcg_gen, rc4_gen]
TESTS = [ks_test, bds_test, chisq_test, runs_test]
DATA_PATH = Path(__file__).parent.parent / 'data'

M = 2 ** 15
NOW = datetime.datetime.now()
LENGTH = 10 ** 2
SUBSEQ_LEN = 10 ** 1

def generate_data():
    lcg_data = lcg_gen(M, NOW.hour, NOW.minute, NOW.second, LENGTH)

    glcg_data = glcg_gen(M,
             np.array([NOW.day, NOW.month, NOW.year, NOW.hour, NOW.minute, NOW.second]),
             np.array([NOW.day + 42, NOW.month + 42, NOW.year + 42, NOW.hour + 42, NOW.minute + 42, NOW.second + 42]),
             LENGTH)

    rc4_data = rc4_gen(ksa(np.array([NOW.day, NOW.month, NOW.year, NOW.hour, NOW.minute, NOW.second]),  M), LENGTH)

    return lcg_data, glcg_data, rc4_data


def test_1st_level(data):
    for t in TESTS:
        print(t.__name__)
        print(t(data))
    return None


def test_2nd_level(data):
    res = []
    for d in data.reshape(-1, SUBSEQ_LEN):
        res.append(test_1st_level(d))
    return res


if __name__ == "__main__":
    d1, d2, d3 = generate_data()
    test_1st_level(np.array(d1).astype('float32'))
    test_2nd_level(np.array(d1).astype('float32'))


"""
TODO:
    - zastanów się na jakich typach danych chcesz wykonywać testy
    - klasa tester może sobie móc konwertować między typami
    - użyj Magdowego generatora i opakuj w klasę
    - popraw implementację BDS test
    - zmień implementację na jakieś zgrabne klasy
    - zastanów się jak zareprezentować wyniki
    - eksperymenty i raport
"""