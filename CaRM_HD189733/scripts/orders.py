import numpy as np


def orderl(interval):
    il = len(interval)
    ordi = []
    for k in range(int(il/2)):
        ordi = np.append(ordi, np.arange(interval[2*k], interval[2*k+1]+1, 1))
    return (ordi)
