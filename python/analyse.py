import numpy as np


def analysis_WSLS_v1(a, r):

    aLast = np.insert(a[:-1], 0, np.NaN)
    stay = aLast == a
    rLast = np.insert(r[:-1], 0, np.NaN)

    winStay = np.nanmean(stay[rLast == 1])
    loseStay = np.nanmean(stay[rLast == 0])
    return loseStay, winStay
