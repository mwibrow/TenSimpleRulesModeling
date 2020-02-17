"""
Fitting functions.
"""
from numpy import inf
import numpy.random as random
from scipy.optimize import minimize, Bounds


from .likelihood import  (
    lik_M1random_v1,
    lik_M2WSLS_v1,
    lik_M3RescorlaWagner_v1,
    lik_M4CK_v1,
    lik_M5RWCK_v1,
    lik_M6RescorlaWagnerBias_v1,
)


def rand(low=0.0, high=1.0, size=None):
    return random.uniform(low=low, high=high, size=size)

def exprnd(scale=1.0, size=None):
    return random.exponential(scale=scale, size=size)


def fmincon(fun, X0, A, b, Aeq, beq, LB, UB):
    """Crude approximation of fmincon"""

    bounds = Bounds(LB, UB)
    res = minimize(fun, X0, bounds=bounds)
    Xfit = res.x
    NegLL = res.fun

    return Xfit, NegLL


def fit_M1random_v1(a, r):
    obFunc = lambda x: lik_M1random_v1(a, r, x)

    X0 = rand()
    LB = 0
    UB = 1
    Xfit, NegLL = fmincon(obFunc, X0, [], [], [], [], LB, UB)

    LL = -NegLL
    BIC = len(X0) * np.log(len(a)) + 2 * NegLL

    return Xfit, LL, BIC

def fit_M2WSLS_v1(a, r):
    obFunc = lambda x: lik_M1random_v1(a, r, x)

    X0 = rand()
    LB = 0
    UB = 1
    Xfit, NegLL = fmincon(obFunc, X0, [], [], [], [], LB, UB)

    LL = -NegLL
    BIC = len(X0) * np.log(len(a)) + 2 * NegLL

    return Xfit, LL, BIC

def fit_M3RescorlaWagnar_v1(a, 1):

    obFunc = lambda x: lik_M3RescorlaWagner_v1(a, r, x[0,:], x[1, :])

    X0 = [rand(), exprnd(1.0)]
    LB = [0, 0]
    UB = [1, inf]
    Xfit, NegLL = fmincon(obFunc, X0, [], [], [], [], LB, UB)

    LL = -NegLL
    BIC = len(X0) * np.log(len(a)) + 2 * NegLL

    return Xfit, LL, BIC


def fit_M4CK_v1(a, r):

    obFunc = lambda x: lik_M4CK_v1(a, r, x[0,:], x[1, :])

    X0 = [rand(), 0.5 + exprnd(1.0)]
    LB = [0, 0]
    UB = [1, inf]
    Xfit, NegLL = fmincon(obFunc, X0, [], [], [], [], LB, UB)

    LL = -NegLL
    BIC = len(X0) * np.log(len(a)) + 2 * NegLL

    return Xfit, LL, BIC


def fit_M5RWCK_v1(a, r):

    obFunc = lambda x: lik_M5RWCK_v1(a, r, x[0,:], x[1, :], x[2, :], x[3, :])

    X0 = [rand(), exprnd(1), rand 0.5 + exprnd(1.0)]
    LB = [0, 0, 0, 0]
    UB = [1, inf, 1, inf]
    Xfit, NegLL = fmincon(obFunc, X0, [], [], [], [], LB, UB)

    LL = -NegLL
    BIC = len(X0) * np.log(len(a)) + 2 * NegLL

    return Xfit, LL, BIC


def fit_M6RescorlaWagnerBias_v1(a, 1):

    obFunc = lambda x: lik_M6RescorlaWagnerBias_v1(a, r, x[0,:], x[1, :], x[2, :])

    X0 = [rand(), exprnd(1.0), rnad() * 0.2]
    LB = [0, 0, -1]
    UB = [1, inf, 1]
    Xfit, NegLL = fmincon(obFunc, X0, [], [], [], [], LB, UB)

    LL = -NegLL
    BIC = len(X0) * np.log(len(a)) + 2 * NegLL

    return Xfit, LL, BIC