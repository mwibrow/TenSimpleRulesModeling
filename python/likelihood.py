"""
Likelihood functions
"""

# pylint: disable=invalid-name,missing-function-docstring,too-many-arguments

import numpy as np


def lik_M1random_v1(a, _r, b):
    # note r is not used here but included to fit notation better with other
    # likelihood functions

    a = a - 1

    # compute choice probabilities
    p = np.array([b, 1 - b])

    # compute choice probability for actual choice
    choiceProb = p[a]

    negLL = -np.sum(np.log(choiceProb))
    return negLL


def lik_M2WSLS_v1(a, r, epsilon):

    # last reward/action (initialize as nan)
    rLast = np.nan
    aLast = np.nan

    a = a - 1
    T = len(a)

    choiceProb = np.zeros(T)

    # loop over all trial
    for t in range(T):

        # compute choice probabilities
        if np.isnan(rLast):

            # first trial choose randomly
            p = [0.5, 0.5]

        else:

            # choice depends on last reward
            if rLast == 1:

                # win stay (with probability 1-epsilon)
                p = epsilon / 2 * np.ones(2)
                p[aLast] = 1 - epsilon / 2

            else:

                # lose shift (with probability 1-epsilon)
                p = (1-epsilon / 2) * np.ones(2)
                p[aLast] = epsilon / 2

        # compute choice probability for actual choice
        choiceProb[t] = p[a[t]]

        aLast = a[t]
        rLast = r[t]

    # compute negative log-likelihood
    NegLL = -np.sum(np.log(choiceProb))

    return NegLL


def lik_M3RescorlaWagner_v1(a, r, alpha, beta):

    Q = np.array([0.5, 0.5])

    a = a - 1
    T = len(a)
    choiceProb = np.zeros(T)

    # loop over all trial
    for t in range(T):

       # compute choice probabilities
        p = np.exp(beta * Q) / np.sum(np.exp(beta * Q))

        # compute choice probability for actual choice
        choiceProb[t] = p[a[t]]

        # update values
        delta = r[t] - Q[a[t]]
        Q[a[t]] += alpha * delta

    # compute negative log-likelihood
    NegLL = -np.sum(np.log(choiceProb))
    return NegLL


def lik_M4CK_v1(a, _r, alpha_c, beta_c):

    CK = np.array([0, 0])

    a = a - 1
    T = len(a)
    choiceProb = np.zeros(T)

    # loop over all trial
    for t in range(T):

        # compute choice probabilities
        p = np.exp(beta_c * CK) / np.sum(np.exp(beta_c * CK))

        # compute choice probability for actual choice
        choiceProb[t] = p[a[t]]

        # update choice kernel
        CK = (1 - alpha_c) * CK
        CK[a[t]] += alpha_c * 1

    # compute negative log-likelihood
    NegLL = -np.sum(np.log(choiceProb))
    return NegLL


def lik_M5RWCK_v1(a, r, alpha, beta, alpha_c, beta_c):

    Q = np.array([0.5, 0.5])
    CK = np.array([0, 0])

    a = a - 1
    T = len(a)
    choiceProb = np.zeros(T)

    # loop over all trial
    for t in range(T):

        # compute choice probabilities
        V = beta * Q + beta_c * CK
        p = np.exp(V) / np.sum(np.exp(V))

        # compute choice probability for actual choice
        choiceProb[t] = p[a[t]]

        # update values
        delta = r[t] - Q[a[t]]
        Q[a[t]] += alpha * delta

        # update choice kernel
        CK = (1 - alpha_c) * CK
        CK[a[t]] += alpha_c * 1

    # compute negative log-likelihood
    NegLL = -np.sum(np.log(choiceProb))
    return NegLL


def lik_M6RescorlaWagnerBias_v1(a, r, alpha, beta, Qbias):
    Q = np.array([0.5, 0.5])

    a = a - 1
    T = len(a)
    choiceProb = np.zeros(T)

    # loop over all trial
    for t in range(T):

        # compute choice probabilities
        V = Q
        V[1] += Qbias

        p = np.exp(beta * V) / np.sum(np.exp(beta * V))

        # compute choice probability for actual choice
        choiceProb[t] = p[a[t]]

        # update values
        delta = r[t] - Q[a[t]]
        Q[a[t]] += alpha * delta

    # compute negative log-likelihood
    NegLL = -np.sum(np.log(choiceProb))
    return NegLL


def lik_fullRL_v1(a, r, s, alpha, beta):

    # values for each state
    # Q(a,s) = value of taking action a in state s
    Q = np.zeros((3, 3))

    a = a - 1
    s = s - 1
    T = len(a)
    CP = np.zeros((3, T))
    choiceProb = np.zeros(T)

    for t in range(T):

        # compute choice probabilities
        p = np.exp(beta * Q[:, s[t]])
        p /= np.sum(p)
        CP[:, t] = p

        # compute probability of chosen option
        choiceProb[t] = p[a[t]]

        # update values
        Q[a[t], s[t]] += alpha * (r[t] - Q[a[t], s[t]])

    # compute negative log-likelihood
    NegLL = -np.sum(np.log(choiceProb))

    return NegLL, choiceProb, CP
