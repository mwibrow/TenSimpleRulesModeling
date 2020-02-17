import numpy as np
import numpy.random as random


def choose(p, size=1):
    return random.choice(
        np.arange(len(p), dtype=int),
        p=p,
        size=size) + 1


def simulate_M1random_v1(T, mu, b):
    a = np.zeros(T, dtype=int)
    r = np.zeros(T, dtype=int)

    # compute choice probabilities
    p = np.array([b, 1 - b])

    # make choice according to choice probababilities
    a = choose(p, size=T)

    # generate reward based on choice
    r[:] = random.uniform(size=len(a)) > np.atleast_1d(mu)[a - 1]

    return a, r


def simulate_M2WSLS_v1(T, mu, epsilon):

    mu = np.atleast_1d(mu)
    a = np.zeros(T, dtype=int)
    r = np.zeros(T, dtype=int)

    # last reward/action (initialize as nan)
    rLast = np.NaN
    aLast = np.NaN
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
                p = (1 - epsilon / 2) * np.ones(2)
                p[aLast] = epsilon / 2

        # make choice according to choice probababilities
        a[t] = choose(p)
        i = a[t] - 1

        # generate reward based on choice
        r[t] = random.uniform() < mu[i]

        aLast = i
        rLast = r[t]

    return a, r


def simulate_M3RescorlaWagner_v1(T, mu, alpha, beta):

    a = np.zeros(T, dtype=int)
    r = np.zeros(T, dtype=int)
    Q = np.atleast_1d([0.5, 0.5])

    for t in range(T):

        # compute choice probabilities
        p = np.exp(beta * Q) / np.sum(np.exp(beta * Q))

        # make choice according to choice probababilities
        a[t] = choose(p)
        i = a[t] - 1

        # generate reward based on choice
        r[t] = random.uniform() < mu[i]

        # update values
        delta = r[t] - Q[i]
        Q[i] += alpha * delta

    return a, r


def simulate_M4ChoiceKernel_v1(T, mu, alpha_c, beta_c):
    a = np.zeros(T, dtype=int)
    r = np.zeros(T, dtype=int)
    CK = np.array([0, 0])

    for t in range(T):

        # compute choice probabilities
        p = np.exp(beta_c * CK) / np.sum(np.exp(beta_c * CK))

        # make choice according to choice probababilities
        a[t] = choose(p)
        i = a[t] - 1

        # generate reward based on choice
        r[t] = random.uniform() < mu[i]

        # update choice kernel
        CK = (1 - alpha_c) * CK
        CK[i] = CK[i] + alpha_c * 1

    return a, r


def simulate_M5RWCK_v1(T, mu, alpha, beta, alpha_c, beta_c):

    a = np.zeros(T, dtype=int)
    r = np.zeros(T, dtype=int)
    Q = np.array([0.5, 0.5])
    CK = np.array([0, 0])

    for t in range(T):

        # compute choice probabilities
        V = beta * Q + beta_c * CK
        p = np.exp(V) / np.sum(np.exp(V))

        #  make choice according to choice probababilities
        a[t] = choose(p)
        i = a[t] - 1

        # generate reward based on choice
        r[t] = random.uniform() < mu[i]

        # update values
        delta = r[t] - Q[i]
        Q[i] += alpha * delta

        # update choice kernel
        CK = (1 - alpha_c) * CK
        CK[i] += alpha_c * 1

    return a, r


def simulate_M6RescorlaWagnerBias_v1(T, mu, alpha, beta, Qbias):

    a = np.zeros(T, dtype=int)
    r = np.zeros(T, dtype=int)
    mu = np.atleast_1d(mu)

    Q = np.array([0.5, 0.5])

    for t in range(T):

        # compute choice probabilities
        V = Q[:]
        V[0] += Qbias
        p = np.exp(beta * V) / np.sum(np.exp(beta * V))

        # make choice according to choice probababilities
        a[t] = choose(p)
        i = a[t] - 1

        # generate reward based on choice
        r[t] = random.uniform() < mu[i]

        # update values
        delta = r[t] - Q[i]
        Q[i] += alpha * delta

    return a, r


def simulate_blind_v1(alpha, beta, T):

    AA = np.array(T, dtype=int)
    RR = np.array(T, dtype=int)
    QQ = np.array((3, T), dtype=float)
    SS = np.array(T, dtype=int)

    Q = np.zeros(3)

    for t in range(T):

        s = random.randint(1, 3)

        # compute choice probabilities
        p = np.exp(beta * Q)
        p /= np.sum(p)

        # choose
        a = choose(p)
        i = a - 1

        # determine reward
        if s == 1:
            r = 1 if a == 1 else 0
        elif s == 2:
            r = 1 if a == 1 else 0
        elif s == 3:
            r = 1 if a == 3 else 0

        # update values
        Q[i] += alpha * (r - Q[i])
        QQ[:, t] = Q
        AA[t] = a
        SS[t] = s
        RR[t] = r

    return AA, RR, SS, QQ


def simulate_fullRL_v1(alpha, beta, T):
    a = np.zeros(T, dtype=int)
    r = np.zeros(T, dtype=int)
    s = np.zeros(T, dtype=int)

    # values for each state
    # Q(a,s) = value of taking action a in state s
    Q = np.zeros((3, 3))

    for t in range(T):

        s[t] = random.randint(1, 3)

        # compute choice probabilities
        p = np.exp(beta * Q[:, s[t] - 1])
        p /= np.sum(p)

        # choose
        a[t] = choose(p.T)

        # determine reward
        if s[t] == 1:
            r[t] = a[t] == 1
        elif s[t] == 2:
            r[t] = a[t] == 1
        elif s[t] == 3:
            r[t] = a[t] == 3

        # update values
        i, j = a[t] - 1, s[t] - 1
        Q[i, j] += alpha * (r[t] - Q[i, j])

    return a, r, s


def simulate_validationModel_v1(alpha, beta, T):

    AA = np.array(T, dtype=int)
    RR = np.array(T, dtype=int)
    QQ = np.array((3, T), dtype=float)

    Q = np.zeros(3)
    for t in range(T):

        s = random.randint(1, 2)

        # compute choice probabilities
        p = np.exp(beta * Q)
        p /= np.sum(p)

        # choose
        a = choose(p)

        # determine reward
        if s == 1:
            r = a == 1
        elif s == 2:
            r = a == 1
        elif s == 3:
            r = a == 3

        # update values
        i = a - 1
        Q[i] = Q[i] + alpha * (r - Q[i])
        QQ[:, t] = Q
        AA[t] = a
        RR[t] = r

    return AA, RR, QQ
