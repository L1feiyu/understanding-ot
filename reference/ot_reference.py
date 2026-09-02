"""
NumPy reference implementations of the four transport problems compared in the
article, plus the exact reference `perform_sOT_log` supplied by the author of
Supervised Optimal Transport (Cang, Zhao et al., SIAM J. Appl. Math. 2023 /
arXiv:2206.13410).

These are the ground truth that `src/lib/ot/*.js` is tested against.

Conventions throughout
---------------------
    C   (n, m)  ground cost matrix
    a   (n,)    source marginal (masses, NOT necessarily normalised)
    b   (m,)    target marginal
    eps         entropic regularisation strength
    P   (n, m)  transport plan

Every solver below is the *same* log-domain Sinkhorn loop

    f_i <- PROJ( eps*log(a_i) - eps*log( sum_j exp((f_i+g_j-C_ij)/eps) ) + f_i )
    g_j <- PROJ( eps*log(b_j) - eps*log( sum_i exp((f_i+g_j-C_ij)/eps) ) + g_j )

differing only in the projection PROJ applied to the dual potentials:

    balanced   Sinkhorn   PROJ(x) = x                     (marginals enforced exactly)
    partial    OT         PROJ(x) = min(x, 0) + rescale   (P1 <= a, total mass = s)
    unbalanced OT (KL)    PROJ(x) = tau/(tau+eps) * x     (soft KL marginal penalty)
    supervised OT (sOT)   PROJ(x) = min(x, gamma)         (soft l1/TV penalty, cap gamma)

That single observation is the spine of the article.
"""

import numpy as np

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

TINY = 1e-300


def squared_euclidean(X, Y):
    """(n,m) matrix of squared euclidean distances between rows of X and Y."""
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    return ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)


def _plan(f, g, C, eps):
    return np.exp((f[:, None] + g[None, :] - C) / eps)


def _row_softmin(f, g, C, eps):
    """eps * log sum_j exp((f_i + g_j - C_ij)/eps), stable."""
    Z = (f[:, None] + g[None, :] - C) / eps
    mx = Z.max(axis=1, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    return eps * (mx[:, 0] + np.log(np.exp(Z - mx).sum(axis=1) + TINY))


def _col_softmin(f, g, C, eps):
    Z = (f[:, None] + g[None, :] - C) / eps
    mx = Z.max(axis=0, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    return eps * (mx[0, :] + np.log(np.exp(Z - mx).sum(axis=0) + TINY))


# --------------------------------------------------------------------------- #
# 1. balanced entropic OT  (Cuturi 2013)
# --------------------------------------------------------------------------- #

def sinkhorn(C, a, b, eps, n_iter=500):
    """min <P,C> + eps*H(P)  s.t.  P1 = a, P^T 1 = b."""
    f = np.zeros(len(a))
    g = np.zeros(len(b))
    for _ in range(n_iter):
        f = eps * np.log(a) - _row_softmin(f, g, C, eps) + f
        g = eps * np.log(b) - _col_softmin(f, g, C, eps) + g
    return _plan(f, g, C, eps), f, g


# --------------------------------------------------------------------------- #
# 2. entropic partial OT  (Benamou/Chizat; matches POT.partial.entropic_partial_wasserstein)
# --------------------------------------------------------------------------- #

def entropic_partial_pot(C, a, b, eps, mass, n_iter=1000):
    """Faithful port of POT's `ot.partial.entropic_partial_wasserstein`.

        min <P,C> + eps*H(P)   s.t.  P1 <= a,  P^T 1 <= b,  <P,1> = mass

    This is Dykstra's algorithm over three convex sets, and the correction
    factors q1/q2/q3 are the whole point of it: without them the iteration is
    plain alternating projection, which converges to a feasible but generally
    suboptimal plan. (An earlier version of this file omitted them and was, for
    a while, wrongly used as evidence that POT itself was suboptimal.)
    """
    dx = np.ones(len(a))
    dy = np.ones(len(b))

    K = np.exp(-C / eps)
    K = K * mass / K.sum()

    q1 = np.ones(K.shape)
    q2 = np.ones(K.shape)
    q3 = np.ones(K.shape)

    for _ in range(n_iter):
        Kprev = K
        K = K * q1
        K1 = np.diag(np.minimum(a / (K.sum(axis=1) + TINY), dx)).dot(K)
        q1 = q1 * Kprev / (K1 + TINY)

        K1prev = K1
        K1 = K1 * q2
        K2 = K1.dot(np.diag(np.minimum(b / (K1.sum(axis=0) + TINY), dy)))
        q2 = q2 * K1prev / (K2 + TINY)

        K2prev = K2
        K2 = K2 * q3
        K = K2 * (mass / (K2.sum() + TINY))
        q3 = q3 * K2prev / (K + TINY)

    return K


def entropic_partial_alternating(C, a, b, eps, mass, n_iter=1000):
    """The same iteration WITHOUT Dykstra's corrections.

    Kept only as the contrast that makes the corrections' role visible: it stays
    feasible (P1 <= a, P^T1 <= b, total mass = `mass`) but lands on a plan of
    strictly higher cost. It is not what POT does.
    """
    K = np.exp(-C / eps)
    K = K * mass / K.sum()
    for _ in range(n_iter):
        K = K * np.minimum(a / (K.sum(axis=1) + TINY), 1.0)[:, None]
        K = K * np.minimum(b / (K.sum(axis=0) + TINY), 1.0)[None, :]
        K = K * (mass / (K.sum() + TINY))
    return K


def partial_ot_log(C, a, b, eps, mass, n_iter=1000):
    """The uncorrected alternating iteration, written in the shared log form.

    Dykstra clips the *update* at zero (potentials may only ever decrease), which
    is what enforces P1 <= a rather than P1 = a; a global scalar shift then pins
    the total mass to `mass`.
    """
    f = np.zeros(len(a))
    g = np.zeros(len(b))
    # start from the mass-scaled Gibbs kernel, as POT does
    tot = np.exp(-C / eps).sum()
    shift = 0.5 * eps * np.log(mass / tot)
    f += shift
    g += shift
    for _ in range(n_iter):
        f = f + np.minimum(eps * np.log(a) - _row_softmin(f, g, C, eps), 0.0)
        g = g + np.minimum(eps * np.log(b) - _col_softmin(f, g, C, eps), 0.0)
        total = _plan(f, g, C, eps).sum()
        corr = 0.5 * eps * np.log(mass / (total + TINY))
        f = f + corr
        g = g + corr
    return _plan(f, g, C, eps), f, g


# --------------------------------------------------------------------------- #
# 3. unbalanced OT, KL relaxation  (Chizat et al. 2018; matches POT.unbalanced.sinkhorn_knopp_unbalanced)
# --------------------------------------------------------------------------- #

def unbalanced_multiplicative(C, a, b, eps, tau, n_iter=1000):
    """POT's scaling iteration, kept as an independent cross-check."""
    fi = tau / (tau + eps)
    K = np.exp(-C / eps)
    u = np.ones(len(a))
    v = np.ones(len(b))
    for _ in range(n_iter):
        u = (a / (K.dot(v) + TINY)) ** fi
        v = (b / (K.T.dot(u) + TINY)) ** fi
    return u[:, None] * K * v[None, :]


def unbalanced_ot_log(C, a, b, eps, tau, n_iter=1000):
    """min <P,C> + eps*H(P) + tau*KL(P1|a) + tau*KL(P^T1|b).  PROJ(x) = tau/(tau+eps) * x."""
    fi = tau / (tau + eps)
    f = np.zeros(len(a))
    g = np.zeros(len(b))
    for _ in range(n_iter):
        f = fi * (eps * np.log(a) - _row_softmin(f, g, C, eps) + f)
        g = fi * (eps * np.log(b) - _col_softmin(f, g, C, eps) + g)
    return _plan(f, g, C, eps), f, g


# --------------------------------------------------------------------------- #
# 4. supervised OT  (Cang, Zhao et al.) -- REFERENCE CODE AS SUPPLIED, UNMODIFIED
# --------------------------------------------------------------------------- #

def perform_sOT_log(G, a, b, eps, options):
    niter = options['niter_sOT']
    # tol   = options['tol_sOT']
    f     = options['f_init']
    g     = options['g_init']
    M     = options['penalty']
    # Err = np.array([[1, 1]])
    for q in range(niter):
        f = np.minimum(eps * np.log(a) - eps * np.log(np.sum(np.exp((f[:, None] + g[None, :]  - G) / eps), axis=1)+ 10**-20) + f, M)
        g = np.minimum(eps * np.log(b) - eps * np.log(np.sum(np.exp((f[:, None] + g[None, :]  - G) / eps), axis=0)+ 10**-20) + g, M)
    P = np.exp((f[:, None] + g[None, :] - G) / eps)

    return P, f, g


def supervised_ot(C, a, b, eps, gamma, n_iter=1000):
    """Thin wrapper over the reference implementation."""
    return perform_sOT_log(C, a, b, eps, {
        'niter_sOT': n_iter,
        'f_init': np.zeros(len(a)),
        'g_init': np.zeros(len(b)),
        'penalty': gamma,
    })
