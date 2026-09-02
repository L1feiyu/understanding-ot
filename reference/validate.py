"""Cross-checks between the log-domain forms and the classical multiplicative
implementations, plus limiting-case sanity checks. Also emits golden fixtures
for the JavaScript test suite."""

import json
import numpy as np
from ot_reference import (
    squared_euclidean, sinkhorn,
    entropic_partial_pot, entropic_partial_alternating, partial_ot_log,
    unbalanced_multiplicative, unbalanced_ot_log,
    supervised_ot,
)

rng = np.random.default_rng(0)


def toy(n=6, m=7, shift=1.5):
    X = rng.normal(0, 1, (n, 2))
    Y = rng.normal(0, 1, (m, 2)) + shift
    C = squared_euclidean(X, Y)
    a = rng.uniform(0.5, 1.5, n); a /= a.sum()
    b = rng.uniform(0.5, 1.5, m); b /= b.sum()
    return X, Y, C, a, b


def rel(A, B):
    return np.abs(A - B).max() / max(np.abs(B).max(), 1e-12)


def entropic_objective(P, C, eps):
    """<P,C> + eps*sum(P log P - P) — the quantity these solvers minimise.

    Comparing raw <P,C> instead is a trap: an entropically-regularised solver is
    entitled to a slightly costlier plan if it buys enough entropy, so <P,C>
    alone can rank the wrong plan first.
    """
    Q = np.where(P > 0, P, 1.0)
    return float(np.sum(P * C) + eps * np.sum(P * np.log(Q) - P))


print("=" * 68)
X, Y, C, a, b = toy()
eps = 0.05

# --- 1. balanced Sinkhorn recovers the marginals ---------------------------
P, _, _ = sinkhorn(C, a, b, eps, 2000)
print(f"[balanced ] marginal err  row={np.abs(P.sum(1)-a).max():.2e}  col={np.abs(P.sum(0)-b).max():.2e}")

# --- 2. partial OT: Dykstra's corrections are what make it optimal ----------
# POT's entropic_partial_wasserstein carries correction factors q1/q2/q3. Drop
# them and the iteration stays feasible but lands on a costlier plan, which is
# exactly why the corrections are there.
for s in (0.2, 0.3, 0.45, 0.6, 0.9):
    Pd = entropic_partial_pot(C, a, b, eps, s, 8000)
    Pa = entropic_partial_alternating(C, a, b, eps, s, 8000)
    od, oa = entropic_objective(Pd, C, eps), entropic_objective(Pa, C, eps)
    ok = np.all(Pd.sum(1) <= a + 1e-6) and np.all(Pd.sum(0) <= b + 1e-6)
    print(f"[partial  ] s={s:<4} objective: dykstra={od:.10f}  uncorrected={oa:.10f}"
          f"  mass={Pd.sum():.6f}  P1<=a: {ok}")
    assert od <= oa + 1e-12, f"the corrections made it worse at s={s}"

# --- 3. unbalanced OT: log form == POT-style multiplicative form ------------
for tau in (0.05, 0.5, 5.0):
    Pl, _, _ = unbalanced_ot_log(C, a, b, eps, tau, 3000)
    Pm = unbalanced_multiplicative(C, a, b, eps, tau, 3000)
    print(f"[unbal    ] tau={tau:<5} log-vs-mult rel={rel(Pl,Pm):.2e}  mass={Pl.sum():.6f}")

# --- 4. limiting cases ------------------------------------------------------
Pb, _, _ = sinkhorn(C, a, b, eps, 4000)
Pu, _, _ = unbalanced_ot_log(C, a, b, eps, 1e4, 4000)
print(f"[limit    ] UOT(tau->inf)  vs balanced   rel={rel(Pu,Pb):.2e}")
Ps, _, _ = supervised_ot(C, a, b, eps, 1e3, 4000)
print(f"[limit    ] sOT(gamma->inf) vs balanced  rel={rel(Ps,Pb):.2e}")
Pp = entropic_partial_pot(C, a, b, eps, 1.0, 4000)
print(f"[limit    ] pOT(s=1)        vs balanced  rel={rel(Pp,Pb):.2e}")

# --- 5. sOT behaviour: gamma controls how much mass moves -------------------
print("[sOT      ] gamma sweep (mass transported, should increase with gamma):")
for gamma in (0.05, 0.1, 0.2, 0.5, 1.0, 5.0):
    Ps, f, g = supervised_ot(C, a, b, eps, gamma, 3000)
    print(f"             gamma={gamma:<5} mass={Ps.sum():.6f}  "
          f"|a-P1|_1={np.abs(a-Ps.sum(1)).sum():.4f}  f<=gamma: {f.max() <= gamma + 1e-9}")

# --- 6. sOT (no blocking) == partial OT at the mass gamma selects ------------
# Since P1 <= a, ||a-P1||_1 = |a| - mass(P), so the sOT objective collapses to
#   <P, C - 2*gamma> + const,  i.e. partial OT with Lagrange multiplier 2*gamma.
print("[equiv    ] sOT(gamma) vs partial OT at the same mass (eps=0.002):")
for gamma in (0.3, 0.6, 1.0, 1.5):
    Ps, _, _ = supervised_ot(C, a, b, 0.002, gamma, 40000)
    Pp = entropic_partial_pot(C, a, b, 0.002, Ps.sum(), 40000)
    o1 = (Ps * (C - 2 * gamma)).sum()
    o2 = (Pp * (C - 2 * gamma)).sum()
    print(f"             gamma={gamma:<4} mass={Ps.sum():.5f}  objective gap={o1-o2:+.2e}")

# --- 7. sOT blocking: infinite cost entries forbid couplings ----------------
Cb = C.copy()
Cb[:3, :4] = 1e6          # forbid source cluster 0 -> target cluster 0
Ps, _, _ = supervised_ot(Cb, a, b, eps, 1.0, 3000)
print(f"[sOT block] leaked mass through forbidden block = {Ps[:3,:4].sum():.2e}")

# --------------------------------------------------------------------------- #
# golden fixtures for the JS test suite
# --------------------------------------------------------------------------- #
np.random.seed(1)
Xf = np.round(rng.normal(0, 1, (8, 2)), 6)
Yf = np.round(rng.normal(0, 1, (9, 2)) + 1.2, 6)
Cf = squared_euclidean(Xf, Yf)
af = np.round(rng.uniform(0.5, 1.5, 8), 6); af /= af.sum()
bf = np.round(rng.uniform(0.5, 1.5, 9), 6); bf /= bf.sum()

cases = []
Pf, _, _ = sinkhorn(Cf, af, bf, 0.05, 20000)
cases.append(dict(name="sinkhorn eps=0.05", method="balanced", params=dict(eps=0.05), P=Pf.tolist()))
for s in (0.4, 0.75):
    Pf = entropic_partial_pot(Cf, af, bf, 0.05, s, 20000)
    cases.append(dict(name=f"partial s={s}", method="partial", params=dict(eps=0.05, s=s), P=Pf.tolist()))
for tau in (0.1, 1.0):
    Pf, _, _ = unbalanced_ot_log(Cf, af, bf, 0.05, tau, 20000)
    cases.append(dict(name=f"unbalanced tau={tau}", method="unbalanced", params=dict(eps=0.05, tau=tau), P=Pf.tolist()))
for gamma in (0.1, 0.5):
    Pf, _, _ = supervised_ot(Cf, af, bf, 0.05, gamma, 20000)
    cases.append(dict(name=f"sOT gamma={gamma}", method="supervised", params=dict(eps=0.05, gamma=gamma), P=Pf.tolist()))

with open("fixtures.json", "w") as fh:
    json.dump(dict(X=Xf.tolist(), Y=Yf.tolist(), a=af.tolist(), b=bf.tolist(), cases=cases), fh)
print("=" * 68)
print(f"wrote fixtures.json with {len(cases)} golden cases")
