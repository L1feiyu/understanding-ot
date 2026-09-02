"""Check the balanced / partial / unbalanced references against the REAL POT library.

`ot_reference.py` contains re-implementations written to reproduce POT's
algorithms; it does not import POT. This script closes that gap by running the
actual library and comparing, and it can regenerate the golden fixtures so that
the JavaScript is tested against genuine POT output rather than against a
re-implementation.

    pip install POT
    python3 pot_reference.py                    # compare only
    python3 pot_reference.py --write-fixtures   # and rewrite fixtures.json from POT

Supervised OT is deliberately not in scope here: POT has no sOT, and its
reference is the authors' own `perform_sOT_log`, used verbatim in
`ot_reference.py`.

Version note: POT's unbalanced solver has grown a `reg_type` argument that
selects between an entropy regulariser and a KL-to-the-product one. Those give
different kernels, so this script tries whichever variants the installed version
exposes and reports which one the reference matches, rather than assuming.
"""

import argparse
import inspect
import json
import sys

import numpy as np

try:
    import ot
except ImportError:
    sys.exit(
        "POT is not installed.\n"
        "  pip install POT\n"
        "Then re-run. (The pure-NumPy references in ot_reference.py do not need it.)"
    )

from ot_reference import (
    squared_euclidean, sinkhorn,
    entropic_partial_pot, entropic_partial_alternating,
    unbalanced_multiplicative, unbalanced_ot_log,
    supervised_ot,
)

TOL = 1e-6


def entropic_objective(P, C, eps):
    """<P,C> + eps*sum(P log P - P): the objective these solvers minimise.
    Ranking plans by <P,C> alone can pick the wrong one."""
    Q = np.where(P > 0, P, 1.0)
    return float(np.sum(P * C) + eps * np.sum(P * np.log(Q) - P))


def rel(A, B):
    """Max absolute difference, scaled by the largest entry of the reference."""
    denom = max(float(np.abs(B).max()), 1e-12)
    return float(np.abs(A - B).max()) / denom


def supported(fn, name):
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def call(fn, *args, **kwargs):
    """Drop keyword arguments this version of POT does not accept."""
    try:
        params = inspect.signature(fn).parameters
        kwargs = {k: v for k, v in kwargs.items() if k in params}
    except (TypeError, ValueError):
        pass
    out = fn(*args, **kwargs)
    return out[0] if isinstance(out, tuple) else out


def problem(seed=0, n=8, m=9, shift=1.2):
    rng = np.random.default_rng(seed)
    X = np.round(rng.normal(0, 1, (n, 2)), 6)
    Y = np.round(rng.normal(0, 1, (m, 2)) + shift, 6)
    C = squared_euclidean(X, Y)
    a = np.round(rng.uniform(0.5, 1.5, n), 6); a /= a.sum()
    b = np.round(rng.uniform(0.5, 1.5, m), 6); b /= b.sum()
    return X, Y, np.ascontiguousarray(C), a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-fixtures", action="store_true",
                    help="rewrite fixtures.json using POT as the source of truth")
    args = ap.parse_args()

    print(f"POT {ot.__version__}  ·  numpy {np.__version__}")
    print("=" * 70)

    X, Y, C, a, b = problem()
    eps = 0.05
    failures = []
    cases = []

    def check(label, mine, theirs):
        r = rel(mine, theirs)
        ok = r < TOL
        print(f"[{'ok  ' if ok else 'FAIL'}] {label:<44} rel={r:.2e}")
        if not ok:
            failures.append(label)
        return ok

    # ---- balanced ------------------------------------------------------
    P_pot = call(ot.sinkhorn, a, b, C, eps, numItermax=20000, stopThr=1e-14)
    P_ref, _, _ = sinkhorn(C, a, b, eps, 20000)
    check("ot.sinkhorn", P_ref, P_pot)
    cases.append(dict(name=f"sinkhorn eps={eps}", method="balanced",
                      params=dict(eps=eps), P=np.asarray(P_pot).tolist()))

    # ---- partial -------------------------------------------------------
    # POT runs Dykstra with correction factors q1/q2/q3; the reference is a
    # faithful port of that, and the JS default matches both.
    for s in (0.4, 0.6, 0.75):
        P_pot = call(ot.partial.entropic_partial_wasserstein, a, b, C, eps, m=s,
                     numItermax=20000)
        P_ref = entropic_partial_pot(C, a, b, eps, s, 20000)
        check(f"ot.partial.entropic_partial_wasserstein (m={s})", P_ref, P_pot)
        # And confirm the corrections matter: without them the plan costs more.
        P_alt = entropic_partial_alternating(C, a, b, eps, s, 20000)
        od, oa = entropic_objective(P_ref, C, eps), entropic_objective(P_alt, C, eps)
        verdict = "corrections help" if od < oa - 1e-12 else ("tie" if od <= oa + 1e-12 else "UNEXPECTED")
        print(f"       objective with corrections {od:.10f} vs without {oa:.10f}  ({verdict})")
        cases.append(dict(name=f"partial s={s}", method="partial",
                          params=dict(eps=eps, s=s), P=np.asarray(P_pot).tolist()))

    # ---- unbalanced ----------------------------------------------------
    fn = ot.unbalanced.sinkhorn_unbalanced
    variants = [{}]
    if supported(fn, "reg_type"):
        variants = [{"reg_type": "entropy"}, {"reg_type": "kl"}]

    for tau in (0.1, 1.0):
        P_ref = unbalanced_multiplicative(C, a, b, eps, tau, 20000)
        best, best_r, best_P = None, np.inf, None
        for v in variants:
            P_pot = call(fn, a, b, C, eps, tau, numItermax=20000, stopThr=1e-14, **v)
            r = rel(P_ref, np.asarray(P_pot))
            tag = v.get("reg_type", "default")
            print(f"       ot.unbalanced.sinkhorn_unbalanced (reg_m={tau}, {tag}) rel={r:.2e}")
            if r < best_r:
                best, best_r, best_P = tag, r, np.asarray(P_pot)
        check(f"ot.unbalanced.sinkhorn_unbalanced (reg_m={tau}, best={best})",
              P_ref, best_P)
        cases.append(dict(name=f"unbalanced tau={tau}", method="unbalanced",
                          params=dict(eps=eps, tau=tau), P=best_P.tolist()))

    # ---- supervised: no POT equivalent, use the authors' own routine ----
    for gamma in (0.1, 0.5):
        P_sot, _, _ = supervised_ot(C, a, b, eps, gamma, 20000)
        print(f"[ref ] perform_sOT_log (gamma={gamma})            "
              f"       mass={P_sot.sum():.6f}   [authors' code, no POT equivalent]")
        cases.append(dict(name=f"sOT gamma={gamma}", method="supervised",
                          params=dict(eps=eps, gamma=gamma), P=P_sot.tolist()))

    print("=" * 70)
    if failures:
        print(f"{len(failures)} mismatch(es) against POT {ot.__version__}:")
        for f in failures:
            print("   ", f)
        print("\nThis means ot_reference.py has drifted from the installed POT, or POT\n"
              "changed its algorithm. Investigate before trusting the fixtures.")
    else:
        print(f"All references agree with POT {ot.__version__} to better than {TOL:g}.")

    if args.write_fixtures:
        payload = dict(X=X.tolist(), Y=Y.tolist(), a=a.tolist(), b=b.tolist(), cases=cases)
        with open("fixtures.json", "w") as fh:
            json.dump(payload, fh)
        print(f"\nwrote fixtures.json with {len(cases)} cases "
              f"(balanced/partial/unbalanced straight from POT {ot.__version__})")
        print("Now run `npm test` from the repository root.")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
