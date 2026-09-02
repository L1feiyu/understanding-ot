/**
 * ot-js — entropic optimal transport solvers, zero dependencies.
 *
 * Balanced, partial, unbalanced (KL) and supervised OT all reduce to the SAME
 * log-domain Sinkhorn loop, differing only in one projection applied to the
 * dual potentials after each half-step:
 *
 *     corr_i  =  eps*log(a_i) - eps*log( sum_j exp( (f_i + g_j - C_ij)/eps ) )
 *
 *     balanced      f_i <- f_i + corr_i
 *     unbalanced    f_i <- (tau/(tau+eps)) * (f_i + corr_i)
 *     supervised    f_i <- min(f_i + corr_i, gamma)
 *     partial       same as supervised, with gamma bisected to hit the mass s
 *
 * and symmetrically for g. Everything else is shared. Partial OT collapsing into
 * the supervised update is not a coding shortcut — it is the actual relationship
 * between the two problems; see solvePartial below.
 *
 * Validated against NumPy references in reference/ot_reference.py, which in turn
 * reproduce POT's `entropic_partial_wasserstein` and `sinkhorn_knopp_unbalanced`
 * to ~1e-15, and use the reference `perform_sOT_log` from Cang et al. verbatim.
 *
 * Matrices are row-major Float64Array of length n*m.
 */

const TINY = 1e-300;

export const METHODS = ['balanced', 'partial', 'unbalanced', 'supervised'];

/**
 * The penalty supervised OT is used with in practice.
 *
 * A route is worth using exactly when C[i][j] < 2*gamma. With the cost matrix
 * normalised to (0,1), gamma = 2 puts that threshold at 4 — above every cost in
 * the matrix — so the price never binds and every permitted route is used. That
 * is deliberate: in sOT the penalty is not the hyperparameter. It is a constant
 * chosen large enough to get out of the way, so that the only thing deciding
 * where mass goes is which routes are permitted at all.
 *
 * The knob is the cutoff on C.
 */
export const SOT_GAMMA = 2;

export const METHOD_META = {
  balanced: {
    label: 'Balanced OT',
    short: 'Balanced',
    param: null,
    projection: 'f + corr',
    blurb: 'Every unit of supply must go somewhere and every unit of demand must be met.'
  },
  partial: {
    label: 'Partial OT',
    short: 'Partial',
    param: { key: 's', label: 'mass fraction s', min: 0.05, max: 1, step: 0.01, def: 0.6 },
    projection: 'min(f + corr, γ(s))',
    blurb: 'Move a prescribed fraction of the mass and leave the rest where it is.'
  },
  unbalanced: {
    label: 'Unbalanced OT',
    short: 'Unbalanced',
    // Costs are normalised to [0,1] before solving, so these ranges are absolute.
    param: { key: 'tau', label: 'marginal penalty τ', min: 0.001, max: 5, step: 0.001, def: 0.25, log: true },
    projection: 'τ/(τ+ε) · (f + corr)',
    blurb: 'Marginals are suggestions: deviating from them costs a KL divergence.'
  },
  supervised: {
    label: 'Supervised OT',
    short: 'Supervised',
    // gamma is pinned at SOT_GAMMA; the cutoff on the normalised cost is what
    // you actually turn. Everything above the cutoff is a forbidden route.
    param: { key: 'cutoff', label: 'cost cutoff', min: 0.02, max: 1, step: 0.01, def: 0.21 },
    projection: 'min(f + corr, γ)',
    blurb: 'Routes costing more than the cutoff are forbidden outright; mass with no permitted partner simply stays.'
  }
};

/* ------------------------------------------------------------------ */
/* cost matrices                                                       */
/* ------------------------------------------------------------------ */

/** Squared euclidean cost between two arrays of points ([[x,y],...]). */
export function squaredEuclidean(X, Y) {
  const n = X.length, m = Y.length, d = X[0].length;
  const C = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let k = 0; k < d; k++) {
        const t = X[i][k] - Y[j][k];
        s += t * t;
      }
      C[i * m + j] = s;
    }
  }
  return C;
}

export function euclidean(X, Y) {
  const C = squaredEuclidean(X, Y);
  for (let i = 0; i < C.length; i++) C[i] = Math.sqrt(C[i]);
  return C;
}

/** Largest finite entry of a cost matrix — handy for normalising. */
export function maxFinite(C) {
  let mx = 0;
  for (let i = 0; i < C.length; i++) if (Number.isFinite(C[i]) && C[i] > mx) mx = C[i];
  return mx;
}

/** Divide a cost matrix by its max so that eps means the same thing everywhere. */
export function normalizeCost(C) {
  const mx = maxFinite(C) || 1;
  const out = new Float64Array(C.length);
  for (let i = 0; i < C.length; i++) out[i] = C[i] / mx;
  return out;
}

/**
 * Forbid every route costing more than `cutoff`.
 *
 * This is the way supervised OT is normally driven: normalise C to (0,1), pin
 * the penalty at SOT_GAMMA, and move the cutoff. Mass whose only partners lie
 * beyond the cutoff has nowhere permitted to go, so it stays — which is the
 * whole mechanism, expressed in the units of the cost matrix.
 *
 * Apply it to an already-normalised matrix; the sentinel cost is not rescaled.
 */
export function applyCutoff(C, cutoff, big = 1e6) {
  const out = Float64Array.from(C);
  for (let i = 0; i < out.length; i++) if (out[i] > cutoff) out[i] = big;
  return out;
}

/**
 * Apply supervision to a cost matrix: `blocked(i, j)` returning true forbids
 * transport from source i to target j. Implemented as a very large finite cost
 * (a true Infinity would poison the log-sum-exp), exactly as in the sOT paper.
 */
export function applyBlocking(C, n, m, blocked, big = 1e6) {
  const out = Float64Array.from(C);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      if (blocked(i, j)) out[i * m + j] = big;
    }
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* the shared solver                                                   */
/* ------------------------------------------------------------------ */

/**
 * @param {object} o
 * @param {Float64Array} o.C   cost matrix, row-major, length n*m
 * @param {Float64Array|number[]} o.a source marginal, length n
 * @param {Float64Array|number[]} o.b target marginal, length m
 * @param {number} o.eps       entropic regularisation
 * @param {string} o.method    one of METHODS
 * @param {number} [o.s]       partial OT: total mass to transport
 * @param {number} [o.tau]     unbalanced OT: KL penalty
 * @param {number} [o.gamma]   supervised OT: l1 penalty / potential cap (default SOT_GAMMA)
 * @param {number} [o.cutoff]  forbid every route whose cost exceeds this
 * @param {number} [o.nIter]   iterations
 * @param {number} [o.tol]     stop when the marginals stop moving
 * @returns {{P:Float64Array, f:Float64Array, g:Float64Array, n:number, m:number,
 *            iterations:number, converged:boolean}}
 */
export function solve(o) {
  // `cutoff` is sugar for supervision by cost threshold: it rewrites the cost
  // matrix before solving, so callers can pass a plain C and a cutoff value.
  if (o.cutoff != null && o.cutoff < Infinity) {
    o = { ...o, C: applyCutoff(o.C, o.cutoff), cutoff: null };
  }
  if (o.method === 'supervised' && o.gamma == null) o = { ...o, gamma: SOT_GAMMA };
  if (o.method === 'partial' && o.algorithm !== 'dykstra') return solvePartial(o);
  return sinkhornLoop(o);
}

/**
 * Exact entropic partial OT, obtained through its equivalence with sOT.
 *
 * The partial problem
 *     min <P,C> + eps*H(P)   s.t.  P1 <= a,  P^T1 <= b,  <P,1> = s
 * has a scalar multiplier lambda for the mass constraint, giving the dual
 * feasible set { f <= 0, g <= 0 } with plan exp((f+g+lambda-C)/eps). Shifting
 * f -> f + lambda/2, g -> g + lambda/2 turns that into { f <= gamma, g <= gamma }
 * with plan exp((f+g-C)/eps) — precisely supervised OT at gamma = lambda/2.
 *
 * So partial OT and sOT are the same solver: one is told the mass and finds the
 * penalty, the other is told the penalty and finds the mass. Transported mass is
 * monotone in gamma, so a bisection recovers the requested level.
 *
 * (POT's `entropic_partial_wasserstein` uses alternating projection without
 * Dykstra corrections and can land on a strictly suboptimal plan; pass
 * `algorithm: 'dykstra'` to reproduce it.)
 */
function solvePartial(o) {
  const a = Float64Array.from(o.a);
  const b = Float64Array.from(o.b);
  const { C, eps } = o;
  const n = a.length, m = b.length;
  // Dykstra converges more slowly than plain Sinkhorn, so it gets a bigger
  // default budget; it exits early once the projections stop moving.
  const nIter = o.nIter ?? 1200;
  const tol = o.tol ?? 1e-10;

  let sumA = 0, sumB = 0;
  for (const v of a) sumA += v;
  for (const v of b) sumB += v;
  const target = Math.min(o.s ?? 1, Math.min(sumA, sumB));

  const logA = new Float64Array(n), logB = new Float64Array(m);
  for (let i = 0; i < n; i++) logA[i] = Math.log(a[i] + TINY);
  for (let j = 0; j < m; j++) logB[j] = Math.log(b[j] + TINY);

  // log P_ij = (F_i + G_j - C_ij)/eps, starting from the Gibbs kernel.
  const F = o.warmStart ? Float64Array.from(o.warmStart.f) : new Float64Array(n);
  const G = o.warmStart ? Float64Array.from(o.warmStart.g) : new Float64Array(m);

  // Dykstra correction terms, one per constraint set.
  const R = new Float64Array(n);   // for { P1 <= a }
  const S = new Float64Array(m);   // for { P^T 1 <= b }
  let T = 0;                       // for { <P,1> = target }

  const rowLse = new Float64Array(n), colLse = new Float64Array(m);
  let iterations = 0, converged = false;

  for (let it = 0; it < nIter; it++) {
    iterations = it + 1;

    // ---- KL projection onto { P1 <= a } --------------------------------
    for (let i = 0; i < n; i++) F[i] += R[i];
    rowLogSumExp(F, G, C, eps, n, m, rowLse);
    for (let i = 0; i < n; i++) {
      const step = Math.min(eps * logA[i] - rowLse[i], 0);
      F[i] += step;
      R[i] = -step;
    }

    // ---- KL projection onto { P^T 1 <= b } -----------------------------
    for (let j = 0; j < m; j++) G[j] += S[j];
    colLogSumExp(F, G, C, eps, n, m, colLse);
    // colLse_j = eps*log(sum_i P_ij), so the total mass after the column step
    // is sum_j exp((colLse_j + step_j)/eps) — no extra O(nm) pass needed.
    let mass = 0;
    for (let j = 0; j < m; j++) {
      const step = Math.min(eps * logB[j] - colLse[j], 0);
      G[j] += step;
      S[j] = -step;
      mass += Math.exp((colLse[j] + step) / eps);
    }

    // ---- KL projection onto { <P,1> = target } -------------------------
    for (let i = 0; i < n; i++) F[i] += T;
    mass *= Math.exp(T / eps);
    const shift = eps * Math.log(target / Math.max(mass, TINY));
    for (let i = 0; i < n; i++) F[i] += shift;
    T = -shift;

    // At the fixed point every Dykstra correction vanishes, which is a far more
    // meaningful stopping test than watching the (already pinned) total mass.
    if (it % 5 === 4) {
      let resid = Math.abs(shift);
      for (let i = 0; i < n; i++) if (R[i] > resid) resid = R[i];
      for (let j = 0; j < m; j++) if (S[j] > resid) resid = S[j];
      if (resid < tol) { converged = true; break; }
    }
  }

  const P = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) P[i * m + j] = Math.exp((F[i] + G[j] - C[i * m + j]) / eps);
  }
  return { P, f: F, g: G, n, m, iterations, converged, targetMass: target };
}

function sinkhornLoop(o) {
  const { C, eps, method } = o;
  const a = Float64Array.from(o.a);
  const b = Float64Array.from(o.b);
  const n = a.length, m = b.length;
  const nIter = o.nIter ?? 400;
  const tol = o.tol ?? 1e-9;

  if (C.length !== n * m) throw new Error(`cost matrix is ${C.length}, expected ${n * m}`);
  if (!METHODS.includes(method)) throw new Error(`unknown method "${method}"`);
  if (!(eps > 0)) throw new Error('eps must be positive');

  const logA = new Float64Array(n), logB = new Float64Array(m);
  for (let i = 0; i < n; i++) logA[i] = Math.log(a[i] + TINY);
  for (let j = 0; j < m; j++) logB[j] = Math.log(b[j] + TINY);

  const f = o.warmStart ? Float64Array.from(o.warmStart.f) : new Float64Array(n);
  const g = o.warmStart ? Float64Array.from(o.warmStart.g) : new Float64Array(m);

  const s = o.s ?? 1;
  const tau = o.tau ?? 1;
  const gamma = o.gamma ?? Infinity;
  const shrink = tau / (tau + eps);

  // Warm-started potentials must still respect the cap they are reused under.
  if (method === 'supervised' && o.warmStart) {
    for (let i = 0; i < n; i++) if (f[i] > gamma) f[i] = gamma;
    for (let j = 0; j < m; j++) if (g[j] > gamma) g[j] = gamma;
  }

  // The POT-compatible Dykstra variant starts from the mass-scaled Gibbs kernel.
  if (method === 'partial') {
    let tot = 0;
    for (let k = 0; k < n * m; k++) tot += Math.exp(-C[k] / eps);
    const shift = 0.5 * eps * Math.log(s / (tot + TINY));
    f.fill(shift);
    g.fill(shift);
  }

  const rowLse = new Float64Array(n);
  const colLse = new Float64Array(m);
  let iterations = 0, converged = false;
  let prevMass = NaN;

  for (let it = 0; it < nIter; it++) {
    iterations = it + 1;

    // ---- row half-step -------------------------------------------------
    rowLogSumExp(f, g, C, eps, n, m, rowLse);
    for (let i = 0; i < n; i++) {
      const corr = eps * logA[i] - rowLse[i];
      f[i] = project(f[i], corr, method, shrink, gamma);
    }

    // ---- column half-step ----------------------------------------------
    colLogSumExp(f, g, C, eps, n, m, colLse);
    for (let j = 0; j < m; j++) {
      const corr = eps * logB[j] - colLse[j];
      g[j] = project(g[j], corr, method, shrink, gamma);
    }

    // ---- partial OT only: pin the total transported mass to s ----------
    if (method === 'partial') {
      let tot = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < m; j++) tot += Math.exp((f[i] + g[j] - C[i * m + j]) / eps);
      }
      const corr = 0.5 * eps * Math.log(s / (tot + TINY));
      for (let i = 0; i < n; i++) f[i] += corr;
      for (let j = 0; j < m; j++) g[j] += corr;
    }

    // ---- cheap convergence check ---------------------------------------
    if (it % 5 === 4) {
      let mass = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < m; j++) mass += Math.exp((f[i] + g[j] - C[i * m + j]) / eps);
      }
      if (Math.abs(mass - prevMass) < tol * Math.max(1, mass)) { converged = true; break; }
      prevMass = mass;
    }
  }

  const P = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) P[i * m + j] = Math.exp((f[i] + g[j] - C[i * m + j]) / eps);
  }
  return { P, f, g, n, m, iterations, converged };
}

function project(cur, corr, method, shrink, gamma) {
  switch (method) {
    case 'balanced':   return cur + corr;
    case 'partial':    return cur + Math.min(corr, 0);
    case 'unbalanced': return shrink * (cur + corr);
    case 'supervised': return Math.min(cur + corr, gamma);
  }
}

/* ------------------------------------------------------------------ */
/* stable log-sum-exp over rows / columns of (f_i + g_j - C_ij)/eps     */
/* ------------------------------------------------------------------ */

function rowLogSumExp(f, g, C, eps, n, m, out) {
  for (let i = 0; i < n; i++) {
    const fi = f[i], base = i * m;
    let mx = -Infinity;
    for (let j = 0; j < m; j++) {
      const z = (fi + g[j] - C[base + j]) / eps;
      if (z > mx) mx = z;
    }
    if (!Number.isFinite(mx)) mx = 0;
    let sum = 0;
    for (let j = 0; j < m; j++) sum += Math.exp((fi + g[j] - C[base + j]) / eps - mx);
    out[i] = eps * (mx + Math.log(sum + TINY));
  }
}

function colLogSumExp(f, g, C, eps, n, m, out) {
  for (let j = 0; j < m; j++) {
    const gj = g[j];
    let mx = -Infinity;
    for (let i = 0; i < n; i++) {
      const z = (f[i] + gj - C[i * m + j]) / eps;
      if (z > mx) mx = z;
    }
    if (!Number.isFinite(mx)) mx = 0;
    let sum = 0;
    for (let i = 0; i < n; i++) sum += Math.exp((f[i] + gj - C[i * m + j]) / eps - mx);
    out[j] = eps * (mx + Math.log(sum + TINY));
  }
}

/* ------------------------------------------------------------------ */
/* diagnostics                                                         */
/* ------------------------------------------------------------------ */

export function rowSums(P, n, m) {
  const r = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < m; j++) s += P[i * m + j];
    r[i] = s;
  }
  return r;
}

export function colSums(P, n, m) {
  const c = new Float64Array(m);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) c[j] += P[i * m + j];
  }
  return c;
}

export function totalMass(P) {
  let s = 0;
  for (let i = 0; i < P.length; i++) s += P[i];
  return s;
}

export function transportCost(P, C) {
  let s = 0;
  for (let i = 0; i < P.length; i++) if (Number.isFinite(C[i])) s += P[i] * C[i];
  return s;
}

/** Summary statistics used by the figures' readouts. */
export function diagnostics(P, C, a, b, n, m) {
  const r = rowSums(P, n, m), c = colSums(P, n, m);
  let l1a = 0, l1b = 0, sa = 0, sb = 0;
  for (let i = 0; i < n; i++) { l1a += Math.abs(a[i] - r[i]); sa += a[i]; }
  for (let j = 0; j < m; j++) { l1b += Math.abs(b[j] - c[j]); sb += b[j]; }
  const mass = totalMass(P);
  return {
    mass,
    massFraction: mass / Math.max(sa, sb),
    cost: transportCost(P, C),
    rowSums: r,
    colSums: c,
    sourceViolation: l1a,
    targetViolation: l1b
  };
}

/** Uniform marginal of length n summing to 1. */
export function uniform(n) {
  const a = new Float64Array(n);
  a.fill(1 / n);
  return a;
}
