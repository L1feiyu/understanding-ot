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
 *     partial       min(f_i + corr_i, gamma) with gamma implied by the mass s
 *
 * and symmetrically for g. Everything else is shared. Partial OT collapsing into
 * the supervised update is not a coding shortcut — it is the actual relationship
 * between the two problems, and `gammaEquivalent` on the result reports the
 * gamma that reproduces the plan exactly. The solver itself runs Dykstra, which
 * reaches the same point without having to search for that gamma.
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
    projection: 'min(f + corr, γ(s))',   // equivalent form; solved by Dykstra
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
 * @param {number} [o.tol]     stop when the potentials stop moving (max |Δf|, |Δg|)
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
  if (o.method === 'partial' && o.algorithm !== 'alternating') return solvePartial(o);
  return sinkhornLoop(o);
}

/**
 * Exact entropic partial OT, by Dykstra's algorithm over the three constraint
 * sets { P1 <= a }, { P^T1 <= b } and { <P,1> = s }.
 *
 * This matches POT's `ot.partial.entropic_partial_wasserstein`, which carries
 * the same correction factors, to ~1e-14. Passing `algorithm: 'alternating'`
 * runs the same projections WITHOUT Dykstra's corrections: still feasible, but
 * it settles on a plan with a strictly larger entropic objective. That variant
 * exists only as the contrast — it is not what POT does.
 *
 * The partial problem
 *     min <P,C> + eps*H(P)   s.t.  P1 <= a,  P^T1 <= b,  <P,1> = s
 * has a scalar multiplier lambda for the mass constraint, giving the dual
 * feasible set { f <= 0, g <= 0 } with plan exp((f+g+lambda-C)/eps). Shifting
 * f -> f + lambda/2, g -> g + lambda/2 turns that into { f <= gamma, g <= gamma }
 * with plan exp((f+g-C)/eps) — precisely supervised OT at gamma = lambda/2.
 *
 * So partial OT and sOT are the same problem seen from two sides: one is told
 * the mass and implies the penalty, the other is told the penalty and implies
 * the mass. Dykstra's potentials are only defined up to the gauge f -> f + t,
 * g -> g - t, so the equivalent gamma is the symmetric choice
 *
 *     gammaEquivalent = ( max(f) + max(g) ) / 2
 *
 * which reproduces the plan through the supervised solver to ~1e-15.
 */
function solvePartial(o) {
  const a = Float64Array.from(o.a);
  const b = Float64Array.from(o.b);
  const { C, eps } = o;
  const n = a.length, m = b.length;
  // Dykstra converges more slowly than plain Sinkhorn, so it gets a bigger
  // default budget; it exits early once the projections stop moving.
  const nIter = o.nIter ?? 8000;
  const tol = o.tol ?? 1e-6;

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
  const checkEvery = 10;
  const Pprev = new Float64Array(n * m);
  let havePrev = false;
  let iterations = 0, converged = false, residualNow = Infinity;

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

    // Convergence: feasible (P1 <= a, P^T1 <= b, mass = s) AND the plan has
    // stopped moving between checks. Feasibility alone is not enough — the
    // uncorrected iteration is feasible too — and plan-change alone can fire
    // on a plateau. (Dykstra's corrections R, S, T never vanish; they converge
    // to the constraint multipliers, so watching them would never stop.)
    if (it % checkEvery === checkEvery - 1) {
      let dP = 0, over = 0, total = 0;
      const rsum = new Float64Array(n), csum = new Float64Array(m);
      for (let i = 0; i < n; i++) {
        const fi = F[i], base = i * m;
        for (let j = 0; j < m; j++) {
          const v = Math.exp((fi + G[j] - C[base + j]) / eps);
          const d = Math.abs(v - Pprev[base + j]);
          if (d > dP) dP = d;
          Pprev[base + j] = v;
          rsum[i] += v; csum[j] += v; total += v;
        }
      }
      for (let i = 0; i < n; i++) if (rsum[i] - a[i] > over) over = rsum[i] - a[i];
      for (let j = 0; j < m; j++) if (csum[j] - b[j] > over) over = csum[j] - b[j];
      const infeas = Math.max(over, Math.abs(total - target));
      residualNow = Math.max(infeas, havePrev ? dP : Infinity);
      if (havePrev && dP < tol && infeas < tol) { converged = true; break; }
      havePrev = true;
    }
  }

  const P = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) P[i * m + j] = Math.exp((F[i] + G[j] - C[i * m + j]) / eps);
  }

  // The supervised cap that reproduces this plan (see the note above).
  let maxF = -Infinity, maxG = -Infinity;
  for (const v of F) if (v > maxF) maxF = v;
  for (const v of G) if (v > maxG) maxG = v;

  return {
    P, f: F, g: G, n, m, iterations, converged,
    residual: residualNow,
    targetMass: target,
    gammaEquivalent: (maxF + maxG) / 2
  };
}

function sinkhornLoop(o) {
  const { C, eps, method } = o;
  const a = Float64Array.from(o.a);
  const b = Float64Array.from(o.b);
  const n = a.length, m = b.length;
  // Sinkhorn needs O(1/eps) iterations; the early exit makes a large budget
  // cheap when it is not needed and correct when it is.
  const nIter = o.nIter ?? 20000;
  // On the KKT residual. Marginals are O(1/n), so 1e-6 is a relative error of
  // well under 1e-4 on any figure here; tighten it for numerical work.
  const tol = o.tol ?? 1e-6;

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
  // Over-relaxation (Thibault, Chizat, Dossal, Papadakis 2017): step past the
  // plain Sinkhorn update by a factor omega in (1, 2). Same fixed point, about
  // half the iterations. Applied where the update is a pure scaling (balanced,
  // supervised); the shrink and Dykstra paths keep 1. Pass omega: 1 to
  // reproduce textbook Sinkhorn step for step.
  const omega = o.omega ?? 1.7;

  // Warm-started potentials must still respect the cap they are reused under.
  if (method === 'supervised' && o.warmStart) {
    for (let i = 0; i < n; i++) if (f[i] > gamma) f[i] = gamma;
    for (let j = 0; j < m; j++) if (g[j] > gamma) g[j] = gamma;
  }

  // The uncorrected alternating variant starts from the mass-scaled Gibbs kernel.
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

  // Convergence is judged on the KKT residual of each method's fixed point,
  // checked every few iterations. Neither "potentials stopped moving" nor
  // "plan stopped moving" is safe here: with a cap in play, Sinkhorn converges
  // GEOMETRICALLY to a false plateau, then a potential crosses the cap and the
  // plan jumps to a different regime. The residual stays large on the plateau
  // and is the only test that tells the two apart.
  //
  //   balanced     P1 = a                        exactly
  //   supervised   f < γ  =>  P1 = a ;   f = γ  =>  P1 <= a
  //   unbalanced   P1 = a · exp(−f/τ)             (KL optimality)
  //   partial*     P1 <= a,  <P,1> = s            (*alternating variant only)
  //
  // and the same for columns.
  const checkEvery = 10;
  const rs = new Float64Array(n), cs = new Float64Array(m);
  let residualNow = Infinity;

  for (let it = 0; it < nIter; it++) {
    iterations = it + 1;

    // ---- row half-step -------------------------------------------------
    rowLogSumExp(f, g, C, eps, n, m, rowLse);
    for (let i = 0; i < n; i++) {
      const corr = eps * logA[i] - rowLse[i];
      f[i] = project(f[i], corr, method, shrink, gamma, omega);
    }

    // ---- column half-step ----------------------------------------------
    colLogSumExp(f, g, C, eps, n, m, colLse);
    for (let j = 0; j < m; j++) {
      const corr = eps * logB[j] - colLse[j];
      g[j] = project(g[j], corr, method, shrink, gamma, omega);
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

    // ---- convergence: KKT residual ------------------------------------
    if (it % checkEvery === checkEvery - 1) {
      rs.fill(0); cs.fill(0);
      let total = 0;
      for (let i = 0; i < n; i++) {
        const fi = f[i], base = i * m;
        for (let j = 0; j < m; j++) {
          const v = Math.exp((fi + g[j] - C[base + j]) / eps);
          rs[i] += v; cs[j] += v; total += v;
        }
      }
      let resid = 0;
      for (let i = 0; i < n; i++) {
        const r = residual(method, rs[i], a[i], f[i], gamma, tau);
        if (r > resid) resid = r;
      }
      for (let j = 0; j < m; j++) {
        const r = residual(method, cs[j], b[j], g[j], gamma, tau);
        if (r > resid) resid = r;
      }
      if (method === 'partial') resid = Math.max(resid, Math.abs(total - s));
      residualNow = resid;
      if (resid < tol) { converged = true; break; }
    }
  }

  const P = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) P[i * m + j] = Math.exp((f[i] + g[j] - C[i * m + j]) / eps);
  }
  return { P, f, g, n, m, iterations, converged, residual: residualNow };
}

/** How far one marginal is from its method's fixed-point condition. */
function residual(method, sum, target, pot, gamma, tau) {
  switch (method) {
    case 'balanced':   return Math.abs(sum - target);
    case 'supervised': return pot >= gamma - 1e-12 ? Math.max(0, sum - target) : Math.abs(sum - target);
    case 'unbalanced': return Math.abs(sum - target * Math.exp(-pot / tau));
    case 'partial':    return Math.max(0, sum - target);
  }
}

function project(cur, corr, method, shrink, gamma, omega) {
  switch (method) {
    case 'balanced':   return cur + omega * corr;
    case 'partial':    return cur + Math.min(corr, 0);
    case 'unbalanced': return shrink * (cur + corr);
    case 'supervised': return Math.min(cur + omega * corr, gamma);
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

/**
 * The most expensive route the plan actually uses: the largest C[i][j] among
 * pairs carrying at least `share` of their source point's mass. Entropic plans
 * put a vanishing trace of mass on every permitted pair, so a raw max would
 * only report noise; the share threshold asks for routes that matter.
 *
 * This is the quantity supervised OT controls and the others cannot. With a
 * cutoff in force it is bounded by the cutoff by construction; partial OT at
 * the same transported mass is free to exceed it.
 */
export function longestRoute(P, C, a, n, m, share = 0.01) {
  let longest = 0;
  for (let i = 0; i < n; i++) {
    const floor = share * a[i];
    for (let j = 0; j < m; j++) {
      const k = i * m + j;
      if (P[k] >= floor && C[k] < 1e5 && C[k] > longest) longest = C[k];
    }
  }
  return longest;
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
    longestRoute: longestRoute(P, C, a, n, m),
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
