/**
 * Validates src/lib/ot/solvers.js against golden fixtures produced by the
 * NumPy references in reference/ (which themselves reproduce POT and use the
 * reference `perform_sOT_log` from Cang et al. verbatim).
 *
 *     node --test test/
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  solve, squaredEuclidean, applyBlocking, applyCutoff, normalizeCost, diagnostics,
  rowSums, colSums, totalMass, uniform, SOT_GAMMA, METHOD_META
} from '../src/lib/ot/solvers.js';

const here = dirname(fileURLToPath(import.meta.url));
const fx = JSON.parse(readFileSync(join(here, '..', 'reference', 'fixtures.json'), 'utf8'));

const C = squaredEuclidean(fx.X, fx.Y);
const n = fx.X.length, m = fx.Y.length;
const OPTS = { nIter: 20000, tol: 0 };

function maxAbsDiff(P, ref) {
  let d = 0, mx = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      d = Math.max(d, Math.abs(P[i * m + j] - ref[i][j]));
      mx = Math.max(mx, Math.abs(ref[i][j]));
    }
  }
  return d / Math.max(mx, 1e-12);
}

for (const c of fx.cases) {
  test(`matches NumPy reference — ${c.name}`, () => {
    // Fixtures come from POT-compatible references, so compare against the
    // POT-compatible partial variant; the default partial solver is exact and
    // is checked separately below.
    const algorithm = c.method === 'partial' ? 'dykstra' : undefined;
    const { P } = solve({ C, a: fx.a, b: fx.b, method: c.method, algorithm, ...c.params, ...OPTS });
    const rel = maxAbsDiff(P, c.P);
    assert.ok(rel < 1e-6, `relative error ${rel.toExponential(2)} exceeds 1e-6`);
  });
}

test('balanced OT reproduces both marginals', () => {
  const { P } = solve({ C, a: fx.a, b: fx.b, method: 'balanced', eps: 0.05, ...OPTS });
  const r = rowSums(P, n, m), col = colSums(P, n, m);
  for (let i = 0; i < n; i++) assert.ok(Math.abs(r[i] - fx.a[i]) < 1e-9);
  for (let j = 0; j < m; j++) assert.ok(Math.abs(col[j] - fx.b[j]) < 1e-9);
});

test('partial OT hits the requested mass and never exceeds either marginal', () => {
  for (const s of [0.2, 0.5, 0.85]) {
    const { P } = solve({ C, a: fx.a, b: fx.b, method: 'partial', eps: 0.05, s, ...OPTS });
    assert.ok(Math.abs(totalMass(P) - s) < 1e-7, `mass ${totalMass(P)} != ${s}`);
    const r = rowSums(P, n, m), col = colSums(P, n, m);
    for (let i = 0; i < n; i++) assert.ok(r[i] <= fx.a[i] + 1e-7);
    for (let j = 0; j < m; j++) assert.ok(col[j] <= fx.b[j] + 1e-7);
  }
});

test('unbalanced OT transports more mass as tau grows', () => {
  let prev = -1;
  for (const tau of [0.01, 0.1, 1, 10, 100]) {
    const { P } = solve({ C, a: fx.a, b: fx.b, method: 'unbalanced', eps: 0.05, tau, ...OPTS });
    const mass = totalMass(P);
    assert.ok(mass > prev, `mass did not increase at tau=${tau}`);
    prev = mass;
  }
});

test('unbalanced OT converges to balanced OT as tau grows', () => {
  const bal = solve({ C, a: fx.a, b: fx.b, method: 'balanced', eps: 0.05, ...OPTS });
  const unb = solve({ C, a: fx.a, b: fx.b, method: 'unbalanced', eps: 0.05, tau: 1e5, ...OPTS });
  let d = 0;
  for (let k = 0; k < n * m; k++) d = Math.max(d, Math.abs(bal.P[k] - unb.P[k]));
  assert.ok(d < 1e-4, `max diff ${d}`);
});

test('supervised OT converges to balanced OT as gamma grows', () => {
  const bal = solve({ C, a: fx.a, b: fx.b, method: 'balanced', eps: 0.05, ...OPTS });
  const sot = solve({ C, a: fx.a, b: fx.b, method: 'supervised', eps: 0.05, gamma: 1e3, ...OPTS });
  let d = 0;
  for (let k = 0; k < n * m; k++) d = Math.max(d, Math.abs(bal.P[k] - sot.P[k]));
  assert.ok(d < 1e-9, `max diff ${d}`);
});

test('supervised OT keeps both potentials at or below gamma', () => {
  const gamma = 0.3;
  const { f, g } = solve({ C, a: fx.a, b: fx.b, method: 'supervised', eps: 0.05, gamma, ...OPTS });
  for (const v of f) assert.ok(v <= gamma + 1e-12);
  for (const v of g) assert.ok(v <= gamma + 1e-12);
});

test('supervised OT respects element-wise blocking', () => {
  const blocked = (i, j) => i < 3 && j < 4;
  const Cb = applyBlocking(C, n, m, blocked);
  const { P } = solve({ C: Cb, a: fx.a, b: fx.b, method: 'supervised', eps: 0.05, gamma: 1, ...OPTS });
  let leak = 0;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) leak += P[i * m + j];
  assert.ok(leak < 1e-12, `leaked ${leak} through the forbidden block`);
});

test('supervised OT equals partial OT at the mass gamma selects', () => {
  // Since P1 <= a, ||a-P1||_1 = |a| - mass(P), so sOT = min <P, C - 2*gamma>:
  // the same problem partial OT solves, with the mass level chosen for you.
  const eps = 0.004;
  for (const gamma of [0.3, 0.6, 1.0]) {
    const sot = solve({ C, a: fx.a, b: fx.b, method: 'supervised', eps, gamma, nIter: 60000, tol: 0 });
    const mass = totalMass(sot.P);
    const pot = solve({ C, a: fx.a, b: fx.b, method: 'partial', eps, s: mass, nIter: 60000, tol: 0 });
    let o1 = 0, o2 = 0;
    for (let k = 0; k < n * m; k++) {
      o1 += sot.P[k] * (C[k] - 2 * gamma);
      o2 += pot.P[k] * (C[k] - 2 * gamma);
    }
    assert.ok(Math.abs(o1 - o2) < 1e-3, `gamma=${gamma}: objective gap ${o1 - o2}`);
  }
});

test('diagnostics report marginal violations consistently', () => {
  const { P } = solve({ C, a: fx.a, b: fx.b, method: 'unbalanced', eps: 0.05, tau: 0.2, ...OPTS });
  const d = diagnostics(P, C, fx.a, fx.b, n, m);
  assert.ok(d.sourceViolation > 0 && d.targetViolation > 0);
  assert.ok(Math.abs(d.mass - totalMass(P)) < 1e-12);
});

test('uniform marginals sum to one', () => {
  const u = uniform(7);
  assert.ok(Math.abs(u.reduce((x, y) => x + y, 0) - 1) < 1e-12);
});

test('the exact partial solver is never worse than the POT-compatible one', () => {
  for (const s of [0.3, 0.5, 0.61, 0.8]) {
    const exact = solve({ C, a: fx.a, b: fx.b, method: 'partial', eps: 0.01, s, nIter: 40000, tol: 0 });
    const dyk = solve({ C, a: fx.a, b: fx.b, method: 'partial', algorithm: 'dykstra', eps: 0.01, s, nIter: 40000, tol: 0 });
    // Same mass, so transport cost is directly comparable.
    assert.ok(Math.abs(totalMass(exact.P) - totalMass(dyk.P)) < 1e-6);
    let ce = 0, cd = 0;
    for (let k = 0; k < n * m; k++) { ce += exact.P[k] * C[k]; cd += dyk.P[k] * C[k]; }
    assert.ok(ce <= cd + 1e-9, `s=${s}: exact cost ${ce} > dykstra cost ${cd}`);
  }
});

test('interactive settings solve fast enough for a slider', () => {
  const big = 120;
  const X = Array.from({ length: big }, (_, i) => [Math.cos(i), Math.sin(i)]);
  const Y = Array.from({ length: big }, (_, i) => [2 + Math.cos(i * 1.3), Math.sin(i * 1.3)]);
  const Cb = squaredEuclidean(X, Y);
  const a = uniform(big), b = uniform(big);
  for (const method of ['balanced', 'unbalanced', 'supervised', 'partial']) {
    const t0 = performance.now();
    solve({ C: Cb, a, b, method, eps: 0.05, tau: 0.5, gamma: 0.5, s: 0.6, nIter: 300 });
    const dt = performance.now() - t0;
    assert.ok(dt < 1000, `${method} took ${dt.toFixed(0)}ms`);
  }
});


/* ------------------------------------------------------------------ *
 * Supervised OT as it is actually used: cost normalised to (0,1), the
 * penalty pinned at SOT_GAMMA, and the cutoff on C as the only knob.
 * ------------------------------------------------------------------ */

const Cn = normalizeCost(C);

test('normalised cost lands in (0, 1]', () => {
  let lo = Infinity, hi = -Infinity;
  for (const v of Cn) { if (v < lo) lo = v; if (v > hi) hi = v; }
  assert.ok(lo > 0 && hi <= 1 + 1e-12, `range [${lo}, ${hi}]`);
  assert.ok(Math.abs(hi - 1) < 1e-12, 'max should be exactly 1');
});

test('gamma is saturated at SOT_GAMMA, so it is a constant and not a knob', () => {
  // A route is used when C < 2*gamma. With C <= 1, anything past gamma = 0.5
  // admits every route, so gamma = 2 sits far inside the saturated regime.
  assert.ok(SOT_GAMMA > 0.5, 'SOT_GAMMA must exceed max(C)/2 for C in (0,1)');
  const masses = [0.5, 1, SOT_GAMMA, 10, 100].map((gamma) =>
    totalMass(solve({ C: Cn, a: fx.a, b: fx.b, method: 'supervised', eps: 0.01, gamma, ...OPTS }).P));
  for (const mass of masses) {
    assert.ok(Math.abs(mass - masses[0]) < 1e-9, `mass moved with gamma varies: ${masses}`);
  }
  assert.ok(Math.abs(masses[0] - 1) < 1e-6, 'saturated supervised OT should move all the mass');
});

test('supervised OT defaults gamma to SOT_GAMMA', () => {
  const auto = solve({ C: Cn, a: fx.a, b: fx.b, method: 'supervised', eps: 0.01, cutoff: 0.4, ...OPTS });
  const explicit = solve({ C: Cn, a: fx.a, b: fx.b, method: 'supervised', eps: 0.01, cutoff: 0.4, gamma: SOT_GAMMA, ...OPTS });
  for (let k = 0; k < n * m; k++) assert.ok(Math.abs(auto.P[k] - explicit.P[k]) < 1e-15);
});

/** Quantile of a cost matrix, so the cutoffs under test suit whatever data they run on. */
function quantile(C, p) {
  const sorted = Array.from(C).sort((x, y) => x - y);
  return sorted[Math.floor(p * (sorted.length - 1))];
}

test('the cutoff is the control: mass rises monotonically with it', () => {
  const cutoffs = [0.02, 0.1, 0.25, 0.5, 0.75, 0.95, 1].map((p) => quantile(Cn, p));
  let prev = -1;
  const seen = [];
  for (const cutoff of cutoffs) {
    const mass = totalMass(solve({ C: Cn, a: fx.a, b: fx.b, method: 'supervised', eps: 0.01, cutoff, ...OPTS }).P);
    assert.ok(mass >= prev - 1e-9, `mass fell at cutoff=${cutoff}`);
    prev = mass;
    seen.push(mass);
  }
  assert.ok(seen.at(-1) > 0.99, `the largest cost as cutoff should admit everything, got ${seen.at(-1)}`);
  assert.ok(seen[0] < seen.at(-1) - 0.2, `the cutoff barely changed anything: ${seen}`);
});

test('no mass crosses the cutoff', () => {
  for (const cutoff of [0.15, 0.35, 0.6]) {
    const { P } = solve({ C: Cn, a: fx.a, b: fx.b, method: 'supervised', eps: 0.01, cutoff, ...OPTS });
    let leak = 0;
    for (let k = 0; k < n * m; k++) if (Cn[k] > cutoff) leak += P[k];
    assert.ok(leak < 1e-12, `cutoff=${cutoff} leaked ${leak}`);
  }
});

test('the cutoff option matches pre-applying applyCutoff', () => {
  const viaOption = solve({ C: Cn, a: fx.a, b: fx.b, method: 'supervised', eps: 0.01, cutoff: 0.35, ...OPTS });
  const viaHelper = solve({ C: applyCutoff(Cn, 0.35), a: fx.a, b: fx.b, method: 'supervised', eps: 0.01, ...OPTS });
  for (let k = 0; k < n * m; k++) assert.ok(Math.abs(viaOption.P[k] - viaHelper.P[k]) < 1e-15);
});

test('complementary slackness: the ceiling marks exactly the under-served rows', () => {
  // A capped potential does not mean the row moved nothing — it means the row
  // could not be fully served. The two-way statement is:
  //   f[i] <  gamma  =>  row i is filled exactly (its marginal is met)
  //   f[i] == gamma  =>  row i is strictly short
  for (const cutoff of [0.02, 0.05, 0.12, 0.4]) {
    const { f, P } = solve({ C: Cn, a: fx.a, b: fx.b, method: 'supervised', eps: 0.01, cutoff, ...OPTS });
    const rows = rowSums(P, n, m);
    for (let i = 0; i < n; i++) {
      const filled = rows[i] / fx.a[i];
      if (f[i] >= SOT_GAMMA - 1e-9) {
        assert.ok(filled < 1 - 1e-6, `cutoff=${cutoff}: capped row ${i} is full (${filled})`);
      } else {
        assert.ok(Math.abs(filled - 1) < 1e-6, `cutoff=${cutoff}: uncapped row ${i} is short (${filled})`);
      }
    }
  }
});

test('the supervised control exposed to the figures is the cutoff', () => {
  assert.equal(METHOD_META.supervised.param.key, 'cutoff');
});
