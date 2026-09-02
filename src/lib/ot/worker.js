/**
 * Web Worker entry: runs solve() off the main thread so a slow case (a few
 * hundred milliseconds in the degenerate cutoff regime) never freezes a slider.
 *
 * Protocol: { id, opts } in, { id, P, f, g, iterations, converged, ... } out.
 * Typed arrays are transferred, not copied.
 */

import { solve } from './solvers.js';

self.onmessage = (event) => {
  const { id, opts } = event.data;
  try {
    const r = solve(opts);
    self.postMessage(
      {
        id,
        P: r.P, f: r.f, g: r.g,
        n: r.n, m: r.m,
        iterations: r.iterations,
        converged: r.converged,
        residual: r.residual,
        gammaEquivalent: r.gammaEquivalent,
        targetMass: r.targetMass
      },
      [r.P.buffer, r.f.buffer, r.g.buffer]
    );
  } catch (err) {
    self.postMessage({ id, error: String(err && err.message || err) });
  }
};
