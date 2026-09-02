/**
 * Main-thread client for the solver worker, with a "latest request wins" mode
 * for interactive figures.
 *
 *   const solver = createSolver();
 *   const stop = solver.latest('panel-a', opts, (result) => draw(result), { onBusy });
 *
 * A slider firing thirty times a second must not queue thirty solves. `latest`
 * keeps at most one request in flight per key and one waiting; when a solve
 * finishes, the newest waiting request replaces anything older. Results that
 * arrive for a superseded request are dropped, never drawn.
 *
 * Falls back to the synchronous solver when workers are unavailable (file://
 * URLs, old browsers), so the figures still work — just with the occasional
 * pause the worker exists to avoid.
 */

import { solve as solveSync } from './solvers.js';

export function createSolver() {
  let worker = null;
  let seq = 0;
  const pending = new Map();

  try {
    worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = (event) => {
      const msg = event.data;
      const p = pending.get(msg.id);
      if (!p) return;
      pending.delete(msg.id);
      if (msg.error) p.reject(new Error(msg.error));
      else p.resolve(msg);
    };
    worker.onerror = () => {
      // Module workers fail on file:// URLs. Drop to synchronous solving.
      const failed = worker;
      worker = null;
      failed.terminate();
      for (const [, p] of pending) p.reject(new Error('worker unavailable'));
      pending.clear();
    };
  } catch {
    worker = null;
  }

  function solve(opts) {
    if (!worker) {
      return new Promise((resolve, reject) => {
        // Yield first so the UI can paint the "computing" state.
        setTimeout(() => {
          try { resolve(solveSync(opts)); } catch (err) { reject(err); }
        }, 0);
      });
    }
    const id = ++seq;
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
      // Float64Arrays are copied into the worker; the cost matrix is the only
      // large one and it is reused across many calls, so no transfer here.
      worker.postMessage({ id, opts });
    }).catch((err) => {
      if (err.message === 'worker unavailable') return solveSync(opts);
      throw err;
    });
  }

  const lanes = new Map();

  /**
   * Latest-wins request on a named lane. `onResult` runs only for the newest
   * request; `onBusy(true/false)` brackets the time a solve is in flight.
   */
  function latest(key, opts, onResult, { onBusy } = {}) {
    let lane = lanes.get(key);
    if (!lane) {
      lane = { inFlight: false, next: null, gen: 0 };
      lanes.set(key, lane);
    }
    const gen = ++lane.gen;
    lane.next = { opts, onResult, gen };
    if (lane.inFlight) return;

    const run = () => {
      const job = lane.next;
      lane.next = null;
      if (!job) { lane.inFlight = false; onBusy && onBusy(false); return; }
      lane.inFlight = true;
      onBusy && onBusy(true);
      solve(job.opts).then((res) => {
        if (job.gen === lane.gen) job.onResult(res);
        run();
      }).catch((err) => {
        console.error('solve failed', err);
        run();
      });
    };
    run();
  }

  return { solve, latest };
}

/** One shared solver for the whole page. */
let shared = null;
export function sharedSolver() {
  if (!shared) shared = createSolver();
  return shared;
}
