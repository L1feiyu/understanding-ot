/**
 * Figure: what supervision actually buys you.
 *
 * Everything up to here has been about *how much* mass moves. Supervised OT's
 * separate trick is saying *which pairs are allowed to exchange it at all* —
 * forbidden couplings enter as infinite entries in the cost matrix.
 *
 * The dataset is built so geometry lies: the nearest target cluster is usually
 * the wrong class. Balanced OT has to satisfy its marginals, so when you forbid
 * the wrong-class pairs it shoves mass through them anyway. sOT can simply
 * decline to move that mass.
 */

import { squaredEuclidean, normalizeCost, applyBlocking, totalMass, longestRoute } from '../lib/ot/solvers.js';
import { sharedSolver } from '../lib/ot/client.js';
import { makeDataset } from '../lib/datasets.js';
import { drawTransport, prepareCanvas, fitScale } from '../lib/plot.js';
import { palette } from '../lib/palette.js';
import {
  el, controlRow, slider, segmented, checkbox, readout, legend,
  scheduler, onResize, figureWidth, fmt, pct, solverStatus
} from '../lib/ui.js';

export function blockingFigure(root) {
  const state = { method: 'supervised', blocked: true, eps: 0.008, tau: 0.25, s: 0.7 };
  const p = palette();
  const solver = sharedSolver();
  let warm = null;

  const canvas = el('canvas', { class: 'block-canvas' });
  const stats = readout([
    { key: 'mass', label: 'mass moved' },
    { key: 'leak', label: 'mass through forbidden pairs', title: 'Should be exactly zero when supervision is on and the method can express it' },
    { key: 'agree', label: 'same-class mass', title: 'Share of transported mass that lands on a target of the same class' },
    { key: 'route', label: 'longest route', title: 'Largest cost among routes carrying ≥ 1% of their source point’s mass' },
    { key: 'conv', label: '' }
  ]);

  const paramSlot = el('div', { class: 'param-slot' });
  const body = el('div', { class: 'fig-body' });

  body.appendChild(controlRow([
    segmented({
      label: 'Method',
      options: [
        { value: 'balanced', label: 'Balanced' },
        { value: 'partial', label: 'Partial' },
        { value: 'unbalanced', label: 'Unbalanced' },
        { value: 'supervised', label: 'Supervised' }
      ],
      value: state.method,
      onChange: (v) => { state.method = v; warm = null; buildParam(); render(); }
    }),
    checkbox({
      label: 'forbid cross-class couplings',
      checked: state.blocked,
      onChange: (v) => { state.blocked = v; warm = null; render(); }
    }),
    paramSlot
  ]));

  body.appendChild(legend([
    { colour: p.source, label: 'source' },
    { colour: p.target, label: 'target' }
  ]));
  body.appendChild(el('p', {
    class: 'fig-hint',
    text: 'Three labelled classes on each side (A, B, C), arranged so the nearest target cluster is usually the wrong one. Turn supervision on and compare methods.'
  }));
  body.appendChild(canvas);
  body.appendChild(stats);
  body.appendChild(el('p', { class: 'fig-note', id: 'blocking-verdict' }));
  root.appendChild(body);

  const verdict = body.querySelector('#blocking-verdict');
  const data = makeDataset('crossedClasses', 45);
  const baseC = normalizeCost(squaredEuclidean(data.X, data.Y));
  const isBlocked = (i, j) => data.labelsX[i] !== data.labelsY[j];

  // Class identity is carried by a direct label at each cluster's centroid,
  // never by an extra hue — three more categorical colours would not clear the
  // colour-blind separation floors in a scatter.
  const NAMES = ['A', 'B', 'C'];
  const annotations = [];
  for (const [pts, labels, side] of [[data.X, data.labelsX, -1], [data.Y, data.labelsY, 1]]) {
    const groups = new Map();
    labels.forEach((k, i) => {
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(pts[i]);
    });
    for (const [k, members] of groups) {
      const cx = members.reduce((t, q) => t + q[0], 0) / members.length;
      const cy = members.reduce((t, q) => t + q[1], 0) / members.length;
      annotations.push({ x: cx + side * 0.22, y: cy, text: NAMES[k] ?? String(k) });
    }
  }

  const PARAMS = {
    partial: { key: 's', label: 'mass fraction s', min: 0.05, max: 1, step: 0.01, log: false },
    unbalanced: { key: 'tau', label: 'KL penalty τ', min: 0.001, max: 5, log: true }
    // Supervised OT has no slider here: gamma is pinned at SOT_GAMMA and the
    // supervision is the checkbox. That is the point of the figure.
  };

  function buildParam() {
    paramSlot.textContent = '';
    const cfg = PARAMS[state.method];
    if (!cfg) {
      paramSlot.appendChild(el('span', {
        class: 'param-none',
        text: state.method === 'supervised' ? 'γ pinned at 2 — supervision is the knob' : 'no parameter'
      }));
      return;
    }
    paramSlot.appendChild(slider({
      label: cfg.label, min: cfg.min, max: cfg.max, step: cfg.step || 0.01, log: !!cfg.log,
      value: state[cfg.key],
      format: (v) => v.toFixed(v < 1 ? 3 : 2),
      onInput: (v) => { state[cfg.key] = v; render(); }
    }));
  }

  function draw(res, C, width, height) {
    const n = data.X.length, m = data.Y.length;
    const ctx = prepareCanvas(canvas, width, height);
    drawTransport(ctx, {
      X: data.X, Y: data.Y, a: data.a, b: data.b, P: res.P,
      width, height, scale: fitScale(width, height, [data.X, data.Y], 26), annotations
    });

    let leak = 0, same = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        const v = res.P[i * m + j];
        if (isBlocked(i, j)) leak += v; else same += v;
      }
    }
    const mass = totalMass(res.P);
    stats.update({
      mass: pct(mass),
      leak: leak < 1e-9 ? '0' : fmt(leak, 4),
      agree: mass > 0 ? pct(same / mass) : '—',
      route: fmt(longestRoute(res.P, baseC, data.a, n, m), 3),
      conv: solverStatus(res)
    });

    verdict.textContent = !state.blocked
      ? 'Supervision is off, so every method is free to follow the geometry — and the geometry is wrong here.'
      : state.method === 'balanced'
        ? 'Balanced OT must hit both marginals, so it pushes mass through pairs you forbade. The ban is unenforceable when the constraints are hard.'
        : state.method === 'partial'
          ? 'Partial OT keeps the ban, because it may leave mass behind — but you have to guess the right mass fraction yourself.'
          : state.method === 'unbalanced'
            ? 'Unbalanced OT keeps the ban too, at the cost of a τ that trades marginal fit against everything else at once.'
            : 'Supervised OT enforces the ban exactly and picks the transported mass itself. This is the case it was designed for.';
  }

  const render = scheduler(() => {
    const width = figureWidth(body, 640);
    const height = Math.round(Math.max(280, Math.min(400, width * 0.72)));
    const n = data.X.length, m = data.Y.length;
    const C = state.blocked ? applyBlocking(baseC, n, m, isBlocked) : baseC;
    solver.latest('blocking', {
      C, a: data.a, b: data.b, method: state.method,
      eps: state.eps, s: state.s, tau: state.tau, warmStart: warm
    }, (res) => { warm = { f: res.f, g: res.g }; draw(res, C, width, height); },
    { onBusy: (busy) => canvas.classList.toggle('is-computing', busy) });
  });

  buildParam();
  onResize(body, render);
  render();
  return { render };
}
