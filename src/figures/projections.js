/**
 * Figure: one geometry, four projections.
 *
 * Every entropic problem here is a KL projection of the Gibbs kernel
 * K = exp(-C/eps) onto a constraint set, or its proximal cousin when a
 * constraint is replaced by a penalty. A KL projection onto a marginal
 * constraint is a row (or column) scaling, and the four methods differ only in
 * WHICH scaling they apply. The figure shows the update, and then the primal
 * evidence of it: each source's achieved row sum against the one it asked for.
 */

import { squaredEuclidean, normalizeCost, totalMass, METHOD_META, SOT_GAMMA } from '../lib/ot/solvers.js';
import { sharedSolver } from '../lib/ot/client.js';
import { makeDataset } from '../lib/datasets.js';
import { prepareCanvas } from '../lib/plot.js';
import { palette, withAlpha } from '../lib/palette.js';
import {
  el, controlRow, slider, segmented, legend, scheduler, onResize, figureWidth, pct
} from '../lib/ui.js';

const LINES = {
  balanced:   'u ← a / (K v)                        <span class="lbl">// KL projection onto { P·1 = a }</span>',
  partial:    'u ← u · min(1, a / (P·1))  + Dykstra   <span class="lbl">// KL projection onto { P·1 ≤ a }</span>',
  unbalanced: 'u ← ( a / (K v) )^(τ/(τ+ε))            <span class="lbl">// KL-prox of  τ·KL(P·1 ‖ a)</span>',
  supervised: 'u ← min( a / (K v),  e^(γ/ε) )          <span class="lbl">// KL-prox of  γ·‖a − P·1‖₁,  P·1 ≤ a</span>'
};

const EXPLAIN = {
  balanced: 'The full projection: every row is rescaled to exactly its target. Both constraint sets are affine, so alternating between them converges to the projection onto their intersection with no bookkeeping at all. Every bar reaches its mark.',
  partial: 'Projection onto an inequality: only rows that overshoot are scaled, and only downward. Then the total is rescaled to s. These sets are convex but not affine, so alternating projections alone drift to a feasible point that is not the projection — Dykstra’s corrections are what put it right. Bars sit at or below their marks, and the ones below are the rows that gave mass up.',
  unbalanced: 'Not a projection but a proximal step: the scaling is raised to the power τ/(τ+ε), a geometric interpolation between doing nothing (τ → 0) and the full projection (τ → ∞). Every bar falls short by a factor tied to its own potential, exp(−f/τ). None reaches its mark, and none is stranded.',
  supervised: 'The full projection, capped: a row is rescaled to its target unless doing so would need a factor beyond e^(γ/ε). With γ pinned at 2 that cap is astronomically large — it binds only for rows that have no permitted partner able to take their mass. Those rows are the short bars; every other bar reaches its mark exactly.'
};

export function projectionsFigure(root) {
  const state = { method: 'supervised', eps: 0.008, cutoff: 0.22, tau: 0.25, s: 0.6 };
  const p = palette();
  const solver = sharedSolver();

  const code = el('pre', { class: 'algo' });
  const canvas = el('canvas', { class: 'potentials-canvas' });
  const explain = el('p', { class: 'fig-note' });
  const paramSlot = el('div', { class: 'param-slot' });
  const body = el('div', { class: 'fig-body' });

  body.appendChild(controlRow([
    segmented({
      label: 'Method',
      options: Object.keys(METHOD_META).map((k) => ({ value: k, label: METHOD_META[k].short })),
      value: state.method,
      onChange: (v) => { state.method = v; warm = null; buildParam(); render(); }
    }),
    slider({
      label: 'entropy ε', min: 0.003, max: 0.05, value: state.eps, log: true,
      format: (v) => v.toFixed(3),
      onInput: (v) => { state.eps = v; render(); }
    }),
    paramSlot
  ]));
  body.appendChild(code);
  body.appendChild(el('h4', { class: 'sub', text: 'Each source’s row sum, against the mass it asked for' }));
  body.appendChild(legend([
    { colour: withAlpha(p.source, 0.85), label: 'achieved  (P·1)ᵢ' },
    { colour: p.inkMuted, label: 'requested  aᵢ  (tick)' },
    { colour: p.accent, label: 'row where the projection was cut short' }
  ]));
  body.appendChild(canvas);
  body.appendChild(explain);
  root.appendChild(body);

  const data = makeDataset('extraCluster', 36);
  const C = normalizeCost(squaredEuclidean(data.X, data.Y));
  let warm = null;

  const PARAMS = {
    partial: { key: 's', label: 'mass fraction s', min: 0.05, max: 1, step: 0.01 },
    unbalanced: { key: 'tau', label: 'KL penalty τ', min: 0.001, max: 5, log: true },
    supervised: { key: 'cutoff', label: 'cost cutoff', min: 0.05, max: 1, step: 0.01 }
  };

  function buildParam() {
    paramSlot.textContent = '';
    const cfg = PARAMS[state.method];
    if (!cfg) {
      paramSlot.appendChild(el('span', { class: 'param-none', text: 'no parameter' }));
      return;
    }
    paramSlot.appendChild(slider({
      label: cfg.label, min: cfg.min, max: cfg.max, step: cfg.step || 0.01, log: !!cfg.log,
      value: state[cfg.key],
      format: (v) => v.toFixed(v < 1 ? 3 : 2),
      onInput: (v) => { state[cfg.key] = v; render(); }
    }));
  }

  function draw(res) {
    const n = data.X.length, m = data.Y.length;
    const width = figureWidth(body, 720);
    const height = 200;
    const ctx = prepareCanvas(canvas, width, height);

    // Row sums and the "cut short" test, per method.
    const rows = new Float64Array(n);
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) rows[i] += res.P[i * m + j];
    const cut = new Array(n).fill(false);
    for (let i = 0; i < n; i++) {
      if (state.method === 'supervised') cut[i] = res.f[i] >= SOT_GAMMA - 1e-9;
      else if (state.method === 'partial') cut[i] = rows[i] < data.a[i] * (1 - 1e-6);
      else if (state.method === 'unbalanced') cut[i] = false;
    }

    // Sort by achieved fraction so the picture reads left to right.
    const order = Array.from({ length: n }, (_, i) => i)
      .sort((x, y) => rows[x] / data.a[x] - rows[y] / data.a[y]);

    const padT = 14, padB = 22, padL = 40, padR = 12;
    const maxA = Math.max(...data.a) * 1.12;
    const sy = (v) => height - padB - (v / maxA) * (height - padT - padB);
    const bw = (width - padL - padR) / n;
    const baseY = Math.round(sy(0)) + 0.5;

    ctx.strokeStyle = p.grid;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, baseY); ctx.lineTo(width - padR, baseY); ctx.stroke();
    ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
    ctx.fillStyle = p.inkMuted;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText('0', padL - 6, baseY);
    ctx.fillText('aᵢ', padL - 6, sy(data.a[0]));

    for (let k = 0; k < n; k++) {
      const i = order[k];
      const x = padL + k * bw + 1, w = Math.max(bw - 2, 1);
      const y = sy(rows[i]);
      ctx.fillStyle = cut[i] ? withAlpha(p.accent, 0.88) : withAlpha(p.source, 0.85);
      ctx.fillRect(x, Math.min(y, baseY), w, Math.max(Math.abs(baseY - y), 1));
      // requested mass: a tick across the bar's column
      const ty = Math.round(sy(data.a[i])) + 0.5;
      ctx.strokeStyle = p.ink;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x - 1, ty); ctx.lineTo(x + w + 1, ty); ctx.stroke();
    }

    const nCut = cut.filter(Boolean).length;
    const nShort = Array.from(rows).filter((r, i) => r < data.a[i] * (1 - 1e-4)).length;
    explain.textContent =
      `${EXPLAIN[state.method]} ` +
      (state.method === 'unbalanced'
        ? `${nShort} of ${n} rows fall short. `
        : `${nCut} of ${n} rows were cut short. `) +
      `Mass moved: ${pct(totalMass(res.P))}.`;
  }

  const render = scheduler(() => {
    code.innerHTML =
      'K = exp(−C / ε)                            <span class="lbl">// the Gibbs kernel</span>\n' +
      'repeat:\n' +
      '  P = diag(u) · K · diag(v)\n' +
      `  <mark>${LINES[state.method]}</mark>\n` +
      '  … and the same scaling for v, from the columns';

    solver.latest('projections', {
      C, a: data.a, b: data.b, method: state.method,
      eps: state.eps, s: state.s, tau: state.tau,
      cutoff: state.method === 'supervised' ? state.cutoff : undefined,
      warmStart: warm
    }, (res) => { warm = { f: res.f, g: res.g }; draw(res); },
    { onBusy: (busy) => canvas.classList.toggle('is-computing', busy) });
  });

  buildParam();
  onResize(body, render);
  render();
  return { render };
}
