/**
 * Figure: the transport plan as a matrix.
 *
 * The scatter view shows where mass goes; this one shows the shape of the plan
 * itself, and — more to the point — the gap between the marginals you asked for
 * (pale bars) and the ones you got (solid bars). That gap IS the difference
 * between the four methods, and it is invisible in the scatter.
 */

import { squaredEuclidean, normalizeCost, diagnostics, METHOD_META } from '../lib/ot/solvers.js';
import { sharedSolver } from '../lib/ot/client.js';
import { makeDataset } from '../lib/datasets.js';
import { drawCoupling, drawTransport, prepareCanvas, fitScale } from '../lib/plot.js';
import { palette } from '../lib/palette.js';
import {
  el, controlRow, slider, segmented, readout, legend, tooltip,
  scheduler, onResize, figureWidth, fmt, pct, solverStatus
} from '../lib/ui.js';

export function couplingFigure(root) {
  const state = { method: 'unbalanced', eps: 0.008, s: 0.6, tau: 0.3, cutoff: 0.25 };
  const p = palette();
  const solver = sharedSolver();

  const matrix = el('canvas', { class: 'coupling-canvas' });
  const scatter = el('canvas', { class: 'coupling-canvas' });
  const stats = readout([
    { key: 'mass', label: 'mass moved' },
    { key: 'src', label: 'source slack', title: 'ℓ¹ gap between a and P·1' },
    { key: 'tgt', label: 'target slack', title: 'ℓ¹ gap between b and Pᵀ·1' },
    { key: 'route', label: 'longest route', title: 'Largest cost among routes carrying ≥ 1% of their source point’s mass' },
    { key: 'conv', label: '' }
  ]);

  const paramSlot = el('div', { class: 'param-slot' });
  const body = el('div', { class: 'fig-body' });

  const controls = controlRow([
    segmented({
      label: 'Method',
      options: Object.keys(METHOD_META).map((k) => ({ value: k, label: METHOD_META[k].short })),
      value: state.method,
      onChange: (v) => { state.method = v; warm = null; buildParam(); render(); }
    }),
    slider({
      label: 'entropy ε', min: 0.003, max: 0.06, value: state.eps, log: true,
      format: (v) => v.toFixed(3),
      onInput: (v) => { state.eps = v; render(); }
    }),
    paramSlot
  ]);

  body.appendChild(controls);
  body.appendChild(el('p', { class: 'fig-hint', text: 'Pale bar: the marginal you asked for. Solid bar: the marginal the plan delivers. Hover a cell for its mass.' }));
  const panes = el('div', { class: 'coupling-grid' }, [
    el('div', {}, [el('h4', { class: 'sub', text: 'Transport plan P' }), matrix]),
    el('div', {}, [el('h4', { class: 'sub', text: 'The same plan in the plane' }), scatter])
  ]);
  body.appendChild(panes);
  body.appendChild(legend([
    { colour: p.source, label: 'source marginal a' },
    { colour: p.target, label: 'target marginal b' }
  ]));
  body.appendChild(stats);
  root.appendChild(body);

  const tip = tooltip(body);
  const data = makeDataset('imbalance', 36);
  const C = normalizeCost(squaredEuclidean(data.X, data.Y));
  let picker = null;
  let warm = null;

  function buildParam() {
    paramSlot.textContent = '';
    const cfg = METHOD_META[state.method].param;
    if (!cfg) {
      paramSlot.appendChild(el('span', { class: 'param-none', text: 'no parameter — marginals are hard constraints' }));
      return;
    }
    paramSlot.appendChild(slider({
      label: cfg.label, min: cfg.min, max: cfg.max, step: cfg.step, log: !!cfg.log,
      value: state[cfg.key],
      format: (v) => v.toFixed(v < 1 ? 3 : 2),
      onInput: (v) => { state[cfg.key] = v; render(); }
    }));
  }

  function draw(res, size) {
    const mctx = prepareCanvas(matrix, size, size);
    picker = drawCoupling(mctx, {
      P: res.P, n: data.X.length, m: data.Y.length,
      a: data.a, b: data.b, width: size, height: size
    });
    const sctx = prepareCanvas(scatter, size, size);
    drawTransport(sctx, {
      X: data.X, Y: data.Y, a: data.a, b: data.b, P: res.P,
      width: size, height: size, scale: fitScale(size, size, [data.X, data.Y], 20)
    });
    const d = diagnostics(res.P, C, data.a, data.b, data.X.length, data.Y.length);
    stats.update({
      mass: pct(d.massFraction),
      src: fmt(d.sourceViolation, 4),
      tgt: fmt(d.targetViolation, 4),
      route: fmt(d.longestRoute, 3),
      conv: solverStatus(res)
    });
  }

  const render = scheduler(() => {
    const width = figureWidth(body, 940);
    const half = Math.floor((width - 20) / 2);
    const size = Math.max(220, Math.min(380, half));
    solver.latest('coupling', {
      C, a: data.a, b: data.b, method: state.method,
      eps: state.eps, s: state.s, tau: state.tau,
      cutoff: state.method === 'supervised' ? state.cutoff : undefined,
      warmStart: warm
    }, (res) => {
      warm = { f: res.f, g: res.g };
      draw(res, size);
    }, { onBusy: (busy) => panes.classList.toggle('is-computing', busy) });
  });

  matrix.addEventListener('mousemove', (ev) => {
    if (!picker) return;
    const box = matrix.getBoundingClientRect();
    const cell = picker.cellAt(ev.clientX - box.left, ev.clientY - box.top);
    if (!cell) { tip.hide(); return; }
    const parent = body.getBoundingClientRect();
    tip.show(
      `source #${cell.i} → target #${cell.j}<br><strong>${cell.value.toExponential(2)}</strong>`,
      ev.clientX - parent.left, ev.clientY - parent.top
    );
  });
  matrix.addEventListener('mouseleave', () => tip.hide());

  buildParam();
  onResize(body, render);
  render();
  return { render };
}
