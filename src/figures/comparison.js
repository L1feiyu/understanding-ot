/**
 * Figure: the four problems side by side on the same data, with the same eps.
 *
 * Small multiples rather than four colours on one plot — method identity is
 * carried by the panel title, leaving the two categorical hues free to mean
 * what they mean everywhere else in the article (source, target).
 */

import { solve, squaredEuclidean, normalizeCost, diagnostics, METHOD_META } from '../lib/ot/solvers.js';
import { makeDataset, DATASET_ORDER } from '../lib/datasets.js';
import { drawTransport, prepareCanvas, fitScale, pickPoint } from '../lib/plot.js';
import { palette } from '../lib/palette.js';
import {
  el, controlRow, slider, segmented, readout, legend, tooltip,
  scheduler, onResize, figureWidth, fmt, pct
} from '../lib/ui.js';

const METHODS = ['balanced', 'partial', 'unbalanced', 'supervised'];

export function comparisonFigure(root) {
  const state = {
    dataset: 'extraCluster',
    eps: 0.006,
    s: 0.6,
    tau: 0.25,
    cutoff: 0.21,
    focus: null
  };

  const p = palette();
  const body = el('div', { class: 'fig-body' });
  const grid = el('div', { class: 'panel-grid' });

  const controls = controlRow([
    segmented({
      label: 'Dataset',
      options: DATASET_ORDER.map((k) => ({ value: k, label: makeDataset(k).name })),
      value: state.dataset,
      onChange: (v) => { state.dataset = v; rebuild(); }
    }),
    slider({
      label: 'entropy ε', min: 0.001, max: 0.06, value: state.eps, log: true,
      format: (v) => v.toFixed(3),
      onInput: (v) => { state.eps = v; render(); }
    })
  ]);

  body.appendChild(controls);
  body.appendChild(legend([
    { colour: p.source, label: 'source' },
    { colour: p.target, label: 'target' }
  ]));
  body.appendChild(el('p', {
    class: 'fig-hint',
    text: 'A point is drawn solid when the plan moves all of its mass and hollow when it keeps it. Hover a point to isolate its flows.'
  }));
  body.appendChild(grid);
  root.appendChild(body);

  const tip = tooltip(body);
  const panels = METHODS.map((method) => makePanel(method, state, () => render(), tip));
  panels.forEach((pn) => grid.appendChild(pn.node));

  let data = makeDataset(state.dataset);
  let C = null;

  function rebuild() {
    data = makeDataset(state.dataset);
    note.textContent = data.note;
    C = null;
    render();
  }

  const render = scheduler(() => {
    const width = figureWidth(grid, 980);
    const cols = width < 560 ? 1 : 2;
    const panelW = Math.floor((width - (cols - 1) * 16) / cols);
    const panelH = Math.round(Math.min(240, Math.max(180, panelW * 0.82)));
    if (!C) C = normalizeCost(squaredEuclidean(data.X, data.Y));
    for (const pn of panels) pn.render({ data, C, width: panelW, height: panelH });
  });

  const note = el('p', { class: 'fig-note', text: data.note });
  body.appendChild(note);

  onResize(grid, render);
  render();
  return { render };
}

function makePanel(method, state, requestRender, tip) {
  const meta = METHOD_META[method];
  const canvas = el('canvas', { class: 'panel-canvas' });
  const stats = readout([
    { key: 'mass', label: 'mass moved' },
    { key: 'cost', label: 'transport cost' },
    { key: 'slack', label: 'marginal slack', title: 'L1 distance between the requested marginals and the ones the plan achieves' }
  ]);

  let control = null;
  if (meta.param) {
    const cfg = meta.param;
    control = slider({
      label: cfg.label,
      min: cfg.min, max: cfg.max, step: cfg.step, log: !!cfg.log,
      value: state[cfg.key],
      format: (v) => (cfg.log ? v.toFixed(v < 1 ? 3 : 2) : v.toFixed(2)),
      onInput: (v) => { state[cfg.key] = v; requestRender(); }
    });
  }

  const node = el('div', { class: 'panel' }, [
    el('div', { class: 'panel-head' }, [
      el('h4', { text: meta.label }),
      el('code', { class: 'panel-projection', text: meta.projection })
    ]),
    canvas,
    control,
    stats
  ]);

  let last = null;

  canvas.addEventListener('mousemove', (ev) => {
    if (!last) return;
    const box = canvas.getBoundingClientRect();
    const px = ev.clientX - box.left, py = ev.clientY - box.top;
    const hit = pickPoint(last.scale, last.data.X, last.data.Y, px, py);
    if (!hit) { state.focus = null; tip.hide(); requestRender(); return; }
    state.focus = hit;
    const isSrc = hit.side === 'source';
    const mass = isSrc ? last.data.a[hit.index] : last.data.b[hit.index];
    const moved = isSrc ? last.rowMoved[hit.index] : last.colMoved[hit.index];
    const nodeBox = node.getBoundingClientRect();
    const parentBox = node.parentElement.parentElement.getBoundingClientRect();
    tip.show(
      `<strong>${isSrc ? 'source' : 'target'} #${hit.index}</strong><br>` +
      `mass ${fmt(mass, 4)}<br>moved ${fmt(moved, 4)} (${pct(moved / (mass || 1))})`,
      nodeBox.left - parentBox.left + px,
      nodeBox.top - parentBox.top + py
    );
    requestRender();
  });
  canvas.addEventListener('mouseleave', () => { state.focus = null; tip.hide(); requestRender(); });

  function render({ data, C, width, height }) {
    const ctx = prepareCanvas(canvas, width, height);
    const scale = fitScale(width, height, [data.X, data.Y], 20);
    const res = solve({
      C, a: data.a, b: data.b, method,
      eps: state.eps, s: state.s, tau: state.tau,
      cutoff: method === 'supervised' ? state.cutoff : undefined
    });
    const info = drawTransport(ctx, {
      X: data.X, Y: data.Y, a: data.a, b: data.b, P: res.P,
      width, height, scale, highlight: data.highlightX, focus: state.focus
    });
    const d = diagnostics(res.P, C, data.a, data.b, data.X.length, data.Y.length);
    stats.update({
      mass: pct(d.massFraction),
      cost: fmt(d.cost, 4),
      slack: fmt(d.sourceViolation + d.targetViolation, 3)
    });
    last = { scale, data, rowMoved: info.rowMoved, colMoved: info.colMoved };
  }

  return { node, render };
}
