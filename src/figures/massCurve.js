/**
 * Figure: how much mass each knob actually moves.
 *
 * Three small multiples rather than three lines on one axis — the x axes are
 * different quantities (a mass fraction, a KL penalty, an l1 penalty) and
 * putting them on a shared axis would invent a comparison that does not exist.
 * Each panel is one series, so none of them needs a legend.
 *
 * The shape worth noticing is in the third panel: mass as a function of the
 * cutoff is a staircase, because raising the cutoff admits whole groups of
 * routes at once, and it saturates as soon as every point has a partner.
 */

import { solve, squaredEuclidean, normalizeCost, totalMass } from '../lib/ot/solvers.js';
import { makeDataset, DATASET_ORDER } from '../lib/datasets.js';
import { drawLineChart, prepareCanvas } from '../lib/plot.js';
import { palette } from '../lib/palette.js';
import { el, controlRow, slider, segmented, scheduler, onResize, figureWidth } from '../lib/ui.js';

const PANELS = [
  {
    method: 'partial', key: 's', title: 'Partial OT', axis: 'mass fraction s',
    values: linspace(0.05, 1, 22), log: false, ticks: [0.25, 0.5, 0.75, 1],
    caption: 'You name the mass. The line is the identity — that is the whole point of the method.'
  },
  {
    method: 'unbalanced', key: 'tau', title: 'Unbalanced OT', axis: 'KL penalty τ',
    values: logspace(0.001, 5, 30), log: true, ticks: [0.001, 0.01, 0.1, 1],
    caption: 'Mass rises smoothly with τ and only reaches the marginals in the limit.'
  },
  {
    method: 'supervised', key: 'cutoff', title: 'Supervised OT', axis: 'cost cutoff',
    values: linspace(0.03, 1, 44), log: false, ticks: [0.25, 0.5, 0.75, 1],
    caption: 'A staircase: the cutoff admits whole groups of routes at once, and saturates once every point has a partner.'
  }
];

function linspace(a, b, n) {
  return Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
}
function logspace(a, b, n) {
  return Array.from({ length: n }, (_, i) => Math.exp(Math.log(a) + ((Math.log(b) - Math.log(a)) * i) / (n - 1)));
}

export function massCurveFigure(root) {
  const state = { dataset: 'extraCluster', eps: 0.005 };
  const body = el('div', { class: 'fig-body' });
  const grid = el('div', { class: 'curve-grid' });

  body.appendChild(controlRow([
    segmented({
      label: 'Dataset',
      options: DATASET_ORDER.slice(0, 4).map((k) => ({ value: k, label: makeDataset(k).name })),
      value: state.dataset,
      onChange: (v) => { state.dataset = v; render(); }
    }),
    slider({
      label: 'entropy ε', min: 0.002, max: 0.05, value: state.eps, log: true,
      format: (v) => v.toFixed(3),
      onInput: (v) => { state.eps = v; render(); }
    })
  ]));
  body.appendChild(grid);
  root.appendChild(body);

  const panels = PANELS.map((cfg) => {
    const canvas = el('canvas', {});
    const node = el('div', { class: 'curve-panel' }, [
      el('h4', { class: 'sub', text: cfg.title }),
      canvas,
      el('p', { class: 'curve-caption', text: cfg.caption })
    ]);
    grid.appendChild(node);
    return { cfg, canvas };
  });

  const render = scheduler(() => {
    const data = makeDataset(state.dataset, 30);
    const C = normalizeCost(squaredEuclidean(data.X, data.Y));
    const width = figureWidth(grid, 960);
    const cols = width < 620 ? 1 : 3;
    const w = Math.floor((width - (cols - 1) * 14) / cols);
    const h = Math.round(Math.max(165, Math.min(205, w * 0.8)));
    const p = palette();

    for (const { cfg, canvas } of panels) {
      const pts = cfg.values.map((v) => {
        const res = solve({
          C, a: data.a, b: data.b, method: cfg.method, eps: state.eps,
          [cfg.key]: v, nIter: cfg.method === 'partial' ? 500 : 400
        });
        const x = cfg.log ? Math.log10(v) : v;
        return [x, totalMass(res.P)];
      });
      const xs = pts.map((d) => d[0]);
      const ctx = prepareCanvas(canvas, w, h);
      drawLineChart(ctx, {
        points: pts, width: w, height: h,
        xDomain: [Math.min(...xs), Math.max(...xs)],
        yDomain: [0, 1],
        yTicks: [0, 0.25, 0.5, 0.75, 1],
        yFormat: (t) => `${Math.round(t * 100)}%`,
        xTicks: cfg.log ? cfg.ticks.map(Math.log10) : cfg.ticks,
        xFormat: (t) => (cfg.log ? formatLog(t) : t.toFixed(2)),
        xLabel: cfg.axis,
        colour: p.source
      });
      canvas.setAttribute('aria-label', `${cfg.title}: transported mass versus ${cfg.axis}`);
    }
  });

  onResize(grid, render);
  render();
  return { render };
}

function formatLog(t) {
  const v = Math.pow(10, t);
  if (v >= 1) return String(Math.round(v));
  if (v >= 0.1) return v.toFixed(1);
  if (v >= 0.01) return v.toFixed(2);
  return v.toFixed(3);
}
