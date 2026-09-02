/**
 * Figure: what each knob actually controls.
 *
 * Two rows of small multiples. The top row sweeps each method's parameter and
 * plots the mass it moves; the bottom row plots the longest route the plan
 * uses. Read together they make the article's point: partial OT's knob sets
 * how MUCH moves and leaves how FAR unconstrained, while supervised OT's knob
 * sets how far and lets how much follow.
 *
 * Three panels per row rather than three lines on one axis — the x axes are
 * different quantities, and a shared axis would invent a comparison that does
 * not exist. Each panel is one series, so none needs a legend.
 *
 * The sweep runs in the worker, one point at a time with warm starts, and the
 * curves fill in as results arrive.
 */

import { squaredEuclidean, normalizeCost, diagnostics } from '../lib/ot/solvers.js';
import { createSolver } from '../lib/ot/client.js';
import { makeDataset, DATASET_ORDER } from '../lib/datasets.js';
import { drawLineChart, prepareCanvas } from '../lib/plot.js';
import { palette } from '../lib/palette.js';
import { el, controlRow, slider, segmented, scheduler, onResize, figureWidth } from '../lib/ui.js';

const PANELS = [
  {
    method: 'partial', key: 's', title: 'Partial OT', axis: 'mass fraction s',
    values: linspace(0.05, 1, 20), log: false, ticks: [0.25, 0.5, 0.75, 1],
    massCaption: 'You name the mass. The line is the identity — that is the whole point of the method.',
    routeCaption: 'And nothing names the distance: as s rises, the plan reaches for whatever routes it needs.'
  },
  {
    method: 'unbalanced', key: 'tau', title: 'Unbalanced OT', axis: 'KL penalty τ',
    values: logspace(0.001, 5, 26), log: true, ticks: [0.001, 0.01, 0.1, 1],
    massCaption: 'Mass rises smoothly with τ and reaches the marginals only in the limit.',
    routeCaption: 'Route length rises with it. τ trades off both at once and controls neither directly.'
  },
  {
    method: 'supervised', key: 'cutoff', title: 'Supervised OT', axis: 'cost cutoff',
    values: linspace(0.03, 1, 36), log: false, ticks: [0.25, 0.5, 0.75, 1],
    massCaption: 'Mass climbs in steps as the cutoff admits groups of routes, then saturates.',
    routeCaption: 'The longest route is pinned to the cutoff — the identity line, by construction.'
  }
];

const ROWS = [
  { key: 'mass', title: 'mass moved', yDomain: [0, 1], yTicks: [0, 0.25, 0.5, 0.75, 1], yFormat: (t) => `${Math.round(t * 100)}%`, pick: (d) => d.massFraction, cap: 'massCaption' },
  { key: 'route', title: 'longest route used', yDomain: [0, 1], yTicks: [0, 0.25, 0.5, 0.75, 1], yFormat: (t) => t.toFixed(2), pick: (d) => d.longestRoute, cap: 'routeCaption' }
];

function linspace(a, b, n) {
  return Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
}
function logspace(a, b, n) {
  return Array.from({ length: n }, (_, i) => Math.exp(Math.log(a) + ((Math.log(b) - Math.log(a)) * i) / (n - 1)));
}

export function massCurveFigure(root) {
  const state = { dataset: 'extraCluster', eps: 0.008 };
  // Own worker: a sweep is ~80 solves and must not queue ahead of a slider elsewhere.
  const solver = createSolver();
  const body = el('div', { class: 'fig-body' });

  body.appendChild(controlRow([
    segmented({
      label: 'Dataset',
      options: DATASET_ORDER.slice(0, 4).map((k) => ({ value: k, label: makeDataset(k).name })),
      value: state.dataset,
      onChange: (v) => { state.dataset = v; render(); }
    }),
    slider({
      label: 'entropy ε', min: 0.003, max: 0.05, value: state.eps, log: true,
      format: (v) => v.toFixed(3),
      onInput: (v) => { state.eps = v; render(); }
    })
  ]));

  // One grid per row, with the row's quantity named once above it.
  const rowNodes = ROWS.map((row) => {
    body.appendChild(el('h4', { class: 'sub curve-row-title', text: row.title }));
    const grid = el('div', { class: 'curve-grid' });
    const cells = PANELS.map((cfg) => {
      const canvas = el('canvas', {});
      const node = el('div', { class: 'curve-panel' }, [
        el('div', { class: 'curve-title', text: cfg.title }),
        canvas,
        el('p', { class: 'curve-caption', text: cfg[row.cap] })
      ]);
      grid.appendChild(node);
      return { cfg, canvas };
    });
    body.appendChild(grid);
    return { row, grid, cells };
  });
  root.appendChild(body);

  let generation = 0;

  const render = scheduler(() => {
    const gen = ++generation;
    const data = makeDataset(state.dataset, 30);
    const C = normalizeCost(squaredEuclidean(data.X, data.Y));
    const n = data.X.length, m = data.Y.length;

    const width = figureWidth(body, 960);
    const cols = width < 620 ? 1 : 3;
    const w = Math.floor((width - (cols - 1) * 14) / cols);
    const h = Math.round(Math.max(150, Math.min(180, w * 0.72)));
    const p = palette();

    // results[panelIndex] = array of diagnostics, filled in as the sweep runs
    const results = PANELS.map(() => []);

    const paint = () => {
      for (const { row, cells } of rowNodes) {
        cells.forEach(({ cfg, canvas }, pi) => {
          const pts = results[pi].map((d, k) => [cfg.log ? Math.log10(cfg.values[k]) : cfg.values[k], row.pick(d)]);
          const xs = cfg.values.map((v) => (cfg.log ? Math.log10(v) : v));
          const ctx = prepareCanvas(canvas, w, h);
          drawLineChart(ctx, {
            points: pts, width: w, height: h,
            xDomain: [Math.min(...xs), Math.max(...xs)],
            yDomain: row.yDomain, yTicks: row.yTicks, yFormat: row.yFormat,
            xTicks: cfg.log ? cfg.ticks.map(Math.log10) : cfg.ticks,
            xFormat: (t) => (cfg.log ? formatLog(t) : t.toFixed(2)),
            xLabel: cfg.axis,
            colour: p.source
          });
          canvas.setAttribute('aria-label', `${cfg.title}: ${row.title} versus ${cfg.axis}`);
        });
      }
    };
    paint();

    // Sweep each panel sequentially so warm starts chain along the parameter.
    PANELS.forEach((cfg, pi) => {
      let warm = null;
      const step = (k) => {
        if (gen !== generation || k >= cfg.values.length) return;
        solver.solve({
          C, a: data.a, b: data.b, method: cfg.method, eps: state.eps,
          [cfg.key]: cfg.values[k], warmStart: warm
        }).then((res) => {
          if (gen !== generation) return;
          warm = { f: res.f, g: res.g };
          results[pi][k] = diagnostics(res.P, C, data.a, data.b, n, m);
          paint();
          step(k + 1);
        });
      };
      step(0);
    });
  });

  onResize(body, render);
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
