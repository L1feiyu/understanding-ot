/**
 * Figure: one algorithm, four projections.
 *
 * The four methods are not four algorithms. They are one Sinkhorn loop with one
 * line changed, and this figure shows both halves of that claim: the line, and
 * what it does to the dual potentials it acts on.
 *
 * The strip plot is the payoff. Balanced leaves the potentials alone; supervised
 * clips them against a ceiling at gamma; unbalanced shrinks them toward zero;
 * partial does the clipping too, with the ceiling chosen for you.
 */

import { solve, squaredEuclidean, normalizeCost, totalMass, METHOD_META, SOT_GAMMA } from '../lib/ot/solvers.js';
import { makeDataset } from '../lib/datasets.js';
import { prepareCanvas } from '../lib/plot.js';
import { palette, withAlpha } from '../lib/palette.js';
import {
  el, controlRow, slider, segmented, scheduler, onResize, figureWidth, fmt, pct
} from '../lib/ui.js';

const LINES = {
  balanced:   'f[i] = f[i] + corr',
  partial:    'f[i] = min(f[i] + corr, γ)      // γ solved for, to hit mass s',
  unbalanced: 'f[i] = (τ/(τ+ε)) * (f[i] + corr)',
  supervised: 'f[i] = min(f[i] + corr, γ)      // γ = 2, fixed; C[i][j] = ∞ past the cutoff'
};

const EXPLAIN = {
  balanced: 'No projection at all. The update sets the row sum to exactly a[i], every time, which is what makes the marginal a hard constraint.',
  partial: 'The same ceiling as supervised OT, but you specify the mass and the solver searches for the γ that delivers it. Same algorithm, opposite direction.',
  unbalanced: 'A shrink toward zero. Potentials never reach the value that would enforce the marginal, and how far short they fall is set by τ.',
  supervised: 'A hard ceiling at γ, which is pinned at 2 and never moved. Since every normalised cost is below 2γ, the price never bites on a permitted route — the only potentials that reach the ceiling belong to points stranded by the cutoff, with no permitted partner left.'
};

export function projectionsFigure(root) {
  const state = { method: 'supervised', eps: 0.008, cutoff: 0.22, tau: 0.25, s: 0.6 };
  const p = palette();

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
      onChange: (v) => { state.method = v; buildParam(); render(); }
    }),
    slider({
      label: 'entropy ε', min: 0.002, max: 0.05, value: state.eps, log: true,
      format: (v) => v.toFixed(3),
      onInput: (v) => { state.eps = v; render(); }
    }),
    paramSlot
  ]));
  body.appendChild(code);
  body.appendChild(el('h4', { class: 'sub', text: 'The source potentials f, sorted' }));
  body.appendChild(el('p', { class: 'fig-hint', text: 'Each bar is one f[i]. The dashed rule is the ceiling the projection imposes, where there is one.' }));
  body.appendChild(canvas);
  body.appendChild(explain);
  root.appendChild(body);

  const data = makeDataset('extraCluster', 36);
  const C = normalizeCost(squaredEuclidean(data.X, data.Y));

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

  const render = scheduler(() => {
    code.innerHTML =
      'for each iteration:\n' +
      '  corr  = ε·log(a[i]) − ε·log( Σ<sub>j</sub> exp((f[i] + g[j] − C[i][j]) / ε) )\n' +
      `  <mark>${LINES[state.method]}</mark>\n` +
      '  … and the same for g';

    const res = solve({
      C, a: data.a, b: data.b, method: state.method,
      eps: state.eps, s: state.s, tau: state.tau,
      cutoff: state.method === 'supervised' ? state.cutoff : undefined
    });

    const width = figureWidth(body, 720);
    const height = 190;
    const ctx = prepareCanvas(canvas, width, height);
    const f = Array.from(res.f).sort((x, y) => x - y);

    const ceiling =
      state.method === 'supervised' ? SOT_GAMMA :
      state.method === 'partial' ? res.gammaEquivalent :
      null;

    // Stranded points sit exactly at gamma = 2 while everything else lives near
    // 0.1–0.4, so scaling to the ceiling would flatten every bar that matters.
    // Scale to the uncapped potentials instead and let capped bars run off the
    // top, marked as clipped.
    const atCap = (v) => ceiling != null && v >= ceiling - 1e-9;
    const free = f.filter((v) => !atCap(v));
    const lo = Math.min(0, ...f);
    const hi = free.length
      ? Math.max(...free) * 1.3
      : Math.max(...f, ceiling ?? -Infinity, 0.001);
    const clipping = ceiling != null && ceiling > hi;
    const padT = 14, padB = 22, padL = 40, padR = 12;
    const sy = (v) => height - padB - ((v - lo) / (hi - lo || 1)) * (height - padT - padB);
    const bw = (width - padL - padR) / f.length;

    // zero rule, solid hairline
    ctx.strokeStyle = p.grid;
    ctx.lineWidth = 1;
    const zeroY = Math.round(sy(0)) + 0.5;
    ctx.beginPath(); ctx.moveTo(padL, zeroY); ctx.lineTo(width - padR, zeroY); ctx.stroke();
    ctx.fillStyle = p.inkMuted;
    ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText('0', padL - 6, zeroY);

    for (let i = 0; i < f.length; i++) {
      const capped = atCap(f[i]);
      ctx.fillStyle = capped ? withAlpha(p.accent, 0.88) : withAlpha(p.source, 0.85);
      const y = capped && clipping ? padT : sy(f[i]);
      const h = Math.abs(zeroY - y);
      // 2px surface gap between adjacent bars
      const x = padL + i * bw + 1, w = Math.max(bw - 2, 1);
      ctx.fillRect(x, Math.min(y, zeroY), w, Math.max(h, 1));
      // Break marker so a clipped bar never reads as a real value.
      if (capped && clipping) {
        ctx.strokeStyle = p.surface;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x - 0.5, padT + 7); ctx.lineTo(x + w + 0.5, padT + 3);
        ctx.moveTo(x - 0.5, padT + 12); ctx.lineTo(x + w + 0.5, padT + 8);
        ctx.stroke();
      }
    }

    if (ceiling != null) {
      const cy = clipping ? padT + 0.5 : Math.round(sy(ceiling)) + 0.5;
      ctx.save();
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = p.accent;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(padL, cy); ctx.lineTo(width - padR, cy); ctx.stroke();
      ctx.restore();
      ctx.fillStyle = p.accent;
      ctx.textAlign = 'left';
      ctx.fillText(
        clipping ? `ceiling γ = ${fmt(ceiling, 2)} (bars clipped)` : `ceiling γ = ${fmt(ceiling, 3)}`,
        padL + 4, cy + 9
      );
    }

    const capped = f.filter(atCap).length;
    explain.textContent =
      `${EXPLAIN[state.method]} ` +
      (ceiling != null
        ? `${capped} of ${f.length} source potentials sit against the ceiling; those are the points keeping their mass. `
        : '') +
      `Mass moved: ${pct(totalMass(res.P))}.`;
  });

  buildParam();
  onResize(body, render);
  render();
  return { render };
}
