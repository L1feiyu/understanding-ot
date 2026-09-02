/**
 * Page entry point: mount each figure into its slot and wire the theme toggle.
 *
 * Figures are mounted lazily as they scroll into view — every one of them runs a
 * live solver, and there is no reason to spend that before the reader arrives.
 */

import { comparisonFigure } from './figures/comparison.js';
import { couplingFigure } from './figures/coupling.js';
import { projectionsFigure } from './figures/projections.js';
import { massCurveFigure } from './figures/massCurve.js';
import { threeFigure } from './figures/three.js';
import { invalidatePalette } from './lib/palette.js';

const FIGURES = [
  ['fig-comparison', comparisonFigure],
  ['fig-coupling', couplingFigure],
  ['fig-projections', projectionsFigure],
  ['fig-mass', massCurveFigure],
  ['fig-three', threeFigure]
];

const mounted = [];

function mount(node, build) {
  if (node.dataset.mounted) return;
  node.dataset.mounted = '1';
  try {
    const handle = build(node);
    if (handle && handle.render) mounted.push(handle);
  } catch (err) {
    node.appendChild(Object.assign(document.createElement('p'), {
      className: 'fig-note',
      textContent: `This figure failed to load: ${err.message}`
    }));
    console.error('figure failed', node.id, err);
  }
}

function init() {
  const slots = FIGURES
    .map(([id, build]) => [document.getElementById(id), build])
    .filter(([node]) => node);

  if (typeof IntersectionObserver === 'undefined') {
    slots.forEach(([node, build]) => mount(node, build));
  } else {
    const io = new IntersectionObserver((entries, obs) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        const found = slots.find(([node]) => node === e.target);
        if (found) mount(found[0], found[1]);
        obs.unobserve(e.target);
      }
    }, { rootMargin: '300px 0px' });
    slots.forEach(([node]) => io.observe(node));
    // The first figure is above the fold on most screens; do not wait for a scroll.
    if (slots.length) mount(slots[0][0], slots[0][1]);
  }

  setupTheme();
}

function setupTheme() {
  const btn = document.getElementById('theme-toggle');
  if (!btn) return;

  const stored = safeGet('ot-theme');
  if (stored === 'light' || stored === 'dark') document.documentElement.dataset.theme = stored;
  label();

  btn.addEventListener('click', () => {
    const current = document.documentElement.dataset.theme ||
      (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    safeSet('ot-theme', next);
    label();
    redraw();
  });

  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (!document.documentElement.dataset.theme) redraw();
  });

  function label() {
    const isDark = (document.documentElement.dataset.theme ||
      (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')) === 'dark';
    btn.textContent = isDark ? 'light theme' : 'dark theme';
  }
}

function redraw() {
  invalidatePalette();
  // Let the new custom properties land before figures read them back.
  requestAnimationFrame(() => mounted.forEach((f) => f.render()));
}

function safeGet(key) {
  try { return window.localStorage.getItem(key); } catch { return null; }
}
function safeSet(key, value) {
  try { window.localStorage.setItem(key, value); } catch { /* private mode */ }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
