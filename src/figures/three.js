/**
 * Figure: the same four problems on a 3D helix, slowly turning.
 *
 * The plan is solved once per parameter change (in the worker); the rotation
 * is only a re-projection, so it costs nothing and never waits on a solve.
 * Drag to turn it by hand; it resumes on its own when you let go.
 */

import { squaredEuclidean, normalizeCost, diagnostics, METHOD_META } from '../lib/ot/solvers.js';
import { sharedSolver } from '../lib/ot/client.js';
import { helix3d } from '../lib/datasets.js';
import { prepareCanvas } from '../lib/plot.js';
import { palette, withAlpha } from '../lib/palette.js';
import {
  el, controlRow, slider, segmented, checkbox, readout, legend,
  scheduler, onResize, figureWidth, fmt, pct, solverStatus
} from '../lib/ui.js';

export function threeFigure(root) {
  const state = {
    method: 'supervised', eps: 0.008, s: 0.7, tau: 0.3, cutoff: 0.12,
    spin: true, yaw: 0.6, pitch: 0.42
  };
  const p = palette();
  const solver = sharedSolver();

  const canvas = el('canvas', { class: 'three-canvas' });
  const stats = readout([
    { key: 'mass', label: 'mass moved' },
    { key: 'cost', label: 'transport cost' },
    { key: 'slack', label: 'marginal slack' },
    { key: 'route', label: 'longest route', title: 'Largest cost among routes carrying ≥ 1% of their source point’s mass' },
    { key: 'conv', label: '' }
  ]);
  const paramSlot = el('div', { class: 'param-slot' });
  const body = el('div', { class: 'fig-body' });

  body.appendChild(controlRow([
    segmented({
      label: 'Method',
      options: Object.keys(METHOD_META).map((k) => ({ value: k, label: METHOD_META[k].short })),
      value: state.method,
      onChange: (v) => { state.method = v; warm = null; buildParam(); resolve(); }
    }),
    slider({
      label: 'entropy ε', min: 0.003, max: 0.05, value: state.eps, log: true,
      format: (v) => v.toFixed(3),
      onInput: (v) => { state.eps = v; resolve(); }
    }),
    paramSlot,
    checkbox({ label: 'rotate', checked: state.spin, onChange: (v) => { state.spin = v; } })
  ]));
  body.appendChild(legend([
    { colour: p.source, label: 'source helix' },
    { colour: p.target, label: 'target helix (top turn missing, plus a stray clump)' }
  ]));
  body.appendChild(el('p', { class: 'fig-hint', text: 'Drag to turn. Neighbouring turns of the helix are close in space, so a small cutoff permits matches along the curve but refuses the jump between turns.' }));
  body.appendChild(canvas);
  body.appendChild(stats);
  root.appendChild(body);

  const data = helix3d();
  const n = data.X.length, m = data.Y.length;
  const C = normalizeCost(squaredEuclidean(data.X, data.Y));
  let warm = null;
  let plan = null;
  let size = { w: 640, h: 440 };

  const PARAMS = {
    partial: { key: 's', label: 'mass fraction s', min: 0.05, max: 1, step: 0.01 },
    unbalanced: { key: 'tau', label: 'KL penalty τ', min: 0.001, max: 5, log: true },
    supervised: { key: 'cutoff', label: 'cost cutoff', min: 0.02, max: 1, step: 0.005 }
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
      onInput: (v) => { state[cfg.key] = v; resolve(); }
    }));
  }

  /* ---- solving --------------------------------------------------------- */

  function resolve() {
    solver.latest('three', {
      C, a: data.a, b: data.b, method: state.method,
      eps: state.eps, s: state.s, tau: state.tau,
      cutoff: state.method === 'supervised' ? state.cutoff : undefined,
      warmStart: warm
    }, (res) => {
      warm = { f: res.f, g: res.g };
      plan = res;
      const d = diagnostics(res.P, C, data.a, data.b, n, m);
      stats.update({
        mass: pct(d.massFraction),
        cost: fmt(d.cost, 4),
        slack: fmt(d.sourceViolation + d.targetViolation, 3),
        route: fmt(d.longestRoute, 3),
        conv: solverStatus(res)
      });
      drawFrame();
    }, { onBusy: (busy) => canvas.classList.toggle('is-computing', busy) });
  }

  /* ---- projection ------------------------------------------------------ */

  // Rotate about the vertical axis by yaw, then tip toward the viewer by pitch.
  function project(pt) {
    const [x, y, z] = pt;
    const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
    const x1 = x * cy + z * sy;
    const z1 = -x * sy + z * cy;
    const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
    const y2 = y * cp - z1 * sp;
    const z2 = y * sp + z1 * cp;
    return [x1, y2, z2];   // z2 toward the viewer
  }

  // Fit to the bulk of the cloud, not to the farthest stray point: use the
  // 92nd-percentile radius so the helix fills the frame and the clump can sit
  // near the edge.
  const radii = [...data.X, ...data.Y].map((q) => Math.hypot(q[0], q[1], q[2])).sort((u, v) => u - v);
  const extent = Math.max(0.5, radii[Math.floor(radii.length * 0.92)]);

  function drawFrame() {
    const { w, h } = size;
    const ctx = prepareCanvas(canvas, w, h);
    const k = Math.min(w, h) * 0.46 / extent;
    const cx = w / 2, cy = h / 2;
    const toScreen = (q) => [cx + q[0] * k, cy - q[1] * k];
    const depthScale = (z) => 0.78 + 0.22 * ((z / extent) + 1) / 2;   // nearer = bigger

    const PX = data.X.map(project), PY = data.Y.map(project);

    // ---- flows, drawn first, back to front by mean depth ---------------
    if (plan) {
      const P = plan.P;
      let maxP = 0;
      for (let q = 0; q < n * m; q++) if (P[q] > maxP) maxP = P[q];
      const cutoff = maxP * 0.035;
      const segs = [];
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < m; j++) {
          const v = P[i * m + j];
          if (v > cutoff) segs.push([i, j, v, (PX[i][2] + PY[j][2]) / 2]);
        }
      }
      segs.sort((s1, s2) => s1[3] - s2[3]);
      ctx.lineCap = 'round';
      for (const [i, j, v, z] of segs) {
        const t = v / maxP;
        const ds = depthScale(z);
        ctx.strokeStyle = withAlpha(p.flow, (0.06 + 0.55 * Math.pow(t, 0.9)) * ds);
        ctx.lineWidth = (0.6 + 1.8 * Math.pow(t, 0.8)) * ds;
        const a = toScreen(PX[i]), b = toScreen(PY[j]);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      }
    }

    // ---- points, back to front ---------------------------------------
    const rowMoved = new Float64Array(n), colMoved = new Float64Array(m);
    if (plan) {
      for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) { const v = plan.P[i * m + j]; rowMoved[i] += v; colMoved[j] += v; }
    }
    const pts = [];
    PX.forEach((q, i) => pts.push({ q, colour: p.source, frac: plan ? Math.min(1, rowMoved[i] / data.a[i]) : 1 }));
    PY.forEach((q, j) => pts.push({ q, colour: p.target, frac: plan ? Math.min(1, colMoved[j] / data.b[j]) : 1 }));
    pts.sort((u, v) => u.q[2] - v.q[2]);

    for (const { q, colour, frac } of pts) {
      const [sx, sy] = toScreen(q);
      const r = 4.2 * depthScale(q[2]);
      ctx.beginPath(); ctx.arc(sx, sy, r + 1.3, 0, Math.PI * 2);
      ctx.fillStyle = p.surface; ctx.fill();
      ctx.beginPath(); ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fillStyle = withAlpha(colour, 0.14 + 0.76 * frac); ctx.fill();
      ctx.lineWidth = 1.1; ctx.strokeStyle = withAlpha(colour, 0.95); ctx.stroke();
    }
  }

  /* ---- animation and drag --------------------------------------------- */

  let last = performance.now();
  let dragging = false, dragX = 0, dragY = 0, visible = true;

  function tick(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    if (state.spin && !dragging && visible) {
      state.yaw += dt * 0.35;
      drawFrame();
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  if (typeof IntersectionObserver !== 'undefined') {
    new IntersectionObserver((es) => { visible = es.some((e) => e.isIntersecting); }).observe(canvas);
  }

  canvas.addEventListener('pointerdown', (ev) => {
    dragging = true; dragX = ev.clientX; dragY = ev.clientY;
    canvas.setPointerCapture(ev.pointerId);
  });
  canvas.addEventListener('pointermove', (ev) => {
    if (!dragging) return;
    state.yaw += (ev.clientX - dragX) * 0.008;
    state.pitch = Math.max(-1.2, Math.min(1.2, state.pitch + (ev.clientY - dragY) * 0.006));
    dragX = ev.clientX; dragY = ev.clientY;
    drawFrame();
  });
  const release = () => { dragging = false; };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);

  const layout = scheduler(() => {
    const w = figureWidth(body, 720);
    size = { w, h: Math.round(Math.max(300, Math.min(460, w * 0.66))) };
    drawFrame();
  });

  buildParam();
  onResize(body, layout);
  layout();
  resolve();
  return { render: () => { layout(); resolve(); } };
}
