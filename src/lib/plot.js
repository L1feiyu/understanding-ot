/**
 * Rendering primitives shared by every figure: the transport plot, the coupling
 * matrix, marginal bars and a small line chart. Canvas for anything O(n*m),
 * SVG-free everywhere so there is nothing to install.
 *
 * Mark conventions follow one rule set across the article: thin marks, hairline
 * recessive axes, a 2px surface ring on overlapping markers, and no number
 * printed on every point — values live in the hover layer and the readouts.
 */

import { palette, withAlpha, sequential } from './palette.js';

/* ------------------------------------------------------------------ */
/* canvas plumbing                                                     */
/* ------------------------------------------------------------------ */

/** Size a canvas for the device pixel ratio and return a ready 2D context. */
export function prepareCanvas(canvas, width, height) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return ctx;
}

/**
 * Map data to a padded box using the true extent of the points, preserving
 * aspect. Assuming [-1,1] wastes most of the canvas on datasets that do not
 * happen to fill the square.
 */
export function fitScale(width, height, clouds, pad = 22) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const pts of clouds) {
    for (const [x, y] of pts) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  if (!Number.isFinite(minX)) return makeScale(width, height, pad);
  const spanX = Math.max(maxX - minX, 1e-6), spanY = Math.max(maxY - minY, 1e-6);
  const k = Math.min((width - pad * 2) / spanX, (height - pad * 2) / spanY);
  const ox = (width - spanX * k) / 2 - minX * k;
  const oy = (height - spanY * k) / 2 + maxY * k;
  return {
    x: (v) => ox + v * k,
    y: (v) => oy - v * k,
    k,
    invert: (px, py) => [(px - ox) / k, (oy - py) / k]
  };
}

/** Map data coordinates in [-1,1]^2 onto a padded box, preserving aspect. */
export function makeScale(width, height, pad = 18) {
  const w = width - pad * 2, h = height - pad * 2;
  const k = Math.min(w, h) / 2;
  const cx = width / 2, cy = height / 2;
  return {
    x: (v) => cx + v * k,
    y: (v) => cy - v * k,
    k,
    invert: (px, py) => [(px - cx) / k, (cy - py) / k]
  };
}

/* ------------------------------------------------------------------ */
/* the transport plot                                                  */
/* ------------------------------------------------------------------ */

/**
 * Draw a source cloud, a target cloud, and the transport plan between them.
 *
 * Point area encodes mass. Fill opacity encodes the fraction of that point's
 * mass the plan actually moved, so a point that keeps its mass reads as a hollow
 * ring — which is exactly the thing partial, unbalanced and supervised OT do
 * differently, made visible without spending another colour on it.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {object} o
 * @param {number[][]} o.X source points        @param {number[][]} o.Y target points
 * @param {ArrayLike<number>} o.a source masses @param {ArrayLike<number>} o.b target masses
 * @param {Float64Array} [o.P] transport plan, row-major n*m
 * @param {number} o.width @param {number} o.height
 * @param {number[]} [o.highlight] indices of source points to ring in the accent colour
 * @param {{side:string,index:number}} [o.focus] point whose flows are emphasised
 */
export function drawTransport(ctx, o) {
  const { X, Y, a, b, P, width, height } = o;
  const n = X.length, m = Y.length;
  const p = palette();
  const scale = o.scale || makeScale(width, height);

  ctx.clearRect(0, 0, width, height);

  const rowMoved = new Float64Array(n);
  const colMoved = new Float64Array(m);
  let maxP = 0;
  if (P) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        const v = P[i * m + j];
        rowMoved[i] += v;
        colMoved[j] += v;
        if (v > maxP) maxP = v;
      }
    }
  }

  // ---- flows ---------------------------------------------------------
  if (P && maxP > 0) {
    const cutoff = maxP * 0.035;
    ctx.lineCap = 'round';
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        const v = P[i * m + j];
        if (v <= cutoff) continue;
        const t = v / maxP;
        let alpha = 0.06 + 0.60 * Math.pow(t, 0.9);
        let colour = p.flow;
        if (o.focus) {
          const isFocus = (o.focus.side === 'source' && o.focus.index === i) ||
                          (o.focus.side === 'target' && o.focus.index === j);
          if (isFocus) { alpha = Math.min(0.95, alpha + 0.35); colour = p.accent; }
          else alpha *= 0.18;
        }
        ctx.strokeStyle = withAlpha(colour, alpha);
        ctx.lineWidth = 0.6 + 1.9 * Math.pow(t, 0.8);
        ctx.beginPath();
        ctx.moveTo(scale.x(X[i][0]), scale.y(X[i][1]));
        ctx.lineTo(scale.x(Y[j][0]), scale.y(Y[j][1]));
        ctx.stroke();
      }
    }
  }

  // ---- points --------------------------------------------------------
  const maxMass = Math.max(...a, ...b);
  const radius = (mass) => 2.6 + 4.4 * Math.sqrt(mass / maxMass);

  const drawSide = (pts, masses, moved, colour, highlight) => {
    for (let i = 0; i < pts.length; i++) {
      const cx = scale.x(pts[i][0]), cy = scale.y(pts[i][1]);
      const r = radius(masses[i]);
      const frac = P ? Math.max(0, Math.min(1, moved[i] / (masses[i] || 1e-12))) : 1;

      // 2px surface ring keeps overlapping markers legible without a border.
      ctx.beginPath();
      ctx.arc(cx, cy, r + 1.4, 0, Math.PI * 2);
      ctx.fillStyle = p.surface;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = withAlpha(colour, 0.14 + 0.76 * frac);
      ctx.fill();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = withAlpha(colour, 0.95);
      ctx.stroke();

      if (highlight && highlight.includes(i)) {
        ctx.beginPath();
        ctx.arc(cx, cy, r + 4.2, 0, Math.PI * 2);
        ctx.lineWidth = 1.6;
        ctx.strokeStyle = p.accent;
        ctx.stroke();
      }
    }
  };

  drawSide(X, a, rowMoved, p.source, o.highlight);
  drawSide(Y, b, colMoved, p.target, o.highlightTarget);

  // Direct labels, so group identity never rests on position alone.
  if (o.annotations) {
    ctx.font = '600 11px ui-sans-serif, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const an of o.annotations) {
      const tx = scale.x(an.x), ty = scale.y(an.y);
      ctx.lineWidth = 3.5;
      ctx.strokeStyle = p.surface;
      ctx.strokeText(an.text, tx, ty);
      ctx.fillStyle = p.inkSecondary;
      ctx.fillText(an.text, tx, ty);
    }
  }

  return { scale, rowMoved, colMoved };
}

/** Nearest point to a pixel position, across both clouds. */
export function pickPoint(scale, X, Y, px, py, radius = 14) {
  let best = null, bestD = radius * radius;
  const scan = (pts, side) => {
    for (let i = 0; i < pts.length; i++) {
      const dx = scale.x(pts[i][0]) - px, dy = scale.y(pts[i][1]) - py;
      const d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = { side, index: i }; }
    }
  };
  scan(X, 'source');
  scan(Y, 'target');
  return best;
}

/* ------------------------------------------------------------------ */
/* coupling matrix                                                     */
/* ------------------------------------------------------------------ */

/**
 * The transport plan as a heatmap, with the requested marginals drawn as
 * hairline bars along the top and left and the achieved marginals filled in.
 * Sequential encoding, one hue, light to dark.
 */
export function drawCoupling(ctx, o) {
  const { P, n, m, a, b, width, height } = o;
  const p = palette();
  const gutter = 34;
  const cellW = (width - gutter) / m;
  const cellH = (height - gutter) / n;

  ctx.clearRect(0, 0, width, height);

  let maxP = 0;
  for (let k = 0; k < n * m; k++) if (P[k] > maxP) maxP = P[k];
  const norm = maxP > 0 ? 1 / maxP : 0;

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      const t = Math.pow(P[i * m + j] * norm, 0.45);
      if (t < 0.004) continue;
      ctx.fillStyle = sequential(t);
      ctx.fillRect(gutter + j * cellW, gutter + i * cellH, Math.max(cellW - 0.5, 0.5), Math.max(cellH - 0.5, 0.5));
    }
  }

  // achieved vs requested marginals
  const rows = new Float64Array(n), cols = new Float64Array(m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) { rows[i] += P[i * m + j]; cols[j] += P[i * m + j]; }
  const maxA = Math.max(...a), maxB = Math.max(...b);
  const barMax = gutter - 10;

  for (let i = 0; i < n; i++) {
    const y = gutter + i * cellH, h = Math.max(cellH - 0.5, 0.5);
    ctx.fillStyle = withAlpha(p.source, 0.30);
    ctx.fillRect(4, y, (a[i] / maxA) * barMax, h);
    ctx.fillStyle = withAlpha(p.source, 0.95);
    ctx.fillRect(4, y, (rows[i] / maxA) * barMax, h);
  }
  for (let j = 0; j < m; j++) {
    const x = gutter + j * cellW, w = Math.max(cellW - 0.5, 0.5);
    const full = (b[j] / maxB) * barMax, got = (cols[j] / maxB) * barMax;
    ctx.fillStyle = withAlpha(p.target, 0.30);
    ctx.fillRect(x, gutter - 4 - full, w, full);
    ctx.fillStyle = withAlpha(p.target, 0.95);
    ctx.fillRect(x, gutter - 4 - got, w, got);
  }

  ctx.strokeStyle = p.grid;
  ctx.lineWidth = 1;
  ctx.strokeRect(gutter + 0.5, gutter + 0.5, m * cellW - 1, n * cellH - 1);

  ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
  ctx.fillStyle = p.inkMuted;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText('target j →', gutter, 2);
  ctx.save();
  ctx.translate(9, gutter);
  ctx.rotate(Math.PI / 2);
  ctx.fillText('source i →', 0, 0);
  ctx.restore();

  return {
    cellAt(px, py) {
      const j = Math.floor((px - gutter) / cellW);
      const i = Math.floor((py - gutter) / cellH);
      if (i < 0 || j < 0 || i >= n || j >= m) return null;
      return { i, j, value: P[i * m + j] };
    }
  };
}

/* ------------------------------------------------------------------ */
/* small line chart                                                    */
/* ------------------------------------------------------------------ */

/**
 * One series, one hue — a single line needs no legend box, the title names it.
 * Used for the "how much mass moves as the knob turns" panels.
 */
export function drawLineChart(ctx, o) {
  const { points, width, height } = o;
  const p = palette();
  const padL = 38, padR = 14, padT = 12, padB = o.xLabel ? 38 : 26;
  const [x0, x1] = o.xDomain;
  const [y0, y1] = o.yDomain;
  const sx = (v) => padL + ((v - x0) / (x1 - x0 || 1)) * (width - padL - padR);
  const sy = (v) => height - padB - ((v - y0) / (y1 - y0 || 1)) * (height - padT - padB);

  ctx.clearRect(0, 0, width, height);
  ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
  ctx.textBaseline = 'middle';

  // recessive solid hairline grid, never dashed
  ctx.strokeStyle = p.grid;
  ctx.lineWidth = 1;
  for (const t of o.yTicks || []) {
    const y = Math.round(sy(t)) + 0.5;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(width - padR, y); ctx.stroke();
    ctx.fillStyle = p.inkMuted;
    ctx.textAlign = 'right';
    ctx.fillText(o.yFormat ? o.yFormat(t) : String(t), padL - 6, y);
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const t of o.xTicks || []) {
    ctx.fillStyle = p.inkMuted;
    ctx.fillText(o.xFormat ? o.xFormat(t) : String(t), sx(t), height - padB + 7);
  }
  if (o.xLabel) {
    ctx.fillStyle = p.inkMuted;
    ctx.fillText(o.xLabel, padL + (width - padL - padR) / 2, height - 11);
  }

  ctx.beginPath();
  points.forEach(([x, y], i) => (i ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y))));
  ctx.strokeStyle = o.colour || p.source;
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.stroke();

  if (o.marker != null) {
    const my = points.length ? nearestY(points, o.marker) : 0;
    ctx.beginPath();
    ctx.arc(sx(o.marker), sy(my), 4.5, 0, Math.PI * 2);
    ctx.fillStyle = p.surface;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = o.colour || p.source;
    ctx.stroke();
  }
  return { sx, sy };
}

function nearestY(points, x) {
  let best = points[0][1], bestD = Infinity;
  for (const [px, py] of points) {
    const d = Math.abs(px - x);
    if (d < bestD) { bestD = d; best = py; }
  }
  return best;
}
