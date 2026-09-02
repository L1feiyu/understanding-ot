/**
 * Toy source/target pairs used throughout the article.
 *
 * Each dataset returns { X, Y, a, b, labelsX, labelsY, name, note } where X and Y
 * are arrays of [x, y] in roughly [-1, 1]^2, and a / b are the marginal masses.
 * Optional fields: `legend` (names for the two sides), `marker` (glyph per
 * side: circle | square | diamond | triangle), and `scene` — decorations drawn
 * under the points: polylines (stroke: source | target | ink | water) and
 * labels. Labels exist so class-based supervision can forbid cross-class pairs.
 *
 * Randomness is seeded so that every reader sees the same picture and reloading
 * does not reshuffle the article.
 */

/** Small deterministic PRNG (mulberry32). */
export function rng(seed = 1) {
  let t = seed >>> 0;
  return function next() {
    t += 0x6d2b79f5;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box–Muller, driven by a seeded uniform generator. */
function gaussian(next) {
  let u = 0, v = 0;
  while (u === 0) u = next();
  while (v === 0) v = next();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function cluster(next, n, cx, cy, sd) {
  return Array.from({ length: n }, () => [cx + gaussian(next) * sd, cy + gaussian(next) * sd]);
}

function uniformMass(n) {
  return Array.from({ length: n }, () => 1 / n);
}

/* ------------------------------------------------------------------ */

/** Two clusters against two clusters — the plain case, nothing pathological. */
function twoBlobs(n = 40) {
  const next = rng(7);
  const X = [...cluster(next, n / 2, -0.55, 0.35, 0.13), ...cluster(next, n / 2, -0.55, -0.4, 0.13)];
  const Y = [...cluster(next, n / 2, 0.55, 0.4, 0.13), ...cluster(next, n / 2, 0.55, -0.35, 0.13)];
  const labelsX = X.map((_, i) => (i < n / 2 ? 0 : 1));
  const labelsY = Y.map((_, i) => (i < n / 2 ? 0 : 1));
  return {
    name: 'Two blobs',
    note: 'A clean correspondence. Every relaxation should look much like balanced OT here.',
    X, Y, a: uniformMass(X.length), b: uniformMass(Y.length), labelsX, labelsY
  };
}

/** The target carries an extra cluster the source has no counterpart for. */
function extraCluster(n = 42) {
  const next = rng(11);
  const X = [...cluster(next, n / 2, -0.55, 0.3, 0.13), ...cluster(next, n / 2, -0.55, -0.35, 0.13)];
  const Y = [
    ...cluster(next, n / 3, 0.5, 0.45, 0.12),
    ...cluster(next, n / 3, 0.5, -0.3, 0.12),
    ...cluster(next, n / 3, 0.15, -0.85, 0.1)
  ];
  const labelsX = X.map((_, i) => (i < n / 2 ? 0 : 1));
  const labelsY = Y.map((_, i) => Math.floor(i / (n / 3)));
  return {
    name: 'Extra cluster',
    note: 'The target has a third mode with nothing to match it. Balanced OT must still fill it.',
    X, Y, a: uniformMass(X.length), b: uniformMass(Y.length), labelsX, labelsY
  };
}

/** A single far-away outlier in the source. */
function outlier(n = 40) {
  const next = rng(3);
  const X = [...cluster(next, n - 1, -0.5, 0, 0.18), [-0.15, 0.95]];
  const Y = cluster(next, n, 0.5, 0, 0.18);
  return {
    name: 'Outlier',
    note: 'One stray source point. Watch how much of the plan it distorts.',
    X, Y, a: uniformMass(X.length), b: uniformMass(Y.length),
    labelsX: X.map((_, i) => (i === n - 1 ? 1 : 0)),
    labelsY: Y.map(() => 0),
    highlightX: [n - 1]
  };
}

/** Same two modes on both sides, but the proportions differ sharply. */
function imbalance(n = 40) {
  const next = rng(23);
  const nUp = Math.round(n * 0.75), nDown = n - nUp;
  const X = [...cluster(next, nUp, -0.55, 0.35, 0.12), ...cluster(next, nDown, -0.55, -0.4, 0.12)];
  const Y = [...cluster(next, nDown, 0.55, 0.35, 0.12), ...cluster(next, nUp, 0.55, -0.4, 0.12)];
  const labelsX = X.map((_, i) => (i < nUp ? 0 : 1));
  const labelsY = Y.map((_, i) => (i < nDown ? 0 : 1));
  return {
    name: 'Class imbalance',
    note: '75/25 on the left, 25/75 on the right. Balanced OT has to move mass across classes.',
    X, Y, a: uniformMass(X.length), b: uniformMass(Y.length), labelsX, labelsY
  };
}

/** Three labelled classes on each side, shifted so the nearest neighbour lies. */
function crossedClasses(n = 45) {
  const next = rng(31);
  const k = Math.round(n / 3);
  const X = [
    ...cluster(next, k, -0.6, 0.55, 0.1),
    ...cluster(next, k, -0.6, 0.0, 0.1),
    ...cluster(next, k, -0.6, -0.55, 0.1)
  ];
  const Y = [
    ...cluster(next, k, 0.6, 0.25, 0.1),
    ...cluster(next, k, 0.6, -0.3, 0.1),
    ...cluster(next, k, 0.6, 0.8, 0.1)
  ];
  return {
    name: 'Crossed classes',
    note: 'Geometry disagrees with the labels: the closest target cluster is often the wrong class.',
    X, Y, a: uniformMass(X.length), b: uniformMass(Y.length),
    labelsX: X.map((_, i) => Math.floor(i / k)),
    labelsY: Y.map((_, i) => Math.floor(i / k))
  };
}

/** Two rings — mass is spread over a curve rather than concentrated. */
function rings(n = 44) {
  const next = rng(5);
  const X = Array.from({ length: n }, (_, i) => {
    const t = (i / n) * Math.PI * 2;
    return [-0.5 + Math.cos(t) * 0.32 + gaussian(next) * 0.02, Math.sin(t) * 0.32 + gaussian(next) * 0.02];
  });
  const Y = Array.from({ length: n }, (_, i) => {
    const t = (i / n) * Math.PI * 2;
    return [0.5 + Math.cos(t) * 0.2 + gaussian(next) * 0.02, Math.sin(t) * 0.45 + gaussian(next) * 0.02];
  });
  return {
    name: 'Rings',
    note: 'A circle stretched into an ellipse — a case where every point does have a partner.',
    X, Y, a: uniformMass(X.length), b: uniformMass(Y.length),
    labelsX: X.map(() => 0), labelsY: Y.map(() => 0)
  };
}

/* ------------------------------------------------------------------ */
/* scenes                                                              */
/* ------------------------------------------------------------------ */

function norm(masses) {
  const t = masses.reduce((x, y) => x + y, 0);
  return masses.map((v) => v / t);
}

/**
 * A town: bakeries supply croissants, cafés want them. Supply and demand vary
 * from place to place (marker size), the total matches, and a river cuts the
 * town in two with a single bridge — the two cafés on the far bank are a long
 * way from every bakery. The oldest example in the subject, and still the most
 * legible one.
 */
function bakeries() {
  const next = rng(17);
  const jit = (v) => v + gaussian(next) * 0.012;
  const X = [[-0.72, 0.42], [-0.48, 0.12], [-0.62, -0.32], [-0.22, 0.5], [-0.15, -0.12], [0.02, -0.5]].map(([x, y]) => [jit(x), jit(y)]);
  const supply = [0.9, 1.6, 0.7, 1.1, 2.0, 0.8];
  const Y = [
    [-0.4, 0.62], [-0.05, 0.22], [0.12, 0.62], [-0.32, -0.62], [0.28, -0.22], [0.2, 0.05],
    [0.55, 0.7], [0.68, 0.38],            // across the river
    [0.42, -0.72]
  ].map(([x, y]) => [jit(x), jit(y)]);
  const demand = [0.8, 1.2, 0.6, 0.9, 1.5, 0.7, 1.0, 0.9, 0.5];
  // the river: a gentle S from top to bottom, right of centre
  const river = [];
  for (let t = 0; t <= 1; t += 0.05) river.push([0.4 + Math.sin(t * Math.PI * 1.4) * 0.09, 0.95 - t * 1.9]);
  return {
    name: 'Bakeries & cafés',
    note: 'Marker size is supply or demand. Two cafés sit across the river; reaching them costs more than any other route in town.',
    legend: { source: 'bakeries (supply)', target: 'cafés (demand)' },
    marker: { source: 'square', target: 'circle' },
    scene: [
      { type: 'polyline', points: river, stroke: 'water', width: 10, alpha: 0.35 },
      { type: 'polyline', points: [[0.3, -0.05], [0.56, -0.05]], stroke: 'ink', width: 3, alpha: 0.35 },
      { type: 'label', at: [0.47, -0.11], text: 'bridge' },
      { type: 'label', at: [0.47, 0.86], text: 'river' }
    ],
    X, Y, a: norm(supply), b: norm(demand),
    labelsX: X.map(() => 0), labelsY: Y.map((_, i) => (i >= 6 && i <= 7 ? 1 : 0))
  };
}

/** Points along a parametric heart, and along a five-pointed star missing one arm. */
function heartStar(n = 48) {
  const next = rng(29);
  const X = Array.from({ length: n }, (_, i) => {
    const t = (i / n) * Math.PI * 2;
    const hx = 16 * Math.pow(Math.sin(t), 3);
    const hy = 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t);
    return [-0.5 + hx * 0.026 + gaussian(next) * 0.01, hy * 0.026 + 0.05 + gaussian(next) * 0.01];
  });
  // star outline: 10 vertices alternating outer/inner radius, sampled along edges
  const verts = [];
  for (let k = 0; k < 10; k++) {
    const ang = -Math.PI / 2 + (k * Math.PI) / 5;
    const r = k % 2 === 0 ? 0.46 : 0.19;
    verts.push([0.52 + Math.cos(ang) * r, Math.sin(ang) * r]);
  }
  const Y = [];
  const perEdge = 5;
  for (let k = 0; k < 10; k++) {
    if (k === 2 || k === 3) continue;   // the missing arm: skip both edges of one point
    const a0 = verts[k], a1 = verts[(k + 1) % 10];
    for (let q = 0; q < perEdge; q++) {
      const t = q / perEdge;
      Y.push([a0[0] + (a1[0] - a0[0]) * t + gaussian(next) * 0.008, a0[1] + (a1[1] - a0[1]) * t + gaussian(next) * 0.008]);
    }
  }
  const heartCurve = Array.from({ length: 80 }, (_, i) => {
    const t = (i / 79) * Math.PI * 2;
    return [-0.5 + 16 * Math.pow(Math.sin(t), 3) * 0.026, (13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)) * 0.026 + 0.05];
  });
  const starCurve = verts.concat([verts[0]]);
  return {
    name: 'Heart → star',
    note: 'Two outlines. The star has lost an arm, so a fifth of the heart has nowhere natural to go.',
    legend: { source: 'heart', target: 'star (one arm missing)' },
    scene: [
      { type: 'polyline', points: heartCurve, stroke: 'source', width: 1.2, alpha: 0.25 },
      { type: 'polyline', points: starCurve, stroke: 'target', width: 1.2, alpha: 0.25 }
    ],
    X, Y, a: uniformMass(X.length), b: uniformMass(Y.length),
    labelsX: X.map(() => 0), labelsY: Y.map(() => 0)
  };
}

/**
 * Two slices through the same tissue: three cell-type layers arranged as
 * concentric bands. The second slice is turned a little, its outer layer is
 * torn on one side, and a few immune cells have infiltrated. Aligning cells
 * across slices is the setting supervised OT was built for, and the cutoff is
 * the radius over which a cell can plausibly correspond.
 */
function tissue() {
  const next = rng(53);
  const bands = [[0.14, 14], [0.27, 22], [0.40, 30]];   // radius, count per band
  const make = (cx, cy, rot, gapBand, gapFrom, gapTo) => {
    const pts = [], labels = [];
    bands.forEach(([r, cnt], b) => {
      for (let k = 0; k < cnt; k++) {
        const ang = (k / cnt) * Math.PI * 2 + rot;
        const norm = ((ang - rot) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2);
        if (b === gapBand && norm > gapFrom && norm < gapTo) continue;
        pts.push([cx + Math.cos(ang) * r + gaussian(next) * 0.014, cy + Math.sin(ang) * r + gaussian(next) * 0.014]);
        labels.push(b);
      }
    });
    return { pts, labels };
  };
  const src = make(-0.5, 0, 0, -1, 0, 0);
  const tgt = make(0.5, 0.04, 0.35, 2, Math.PI * 0.55, Math.PI * 1.15);
  const X = src.pts, Y = tgt.pts;
  const labelsX = src.labels, labelsY = tgt.labels;
  for (let k = 0; k < 6; k++) {
    Y.push([0.5 + gaussian(next) * 0.07, 0.04 + gaussian(next) * 0.07]);
    labelsY.push(3);
  }
  const ring = (cx, cy, r) => Array.from({ length: 60 }, (_, i) => [cx + Math.cos((i / 59) * Math.PI * 2) * r, cy + Math.sin((i / 59) * Math.PI * 2) * r]);
  return {
    name: 'Tissue slices',
    note: 'Three cell-type layers on each slice. The second slice is turned, its outer layer is torn, and a few immune cells have moved in.',
    legend: { source: 'slice 1', target: 'slice 2 (turned, outer layer torn, + immune cells)' },
    marker: { source: 'circle', target: 'diamond' },
    scene: [0.14, 0.27, 0.40].flatMap((r) => [
      { type: 'polyline', points: ring(-0.5, 0, r), stroke: 'ink', width: 0.8, alpha: 0.12 },
      { type: 'polyline', points: ring(0.5, 0.04, r), stroke: 'ink', width: 0.8, alpha: 0.12 }
    ]).concat([
      { type: 'label', at: [-0.5, 0.52], text: 'layers I · II · III' },
      { type: 'label', at: [0.5, 0.56], text: 'layers I · II · III' }
    ]),
    X, Y, a: uniformMass(X.length), b: uniformMass(Y.length), labelsX, labelsY
  };
}

/* ------------------------------------------------------------------ */
/* 3D                                                                  */
/* ------------------------------------------------------------------ */

/**
 * A helix and a damaged copy of it, in three dimensions.
 *
 * The target is the same helix turned about the vertical axis and nudged, with
 * its top turn missing and a stray clump added off to one side. Neighbouring
 * turns sit close together in 3D, so a distance cutoff can admit matches along
 * the curve while refusing the jump between turns — which is exactly the case
 * supervised OT is for.
 */
export function helix3d(n = 64) {
  const next = rng(41);
  const turns = 3, radius = 0.55, height = 1.6;
  const point = (t, rot, shift) => {
    const ang = t * turns * Math.PI * 2 + rot;
    return [
      Math.cos(ang) * radius + shift[0] + gaussian(next) * 0.02,
      (t - 0.5) * height + shift[1] + gaussian(next) * 0.02,
      Math.sin(ang) * radius + shift[2] + gaussian(next) * 0.02
    ];
  };
  const X = Array.from({ length: n }, (_, i) => point(i / (n - 1), 0, [0, 0, 0]));
  const keep = Math.round(n * 0.78);
  const Y = Array.from({ length: keep }, (_, i) => point(i / (n - 1) + 0.004, 0.7, [0.18, 0.08, 0.1]));
  for (let k = 0; k < 10; k++) {
    Y.push([0.95 + gaussian(next) * 0.07, 0.55 + gaussian(next) * 0.07, -0.85 + gaussian(next) * 0.07]);
  }
  return {
    name: 'Helix',
    note: 'The target helix is missing its top turn and carries a stray clump. Balanced OT must serve both anyway.',
    X, Y, a: uniformMass(X.length), b: uniformMass(Y.length),
    labelsX: X.map(() => 0), labelsY: Y.map((_, i) => (i < keep ? 0 : 1))
  };
}

export const DATASETS = {
  bakeries, heartStar, tissue, extraCluster, outlier,
  twoBlobs, imbalance, crossedClasses, rings
};

/** What the pickers offer, in order. The rest stay available by key. */
export const DATASET_ORDER = ['bakeries', 'heartStar', 'tissue', 'extraCluster', 'outlier'];

export function makeDataset(key, n) {
  const fn = DATASETS[key] || twoBlobs;
  return fn(n);
}

/**
 * Move one point of a dataset (used by the draggable outlier figure) without
 * disturbing anything else.
 */
export function withMovedPoint(data, side, index, xy) {
  const copy = { ...data, X: data.X.map((p) => p.slice()), Y: data.Y.map((p) => p.slice()) };
  (side === 'source' ? copy.X : copy.Y)[index] = [xy[0], xy[1]];
  return copy;
}
