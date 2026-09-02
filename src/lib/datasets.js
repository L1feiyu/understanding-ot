/**
 * Toy source/target pairs used throughout the article.
 *
 * Each dataset returns { X, Y, a, b, labelsX, labelsY, name, note } where X and Y
 * are arrays of [x, y] in roughly [-1, 1]^2, and a / b are the marginal masses.
 * Labels exist so the supervision figure can forbid cross-class couplings.
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
  twoBlobs, extraCluster, outlier, imbalance, crossedClasses, rings
};

export const DATASET_ORDER = ['twoBlobs', 'extraCluster', 'imbalance', 'outlier', 'crossedClasses', 'rings'];

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
