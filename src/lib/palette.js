/**
 * Colour roles for every figure in the article.
 *
 * Two categorical hues carry identity throughout: the SOURCE distribution is
 * blue and the TARGET distribution is orange, in every figure, always. Method
 * identity is carried by panel titles rather than by a third and fourth hue —
 * four categorical colours in a scatter would not clear the colour-blind
 * separation floors, and there is no reason to spend hues on something a label
 * says better.
 *
 * Values are read from CSS custom properties so light and dark mode stay in one
 * place (src/global.css), with these as the fallback.
 */

const FALLBACK = {
  source: '#2a78d6',
  target: '#eb6834',
  accent: '#1baf7a',
  surface: '#fcfcfb',
  ink: '#0b0b0b',
  inkSecondary: '#52514e',
  inkMuted: '#78766f',
  grid: '#e6e4df',
  flow: '#52514e'
};

let cache = null;
let cacheKey = '';

/** Resolve the palette from CSS custom properties, re-reading on theme change. */
export function palette() {
  if (typeof document === 'undefined') return { ...FALLBACK };
  const key = document.documentElement.dataset.theme || 'auto';
  if (cache && cacheKey === key) return cache;
  const cs = getComputedStyle(document.documentElement);
  const read = (name, fb) => (cs.getPropertyValue(name) || '').trim() || fb;
  cache = {
    source: read('--source', FALLBACK.source),
    target: read('--target', FALLBACK.target),
    accent: read('--accent', FALLBACK.accent),
    surface: read('--surface-1', FALLBACK.surface),
    ink: read('--text-primary', FALLBACK.ink),
    inkSecondary: read('--text-secondary', FALLBACK.inkSecondary),
    inkMuted: read('--text-muted', FALLBACK.inkMuted),
    grid: read('--grid', FALLBACK.grid),
    flow: read('--flow', FALLBACK.flow)
  };
  cacheKey = key;
  return cache;
}

export function invalidatePalette() {
  cache = null;
}

/** Parse "#rgb"/"#rrggbb" into [r,g,b]; returns null for anything else. */
export function parseHex(hex) {
  let h = String(hex).trim().replace('#', '');
  if (h.length === 3) h = h.split('').map((c) => c + c).join('');
  if (h.length !== 6 || /[^0-9a-f]/i.test(h)) return null;
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16)
  ];
}

export function withAlpha(hex, alpha) {
  const rgb = parseHex(hex);
  if (!rgb) return hex;
  return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${Math.max(0, Math.min(1, alpha))})`;
}

export function mix(hexA, hexB, t) {
  const a = parseHex(hexA), b = parseHex(hexB);
  if (!a || !b) return hexA;
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

/**
 * Single-hue sequential ramp used by the coupling-matrix heatmap: light (near
 * zero) to dark (large). One hue, light to dark — never a rainbow.
 */
const BLUE_RAMP = [
  '#f4f8fe', '#cde2fb', '#9ec5f4', '#6da7ec',
  '#3987e5', '#2a78d6', '#256abf', '#184f95', '#0d366b'
];

export function sequential(t) {
  const x = Math.max(0, Math.min(1, t)) * (BLUE_RAMP.length - 1);
  const i = Math.min(BLUE_RAMP.length - 2, Math.floor(x));
  return mix(BLUE_RAMP[i], BLUE_RAMP[i + 1], x - i);
}

export const RAMP_STOPS = BLUE_RAMP;
