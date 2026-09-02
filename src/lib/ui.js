/**
 * Small DOM helpers: the figure shell, the control row, and the hover layer.
 * No framework — every figure is a function that fills a <figure> element.
 */

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else if (v != null) node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c) node.appendChild(c);
  return node;
}

/** Controls sit in one row above the plots, never scattered among them. */
export function controlRow(children) {
  return el('div', { class: 'controls' }, children);
}

/**
 * A labelled slider. `format` renders the live value; `log: true` makes the
 * track logarithmic, which is what τ and γ need — both are interesting across
 * three orders of magnitude and useless on a linear track.
 */
export function slider({ label, min, max, step, value, log = false, format, onInput }) {
  const toSlider = (v) => (log ? Math.log(v) : v);
  const fromSlider = (v) => (log ? Math.exp(v) : v);
  const lo = toSlider(min), hi = toSlider(max);
  const input = el('input', {
    type: 'range',
    min: String(lo),
    max: String(hi),
    step: String(log ? (hi - lo) / 240 : step),
    value: String(toSlider(value))
  });
  const out = el('output', { text: format ? format(value) : String(value) });
  const wrap = el('label', { class: 'control control-slider' }, [
    el('span', { class: 'control-label' }, [el('span', { text: label }), out]),
    input
  ]);
  input.addEventListener('input', () => {
    const v = fromSlider(parseFloat(input.value));
    out.textContent = format ? format(v) : String(v);
    onInput(v);
  });
  wrap.setValue = (v) => {
    input.value = String(toSlider(v));
    out.textContent = format ? format(v) : String(v);
  };
  wrap.getValue = () => fromSlider(parseFloat(input.value));
  return wrap;
}

/** A segmented control — clearer than a <select> for three or four options. */
export function segmented({ label, options, value, onChange }) {
  const buttons = options.map((o) =>
    el('button', {
      type: 'button',
      class: o.value === value ? 'seg is-active' : 'seg',
      text: o.label,
      title: o.title || '',
      onclick: () => {
        if (group.value === o.value) return;
        group.value = o.value;
        [...group.children].forEach((b, i) =>
          b.classList.toggle('is-active', options[i].value === o.value));
        onChange(o.value);
      }
    })
  );
  const group = el('div', { class: 'segmented', role: 'group' }, buttons);
  group.value = value;
  return el('label', { class: 'control' }, [
    label ? el('span', { class: 'control-label' }, [el('span', { text: label })]) : null,
    group
  ]);
}

export function checkbox({ label, checked, onChange }) {
  const input = el('input', { type: 'checkbox' });
  input.checked = !!checked;
  input.addEventListener('change', () => onChange(input.checked));
  return el('label', { class: 'control control-check' }, [input, el('span', { text: label })]);
}

/** A key/value readout strip. Values update in place; nothing is re-created. */
export function readout(fields) {
  const cells = {};
  const node = el('div', { class: 'readout' }, fields.map((f) => {
    const v = el('span', { class: 'readout-value', text: '—' });
    cells[f.key] = v;
    return el('div', { class: 'readout-item', title: f.title || '' }, [
      el('span', { class: 'readout-label', text: f.label }), v
    ]);
  }));
  node.update = (values) => {
    for (const [k, v] of Object.entries(values)) if (cells[k]) cells[k].textContent = v;
  };
  return node;
}

/** Shared hover layer: one tooltip element per figure, positioned on demand. */
export function tooltip(parent) {
  const node = el('div', { class: 'tooltip' });
  parent.appendChild(node);
  return {
    node,
    show(html, x, y) {
      node.innerHTML = html;
      node.style.display = 'block';
      const box = parent.getBoundingClientRect();
      const w = node.offsetWidth, h = node.offsetHeight;
      node.style.left = `${Math.max(4, Math.min(box.width - w - 4, x - w / 2))}px`;
      node.style.top = `${Math.max(4, y - h - 12)}px`;
    },
    hide() { node.style.display = 'none'; }
  };
}

/** Legend — always present when two things share a plot; identity is never colour alone. */
export function legend(items) {
  return el('div', { class: 'legend' }, items.map((i) =>
    el('span', { class: 'legend-item' }, [
      el('span', { class: 'legend-swatch', style: `background:${i.colour}` }),
      el('span', { text: i.label })
    ])
  ));
}

/**
 * Run `fn` at most once per animation frame, and never while the tab is hidden.
 * Solvers take tens of milliseconds, so dragging a slider must not queue them up.
 */
export function scheduler(fn) {
  let pending = false;
  return function schedule(...args) {
    if (pending) { schedule.latest = args; return; }
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      const a = schedule.latest || args;
      schedule.latest = null;
      fn(...a);
    });
  };
}

/** Re-render figures when the container resizes or the theme flips. */
export function onResize(node, fn) {
  if (typeof ResizeObserver === 'undefined') {
    window.addEventListener('resize', () => fn());
    return;
  }
  const ro = new ResizeObserver(() => fn());
  ro.observe(node);
}

export function figureWidth(node, max = 960) {
  const w = node.getBoundingClientRect().width || max;
  return Math.max(280, Math.min(max, w));
}

export function fmt(v, digits = 3) {
  if (!Number.isFinite(v)) return '—';
  return v.toFixed(digits);
}

export function pct(v) {
  return `${(v * 100).toFixed(1)}%`;
}

/** Solver status for a readout: honest about the iteration cap, calm about a tiny residual. */
export function solverStatus(res) {
  if (res.converged) return `converged · ${res.iterations} it`;
  const r = Number.isFinite(res.residual) ? res.residual.toExponential(0) : '?';
  return `${res.iterations} it · residual ${r}`;
}
