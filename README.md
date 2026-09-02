# Understanding Relaxed Optimal Transport

An interactive article comparing **partial**, **unbalanced** and **supervised** optimal transport
against the balanced baseline — in the spirit of
[Understanding UMAP](https://pair-code.github.io/understanding-umap/) by Google PAIR.

Every figure solves its transport problem live in the browser. Nothing is precomputed, and there is
no build step: the site is plain ES modules, so `index.html` works from a static server or straight
off GitHub Pages.

**→ [Read the article](https://L1feiyu.github.io/understanding-ot/)**

---

## The short version

Classical optimal transport requires `P·1 = a` and `Pᵀ·1 = b`: all supply ships, all demand is met.
Three relaxations drop that requirement in three ways, and they are usually presented as three
separate methods with three separate algorithms. They are not. Entropically regularised, all four
problems are **KL projections of the same Gibbs kernel** (or proximal steps in the same geometry),
differing only in the scaling each applies to the rows and columns:

| Problem | Constraint / penalty on `P·1` | Row scaling of `P` (KL projection or prox) |
|---|---|---|
| Balanced | `P·1 = a` | `diag(a / (P·1))` — the projection |
| Partial | `P·1 ≤ a`, plus `⟨P,1⟩ = s` | `diag(min(1, a / (P·1)))` — with Dykstra corrections |
| Unbalanced | `τ·KL(P·1 ‖ a)` | `diag((a / (P·1))^(τ/(τ+ε)))` — a fractional step |
| Supervised | `γ·‖a − P·1‖₁`, `P·1 ≤ a`, `C[i][j] = ∞` past a cutoff | `u ← min(a/(K·v), e^(γ/ε))` — the projection, capped |

with `K = exp(−C/ε)` the Gibbs kernel and `P = diag(u)·K·diag(v)`. This is the Bregman-projection
view of Benamou, Carlier, Cuturi, Nenna and Peyré; the dual potentials of the log-domain
implementation are `f = ε·log u`.

What the article argues from there:

1. **Partial OT budgets mass; supervised OT bounds distance.** With nothing forbidden the two
   coincide — sOT is partial OT with an extra maximisation over the transported mass `θ`, and both
   collapse to balanced OT. But once the cutoff is in force they are different objects. Partial OT
   says how *much* moves and will use routes of any length to hit the quota; supervised OT says how
   *far* mass may travel and lets the amount follow. Every figure reports **longest route** — the
   costliest pair carrying ≥ 1% of its source's mass — alongside mass and cost, because at equal
   transported mass the two plans have nearly equal cost and very different longest routes. Cost is
   a sum and hides a few long routes; the cutoff is a maximum and forbids them.
2. **`γ` is not a hyperparameter.** A route is used exactly when `C[i][j] < 2γ`, so with `C`
   normalised to `(0,1)` and `γ = 2` the threshold sits at 4, above every cost, and the price never
   binds. `γ` is set large enough to get out of the way; **the knob is the cutoff on `C`**.
3. **The relaxation is a precondition, not the content.** With forbidden routes, partial OT at a
   prescribed `θ` can have an empty feasible set; sOT stays well posed because `θ = 0` is always
   available. Letting the mass follow from what is permitted is what keeps the question answerable.

## Running it



```bash
git clone https://github.com/L1feiyu/understanding-ot.git
cd understanding-ot
npm start          # or: python3 -m http.server 8123
```

Then open <http://localhost:8123/>. There is nothing to install — no bundler, no dependencies.

## The solver

`src/lib/ot/solvers.js` is a dependency-free ES module. It works for Node and the browser alike.

```js
import { solve, squaredEuclidean, normalizeCost, applyBlocking, diagnostics }
  from './src/lib/ot/solvers.js';

const C = normalizeCost(squaredEuclidean(X, Y));   // X, Y are arrays of [x, y]

solve({ C, a, b, method: 'balanced',   eps: 0.01 });
solve({ C, a, b, method: 'partial',    eps: 0.01, s: 0.6 });      // move 60% of the mass
solve({ C, a, b, method: 'unbalanced', eps: 0.01, tau: 0.25 });   // KL penalty
solve({ C, a, b, method: 'supervised', eps: 0.01, cutoff: 0.3 });  // routes past 0.3 forbidden

// `cutoff` is the usual supervision: gamma defaults to SOT_GAMMA (= 2), and every
// route costing more than the cutoff becomes unusable, so mass with no partner
// inside it simply stays put.

// Any other rule works the same way — here, forbid cross-class transport:
const Cb = applyBlocking(C, X.length, Y.length, (i, j) => labelsX[i] !== labelsY[j]);
const { P } = solve({ C: Cb, a, b, method: 'supervised', eps: 0.01 });
```

`solve` returns `{ P, f, g, n, m, iterations, converged, residual }`, with `P` a row-major
`Float64Array` of length `n*m`. Everything runs in the log domain, so small `ε` does not overflow.
Forbidden routes are a large finite cost rather than a true `Infinity`, which would poison the
log-sum-exp.

Three implementation points that turned out to matter:

- **Convergence is judged on the KKT residual** of each method's fixed point, not on the potentials
  or the plan settling. With a cap in play Sinkhorn converges *geometrically* to a false plateau,
  sits there with `|ΔP| ≈ 1e-12`, then a potential crosses the cap and the plan jumps to a different
  regime. Any "it stopped changing" rule reports the plateau with full confidence; the residual stays
  around `1e-2` on it. (Watching total mass is worse still: the row step pins it to `|a|` and the
  column step to `|b|`, so it looks stable from iteration one.)
- **Over-relaxation** (`omega`, default 1.7) roughly halves the iteration count on the balanced and
  supervised updates and lands on the same fixed point. Pass `omega: 1` for textbook Sinkhorn.
- **The figures solve in a Web Worker** (`src/lib/ot/client.js`) with warm starts, so a slow case —
  two sources fighting over one permitted target can take thousands of iterations — never freezes a
  slider. When the iteration cap is hit the readout says so and shows the residual, which in those
  cases is around `1e-6`: accurate far beyond anything a figure can display.

`diagnostics(P, C, a, b, n, m)` returns transported mass, transport cost, the longest route used,
per-point marginals and the ℓ¹ marginal violations that the figures display. `longestRoute(P, C, a,
n, m, share = 0.01)` is the costliest pair carrying at least `share` of its source's mass — an
entropic plan puts a trace of mass everywhere, so a raw max would only report noise.

### Cost scale matters

`ε`, `τ`, `γ` and the cutoff are all measured in the units of `C`, so call `normalizeCost` first (it
divides by the largest finite entry) — supervised OT in particular assumes it. With `C ∈ (0,1]` a
route is used exactly when its cost is below `2γ`; `γ = 2` puts that at 4, above everything, which is
why the cutoff rather than `γ` is what you tune. Apply the cutoff *after* normalising: the sentinel
cost standing in for `∞` is not rescaled.

## Verification

The JavaScript is checked against NumPy references, which in turn reproduce the published
algorithms:

```bash
npm run verify          # regenerate fixtures from NumPy, then run the JS suite
npm test                # JS suite only
```

- `reference/ot_reference.py` implements each method in NumPy, written to reproduce POT's
  algorithms. **It does not import POT** — it is a re-implementation, so it is evidence about the
  algorithm, not about the library.
- `reference/pot_reference.py` closes that gap: it runs the **actual POT library** and compares. It
  also regenerates the fixtures from POT, so the JavaScript is tested against genuine library
  output rather than against a re-implementation:

  ```bash
  pip install POT
  npm run verify:pot        # compare the references against installed POT
  npm run fixtures:pot      # regenerate fixtures.json from POT, then: npm test
  ```

  Run this at least once. It is not in CI because it needs a PyPI install, and it was never executed
  by the original author of this repository — see the provenance note below.
- Supervised OT uses the authors' own `perform_sOT_log` **verbatim**, unmodified.
- `reference/validate.py` cross-checks the log-domain forms against the classical multiplicative
  ones, pins the limiting cases, and emits `reference/fixtures.json`.
- `test/solvers.test.js` runs the JS solvers against those fixtures plus property tests: exact
  marginals for balanced OT, `P·1 ≤ a` for partial, monotone mass in `τ` and in the cutoff, `γ`
  provably saturated at `SOT_GAMMA` (varying it changes nothing), exactly zero mass crossing the
  cutoff, complementary slackness on the ceiling, and the partial ↔ supervised equivalence.

### Provenance — what is whose code

Worth being exact about, because "validated against POT" and "uses POT" are different claims:

| Piece | Where it came from |
|---|---|
| `perform_sOT_log` in `ot_reference.py` | The sOT authors' released code, **verbatim and unmodified**. |
| The rest of `ot_reference.py` | Written here to reproduce POT's algorithms from the papers. Not POT's code. |
| `pot_reference.py` | Calls the **real POT library**. This is the only place POT actually runs. |
| `src/lib/ot/solvers.js` | Written here. A re-implementation of the same updates, sharing one log-sum-exp kernel across all four methods, tested against the fixtures. |

The JavaScript is therefore a port validated against reference output, not a translation of anyone's
source. Differences that are deliberate: the sOT reference adds `1e-20` inside the raw sum where the
JS adds a tiny constant inside a max-shifted log-sum-exp, and the JS exits early on convergence
instead of running a fixed `niter`.

`pot_reference.py` could not be executed in the environment it was written in (no PyPI access), so
it was first run downstream — and it immediately caught a real error in `ot_reference.py`; see the
correction note above. It has since been checked against **POT 0.9.5**, where the references agree
on all three methods. Two things it settled that were otherwise guesswork:

- POT's unbalanced solver takes a `reg_type` argument, and this repository's convention matches
  `reg_type='entropy'`, not the `'kl'` default. The script probes both and reports which one fits.
- POT's partial solver carries Dykstra corrections, which is what the reference now reproduces.

Re-run it after any change to the reference implementations.

`scripts/screenshot.py` renders the article in headless Chromium, fails on any console error or
blank canvas, and captures both themes. It needs Playwright (`pip install playwright`).

### Why Dykstra's corrections matter

Entropic partial OT is a KL projection onto the intersection of three convex sets — `P·1 ≤ a`,
`Pᵀ·1 ≤ b`, and `⟨P,1⟩ = s`. Dykstra's algorithm reaches that projection by carrying a correction
factor for each set. Drop the corrections and you get plain alternating projection, which still
lands somewhere *feasible* — the marginals hold and the mass is right — but on a plan with a
strictly larger entropic objective.

POT's `ot.partial.entropic_partial_wasserstein` carries the corrections (`q1`, `q2`, `q3`), and so
does the default solver here; the two agree to ~1e-14. `algorithm: 'alternating'` runs the
uncorrected version, and exists only as the contrast that makes the corrections' role visible.

> **Correction.** An earlier version of this README claimed the opposite — that POT's routine
> omitted the corrections and returned suboptimal plans. That was wrong, and the error is
> instructive. The NumPy "reference" it was measured against was itself missing the corrections, so
> the comparison was between a correct solver and a broken stand-in, with the conclusion attributed
> to POT. Running `npm run verify:pot` against the real library is what surfaced it: one mass level
> matched to 1e-15 while another was off by 15%, which is the signature of two different algorithms
> rather than a convergence problem. Validating against your own re-implementation proves nothing
> about the library it imitates.

A second trap worth naming: when comparing two plans, rank them by the objective actually being
minimised, `⟨P,C⟩ + ε·Σ(P log P − P)`, not by `⟨P,C⟩` alone. A regularised solver may accept a
slightly costlier plan in exchange for enough entropy, so transport cost on its own can order two
plans the wrong way round — it does exactly that on one of the cases in `reference/validate.py`.

## Layout

```
index.html                  the article (math in LaTeX, rendered by KaTeX)
vendor/katex/               KaTeX 0.16 (MIT), vendored so the page has no CDN dependency
src/global.css              colour roles for both themes
src/main.js                 mounts figures lazily, handles the theme toggle
src/lib/ot/solvers.js       the solvers — the only file most people will want
src/lib/datasets.js         seeded scenes: bakeries & cafés, heart → star, tissue slices, a 3D helix
src/lib/plot.js             canvas primitives: transport plot, coupling matrix, line chart
src/lib/palette.js          colour roles, sequential ramp
src/lib/ui.js               sliders, segmented controls, readouts, tooltips
src/lib/ot/worker.js        solver in a Web Worker
src/lib/ot/client.js        latest-wins request lane per figure, sync fallback
src/figures/*.js            one file per figure (three.js is the rotating 3D helix)
reference/ot_reference.py   NumPy re-implementations (no POT import)
reference/pot_reference.py  runs the real POT library and compares / regenerates fixtures
reference/validate.py       cross-checks + emits fixtures.json
test/                       node:test suite
scripts/screenshot.py       headless render + regression check
```

## Deploying

`.github/workflows/deploy.yml` publishes the repository to GitHub Pages on every push to `main`.
Because there is no build step it uploads the tree as-is. Enable it under
**Settings → Pages → Source → GitHub Actions**, then update the article URL at the top of this file.

## Adding a method

1. Add a projection branch to `project()` in `src/lib/ot/solvers.js` and an entry in `METHOD_META`.
2. Add a NumPy reference to `reference/ot_reference.py` and a fixture case in `validate.py`.
3. Run `npm run verify`.

The figures read the method list from `METHOD_META`, so they pick it up automatically.

## References

- Zixuan Cang, Qing Nie, Yanxiang Zhao. *Supervised Optimal Transport.*
  SIAM J. Appl. Math. 82(5), 1851–1877, 2022.
  [doi:10.1137/22M1469171](https://epubs.siam.org/doi/10.1137/22M1469171) ·
  [arXiv:2206.13410](https://arxiv.org/abs/2206.13410)
- L. Chizat, G. Peyré, B. Schmitzer, F.-X. Vialard. *Scaling algorithms for unbalanced optimal
  transport problems.* Math. Comp. 2018. [arXiv:1607.05816](https://arxiv.org/abs/1607.05816)
- J.-D. Benamou, G. Carlier, M. Cuturi, L. Nenna, G. Peyré. *Iterative Bregman projections for
  regularized transportation problems.* SIAM J. Sci. Comput. 2015.
  [arXiv:1412.5154](https://arxiv.org/abs/1412.5154)
- M. Cuturi. *Sinkhorn distances.* NeurIPS 2013. [arXiv:1306.0895](https://arxiv.org/abs/1306.0895)
- L. Chapel, M. Alaya, G. Gasso. *Partial optimal transport with applications on positive-unlabeled
  learning.* NeurIPS 2020. [arXiv:2002.08276](https://arxiv.org/abs/2002.08276)
- G. Peyré, M. Cuturi. *Computational Optimal Transport.* FnT ML 2019.
  [arXiv:1803.00567](https://arxiv.org/abs/1803.00567)
- R. Flamary et al. *POT: Python Optimal Transport.* JMLR 2021. [pythonot.github.io](https://pythonot.github.io/)

## Licence

MIT — see [LICENSE](LICENSE).

The `perform_sOT_log` reference implementation in `reference/ot_reference.py` is reproduced from the
Supervised Optimal Transport authors' released code and remains theirs; if you use it, cite the
paper.
