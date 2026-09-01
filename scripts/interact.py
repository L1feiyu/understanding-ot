"""Interaction smoke test: click every segmented control, move every slider,
toggle every checkbox and the theme, at desktop and mobile widths, and fail on
any console error or non-finite readout."""

import sys
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8123/index.html"

def run(pw, width, height, label):
    browser = pw.chromium.launch(args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": width, "height": height})
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

    page.goto(URL, wait_until="networkidle")
    for _ in range(20):
        page.mouse.wheel(0, 900)
        page.wait_for_timeout(120)
    page.wait_for_timeout(2000)

    segs = page.query_selector_all("button.seg")
    checks = page.query_selector_all(".control-check input")
    sliders = page.query_selector_all('input[type="range"]')
    print(f"  [{label}] {len(segs)} segments, {len(sliders)} sliders, {len(checks)} checkboxes")

    for b in segs:
        try:
            b.scroll_into_view_if_needed()
            b.click(timeout=4000)
            page.wait_for_timeout(160)
        except Exception as exc:
            errors.append(f"segment click failed: {exc}")

    for c in checks:
        try:
            c.scroll_into_view_if_needed()
            c.click(timeout=4000)
            page.wait_for_timeout(200)
            c.click(timeout=4000)
            page.wait_for_timeout(200)
        except Exception as exc:
            errors.append(f"checkbox failed: {exc}")

    # Re-query: switching method rebuilds the per-method parameter slider, so
    # handles captured before those clicks are detached by now.
    sliders = page.query_selector_all('input[type="range"]')
    print(f"  [{label}] {len(sliders)} sliders after control interaction")

    # Drive each slider to both ends and back — the extremes are where solvers break.
    for s in sliders:
        try:
            if not s.evaluate("el => el.isConnected"):
                continue
            s.scroll_into_view_if_needed()
            for frac in (0.0, 1.0, 0.5):
                page.evaluate(
                    """([el, f]) => {
                        const lo = parseFloat(el.min), hi = parseFloat(el.max);
                        el.value = String(lo + (hi - lo) * f);
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                    }""",
                    [s, frac],
                )
                page.wait_for_timeout(220)
        except Exception as exc:
            errors.append(f"slider failed: {exc}")

    page.wait_for_timeout(800)

    # No readout should ever show NaN / Infinity.
    bad = page.evaluate(
        """() => [...document.querySelectorAll('.readout-value')]
             .map(n => n.textContent)
             .filter(t => /NaN|Infinity|undefined/.test(t))"""
    )
    if bad:
        errors.append(f"bad readouts: {bad}")

    page.click("#theme-toggle")
    page.wait_for_timeout(1200)

    browser.close()
    return errors

with sync_playwright() as pw:
    all_errors = []
    for w, h, label in [(1280, 900, "desktop"), (390, 780, "mobile")]:
        all_errors += [f"{label}: {e}" for e in run(pw, w, h, label)]

print("errors:", len(all_errors))
for e in all_errors[:30]:
    print("  ", e)
sys.exit(1 if all_errors else 0)
