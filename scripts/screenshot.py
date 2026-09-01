"""Render the article in headless Chromium, capture screenshots, and fail loudly
on any console error. Run with the dev server up:

    python3 -m http.server 8123 &
    python3 scripts/screenshot.py
"""
import sys
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8123/index.html"
OUT = "/tmp/shots"

FIGURES = ["fig-comparison", "fig-coupling", "fig-projections", "fig-mass", "fig-blocking"]

with sync_playwright() as pw:
    browser = pw.chromium.launch(args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 1200, "height": 1000}, device_scale_factor=2)

    errors, logs = [], []
    page.on("console", lambda m: (errors if m.type == "error" else logs).append(m.text))
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

    page.goto(URL, wait_until="networkidle")
    page.wait_for_timeout(1500)

    # Scroll through so the lazily-mounted figures all build.
    for y in range(0, 12000, 700):
        page.mouse.wheel(0, 700)
        page.wait_for_timeout(180)
    page.wait_for_timeout(2500)

    import os
    os.makedirs(OUT, exist_ok=True)
    page.screenshot(path=f"{OUT}/full.png", full_page=True)

    for fid in FIGURES:
        node = page.query_selector(f"#{fid}")
        if node is None:
            errors.append(f"missing figure slot #{fid}")
            continue
        box = node.bounding_box()
        if not box or box["height"] < 60:
            errors.append(f"#{fid} rendered with height {box['height'] if box else 'none'}")
            continue
        node.screenshot(path=f"{OUT}/{fid}.png")

    # Every canvas should contain actual marks, not a blank rectangle.
    blank = page.evaluate("""() => {
      const out = [];
      for (const c of document.querySelectorAll('canvas')) {
        const ctx = c.getContext('2d');
        const d = ctx.getImageData(0, 0, c.width, c.height).data;
        let ink = 0;
        for (let i = 3; i < d.length; i += 40) if (d[i] > 8) ink++;
        if (ink < 20) out.push((c.className || 'canvas') + ' @' + c.width + 'x' + c.height);
      }
      return out;
    }""")

    # Dark mode pass.
    page.click("#theme-toggle")
    page.wait_for_timeout(1800)
    page.screenshot(path=f"{OUT}/dark.png", full_page=True)

    browser.close()

print("console errors:", len(errors))
for e in errors[:25]:
    print("  ERROR:", e)
print("blank canvases:", blank if blank else "none")
print("screenshots in", OUT)
sys.exit(1 if (errors or blank) else 0)
