#!/usr/bin/env python3
"""The README's and the demonstrations document's figures (docs/figures/execution-paths, decode-throughput and
feed-packing .svg, and their -dark variants), drawn from the committed evidence.

    python3 tools/g17figures.py            write all figures, light and dark
    python3 tools/g17figures.py --check    exit 1 if a committed figure differs from what would be written

Hand-written SVG rather than a plotting library, so text, spacing and colour are exact. The decode chart reads its
numbers from evidence/g17-matched-study-v1/decode-shared-history.json, so a re-measured file moves the figure with it;
the two diagrams encode the execution paths and one tensor-register example (technical reference section 10.5).

Each figure comes in a light and a dark variant with a transparent background, so it sits on the page's own surface;
the README and the demonstrations document pick one with <picture> and prefers-color-scheme. Colour does one job per figure: AGXForge
(categorical slot 1, blue) against Apple (slot 2, orange, or neutral for Apple's infrastructure) and the mlx-lm
baseline (neutral). The two-slot palette was checked with the dataviz validator in both modes (CVD dE >= 24.7, normal
vision dE >= 31.8, contrast >= 3:1). Text always wears text ink, never a series colour.
"""
import html
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")
DECODE = os.path.join(ROOT, "evidence", "g17-matched-study-v1", "decode-shared-history.json")
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

THEMES = {
    "light": dict(ink="#1f2328", ink2="#57606a", muted="#8c959f", grid="#d8dee4",
                  ours="#2a78d6", ours_fill="#ddebfb", ours_edge="#2a78d6",
                  apple="#eb6834", base="#8f8e89",
                  infra_fill="#f2f1ee", infra_edge="#b9b8b2", focus="#1f2328"),
    "dark": dict(ink="#e6edf3", ink2="#9ba7b4", muted="#7d8590", grid="#30363d",
                 ours="#3987e5", ours_fill="#12305a", ours_edge="#3987e5",
                 apple="#d95926", base="#7a7975",
                 infra_fill="#24262a", infra_edge="#4d5057", focus="#e6edf3"),
}


def svg(width, height, body, title):
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" '
            'font-family="%s" role="img" aria-label="%s">\n<title>%s</title>\n%s\n</svg>\n'
            % (width, height, width, height, html.escape(FONT, quote=True), html.escape(title, quote=True),
               html.escape(title), body))


def text(x, y, s, size=13, fill="#000", weight=400, anchor="start"):
    return ('<text x="%g" y="%g" font-size="%g" font-weight="%d" fill="%s" text-anchor="%s">%s</text>'
            % (x, y, size, weight, fill, anchor, html.escape(s)))


def box(x, y, w, h, fill, edge, r=8, width=1.25):
    return ('<rect x="%g" y="%g" width="%g" height="%g" rx="%g" fill="%s" stroke="%s" stroke-width="%g"/>'
            % (x, y, w, h, r, fill, edge, width))


def arrow(points, color, marker):
    d = "M %g %g " % points[0] + " ".join("L %g %g" % p for p in points[1:])
    return ('<path d="%s" fill="none" stroke="%s" stroke-width="1.5" stroke-linejoin="round" marker-end="url(#%s)"/>'
            % (d, color, marker))


def marker(name, color):
    return ('<marker id="%s" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
            '<path d="M0 0 L10 5 L0 10 z" fill="%s"/></marker>' % (name, color))


def labelled_box(t, x, y, w, h, title, sub, ours, bold=False):
    fill, edge = (t["ours_fill"], t["ours_edge"]) if ours else (t["infra_fill"], t["infra_edge"])
    parts = [box(x, y, w, h, fill, edge)]
    if sub:
        parts.append(text(x + w / 2, y + h / 2 - 3, title, 14, t["ink"], 600 if bold else 500, "middle"))
        parts.append(text(x + w / 2, y + h / 2 + 15, sub, 12, t["ink2"], 400, "middle"))
    else:
        parts.append(text(x + w / 2, y + h / 2 + 5, title, 14, t["ink"], 600 if bold else 500, "middle"))
    return "\n".join(parts)


def paths(t):
    """Two routes for the same native code: through Metal (every performance result) and below Metal."""
    a = t["ink2"]
    b = ["<defs>%s</defs>" % marker("ah", a)]
    b.append(labelled_box(t, 16, 28, 200, 58, "Python kernel builders", "AGXForge IR, directly", True))
    b.append(labelled_box(t, 16, 112, 200, 58, "Metal source", "Apple front end → AIR", False))
    b.append(labelled_box(t, 262, 60, 196, 78, "AGXForge compiler", "selection · allocation · encoding", True, True))
    b.append(text(360, 158, "+ object, metadata, container", 12, t["ink2"], 400, "middle"))
    b.append(labelled_box(t, 502, 70, 150, 58, "Native G17 code", "", True, True))
    b.append(labelled_box(t, 700, 18, 284, 62, "Metal: pipelines, command buffers",
                          "through Metal · every performance result", False))
    b.append(labelled_box(t, 700, 118, 244, 62, "AGXForge runtime", "below Metal · launch state, Submit records", True))
    b.append(labelled_box(t, 262, 222, 722, 48, "Apple IOGPU framework · kernel driver · firmware", "", False))
    b.append(labelled_box(t, 262, 290, 722, 44, "M5 GPU and its tensor units", "", False, True))
    b.append(arrow([(216, 57), (240, 57), (240, 92), (260, 92)], a, "ah"))
    b.append(arrow([(216, 141), (240, 141), (240, 108), (260, 108)], a, "ah"))
    b.append(arrow([(458, 99), (500, 99)], a, "ah"))
    b.append(arrow([(652, 92), (676, 92), (676, 49), (698, 49)], a, "ah"))
    b.append(arrow([(652, 106), (676, 106), (676, 149), (698, 149)], a, "ah"))
    b.append(arrow([(964, 80), (964, 220)], a, "ah"))
    b.append(arrow([(822, 180), (822, 220)], a, "ah"))
    b.append(arrow([(623, 270), (623, 288)], a, "ah"))
    b.append(box(16, 232, 14, 14, t["ours_fill"], t["ours_edge"], 3))
    b.append(text(38, 244, "AGXForge", 13, t["ink"]))
    b.append(box(16, 258, 14, 14, t["infra_fill"], t["infra_edge"], 3))
    b.append(text(38, 270, "Apple", 13, t["ink"]))
    b.append(text(16, 300, "MiniLM and Qwen ran below Metal", 12, t["ink2"]))
    b.append(text(16, 316, "with no Metal in the process.", 12, t["ink2"]))
    return svg(1000, 346, "\n".join(b),
               "Two execution paths for the same AGXForge-compiled G17 code: through Metal, and below Metal")


def decode(t):
    """Median decode throughput of three implementations on one shared 128-token history, through Metal."""
    d = json.load(open(DECODE))
    rows = []
    for key in ("mlx", "graph_forced", "graph_twin_all_forced"):
        reps = d["tok_s"][key]["per_rep"]
        rows.append((key, statistics.median(reps), min(reps), max(reps)))
    base = rows[0][1]
    labels = {"mlx": ("mlx-lm", "generate_step", t["base"]),
              "graph_forced": ("AGXForge design", "AGXForge compiler", t["ours"]),
              "graph_twin_all_forced": ("AGXForge design", "Apple's compiler", t["apple"])}
    x0, x1, vmax = 210, 820, 250.0
    sx = lambda v: x0 + (x1 - x0) * v / vmax
    b = [text(16, 26, "Decode throughput, tokens per second (median of 4 repetitions)", 15, t["ink"], 600),
         text(16, 46, "InternLM2.5-1.8B-chat, 4-bit weights, 1,792-token context, one shared 128-token history, "
                      "all three through Metal", 12, t["ink2"])]
    top = 70
    for v in range(0, 251, 50):
        b.append('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="%s" stroke-width="1"/>'
                 % (sx(v), top, sx(v), top + 150, t["grid"]))
        b.append(text(sx(v), top + 168, str(v), 12, t["muted"], 400, "middle"))
    for i, (key, med, lo, hi) in enumerate(rows):
        name, sub, color = labels[key]
        cy = top + 26 + i * 48
        b.append(text(x0 - 12, cy - 2, name, 14, t["ink"], 600, "end"))
        b.append(text(x0 - 12, cy + 14, sub, 12, t["ink2"], 400, "end"))
        w = sx(med) - x0
        # 24px bar, square at the baseline, 4px rounded data end
        b.append('<path d="M %g %g h %g a 4 4 0 0 1 4 4 v 16 a 4 4 0 0 1 -4 4 h %g z" fill="%s"/>'
                 % (x0, cy - 12, w - 4, -(w - 4), color))
        ratio = "" if key == "mlx" else "  ·  %.2f× mlx-lm" % (med / base)
        b.append(text(sx(med) + 12, cy + 5, "%.1f%s" % (med, ratio), 14, t["ink"], 600 if key != "mlx" else 400))
    reps = {k: d["tok_s"][k]["per_rep"] for k in ("mlx", "graph_forced", "graph_twin_all_forced")}
    paired = lambda k: [o / m for o, m in zip(reps[k], reps["mlx"])]
    b.append(text(16, top + 196, "Per repetition, against mlx-lm: %.2f-%.2f× (AGXForge compiler) and %.2f-%.2f× (Apple's "
                                 "compiler). The design's lead survives Apple's compiler; it does not need AGXForge's "
                                 "instruction control." % (min(paired("graph_forced")), max(paired("graph_forced")),
                                                           min(paired("graph_twin_all_forced")),
                                                           max(paired("graph_twin_all_forced"))), 12, t["ink2"]))
    return svg(1000, top + 210, "\n".join(b),
               "Decode throughput: mlx-lm %.1f, the AGXForge design with AGXForge's compiler %.1f, with Apple's compiler %.1f tokens per second"
               % tuple(r[1] for r in rows))


def packing(t):
    """Lane 0's eight accumulator slots, read by the next MMA as its B operand."""
    cw, ch, x0, y0 = 70, 38, 250, 76
    rows = [("Canonical D element (hardware)", ["(%d, %d)" % (j >> 2, j & 3) for j in range(8)], None),
            ("Read by the next MMA as B element", ["(%d, %d)" % (8 * (j >> 2), j & 3) for j in range(8)], None),
            ("Canonical packing puts D row", [str(j >> 2) for j in range(8)], "infra"),
            ("AGXForge packing puts D row", [str(8 * (j >> 2)) for j in range(8)], "ours")]
    b = [text(16, 26, "One register, two meanings: lane 0 of a 16 × 16 accumulator, fed to the next MMA as B", 15,
              t["ink"], 600),
         text(16, 46, "Technical reference 10.5. The hardware writes D in one coordinate map and reads B in another.",
              12, t["ink2"])]
    for j in range(8):
        b.append(text(x0 + j * cw + cw / 2, y0 - 10, "slot %d" % j, 12, t["ink"] if j == 4 else t["muted"],
                      600 if j == 4 else 400, "middle"))
    for r, (label, cells, kind) in enumerate(rows):
        y = y0 + r * (ch + 10)
        b.append(text(x0 - 14, y + ch / 2 + 5, label, 13, t["ink"], 500, "end"))
        fill = t["ours_fill"] if kind == "ours" else t["infra_fill"] if kind == "infra" else "none"
        edge = t["ours_edge"] if kind == "ours" else t["infra_edge"] if kind == "infra" else t["grid"]
        for j, c in enumerate(cells):
            b.append(box(x0 + j * cw + 3, y, cw - 6, ch, fill, edge, 6, 1))
            b.append(text(x0 + j * cw + cw / 2, y + ch / 2 + 5, c, 13, t["ink"], 700 if j == 4 else 400, "middle"))
    b.append('<rect x="%g" y="%g" width="%g" height="%g" rx="8" fill="none" stroke="%s" stroke-width="2"/>'
             % (x0 + 4 * cw, y0 - 4, cw, 4 * (ch + 10) - 2, t["focus"]))
    xr = x0 + 8 * cw + 18
    b.append(text(xr, y0 + 2 * (ch + 10) + 14, "B row 8 receives D row 1:", 12, t["ink"], 600))
    b.append(text(xr, y0 + 2 * (ch + 10) + 30, "rotated; must be undone", 12, t["ink2"]))
    b.append(text(xr, y0 + 3 * (ch + 10) + 14, "B row 8 receives D row 8:", 12, t["ink"], 600))
    b.append(text(xr, y0 + 3 * (ch + 10) + 30, "used directly, no shuffle", 12, t["ink2"]))
    return svg(1000, y0 + 4 * (ch + 10) + 12, "\n".join(b),
               "Lane 0 slot 4 holds D element (1, 0) and is read as B element (8, 0); AGXForge's packing puts D row 8 "
               "there")


FIGURES = {"execution-paths": paths, "decode-throughput": decode, "feed-packing": packing}


def render():
    out = {}
    for name, make in FIGURES.items():
        out[name + ".svg"] = make(THEMES["light"])
        out[name + "-dark.svg"] = make(THEMES["dark"])
    return out


def main(argv):
    check = "--check" in argv
    stale = []
    os.makedirs(OUT, exist_ok=True)
    for name, content in render().items():
        path = os.path.join(OUT, name)
        if check:
            if not os.path.exists(path) or open(path).read() != content:
                stale.append(name)
        else:
            with open(path, "w") as fh:
                fh.write(content)
            print("wrote", os.path.relpath(path, ROOT))
    if stale:
        print("stale figures (run tools/g17figures.py):", ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
