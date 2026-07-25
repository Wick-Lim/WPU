#!/usr/bin/env python3
"""
gen_gerbers.py -- emit RS-274X gerbers + a courtyard DRC for the v3-proto study board.

Produces, to TRUE 1:1 scale, the placement the routability study (escape_analysis.py)
assumes: a 130 x 110 mm board with the SoC (~50x50 mm) centered and 20 LPDDR5X
24 GB 496-FBGA packages (~12.4 x 15 mm) in a ring (6 top / 6 bottom / 4 left /
4 right), plus mounting holes and a drill file.  Outputs open in any gerber viewer
(gerbv, or online at e.g. tracespace.io) and are committable/reviewable.

Scope: this is the PLACEMENT + outline + courtyard layer (fab/assembly-footprint
fidelity), not routed copper -- routability is argued analytically in
escape_analysis.py. The DRC here is the placement-level one that matters for the
"does it physically fit in 130x110 with clearance" question: courtyard overlaps
and board-edge clearance. Package/board geometry mirrors R3_APPLIANCE_SPEC 5c and
the board.html concept diagram.
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gerbers")

# ---- board + package geometry (mm), mirrors R3_APPLIANCE_SPEC 5c ------------------
BOARD_W, BOARD_H = 130.0, 110.0
SOC_W, SOC_H     = 50.0, 50.0
DRAM_W, DRAM_H   = 12.4, 15.0        # 496-FBGA body (short x long)
EDGE_CLEAR       = 1.0               # min copper/courtyard to board edge
COURTYARD_EXP    = 0.25             # courtyard expansion beyond body each side
MOUNT_D          = 3.2               # M3 mounting hole
cx, cy = BOARD_W / 2, BOARD_H / 2

def dram_ring():
    """20 package centers: 6 top + 6 bottom rows (H) + 4 left + 4 right columns (V).

    Geometry constraint the naive ring violates: four 15 mm packages cannot sit
    along the SoC's 50 mm edge (they span ~60 mm). So the L/R columns live in the
    OUTBOARD corner lanes (x beyond the row span) and use nearly the full board
    height; the T/B rows are pulled inboard (tighter pitch) so their end packages
    clear those column lanes. The placement DRC below enforces both.
    """
    places = []
    # top / bottom rows: 6 H-oriented packages, pitch tight enough to clear the columns
    row_pitch = 16.5
    xs = [cx + (i - 2.5) * row_pitch for i in range(6)]
    top_y = cy + SOC_H / 2 + COURTYARD_EXP + DRAM_H / 2 + 0.6
    bot_y = cy - SOC_H / 2 - COURTYARD_EXP - DRAM_H / 2 - 0.6
    for x in xs:
        places.append((x, top_y, "H"))   # H = long axis horizontal
        places.append((x, bot_y, "H"))
    # left / right columns: 4 V-oriented packages in the outboard corner lanes,
    # spanning most of the board height (clears the rows on x)
    left_x  = EDGE_CLEAR + DRAM_W / 2 + 0.3
    right_x = BOARD_W - EDGE_CLEAR - DRAM_W / 2 - 0.3
    ys = [cy + d for d in (-33, -11, 11, 33)]
    for y in ys:
        places.append((left_x, y, "V"))   # V = long axis vertical
        places.append((right_x, y, "V"))
    return places

# --------------------------------------------------------------- RS-274X emitter --
def g_header():
    return ["%FSLAX46Y46*%", "%MOMM*%", "%LPD*%"]

def mm(v):
    return f"{int(round(v * 1e6))}"

def region_rect(cx_, cy_, w, h):
    """A filled rectangle as a gerber region (G36/G37)."""
    x0, y0, x1, y1 = cx_ - w/2, cy_ - h/2, cx_ + w/2, cy_ + h/2
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    out = ["G36*"]
    out.append(f"X{mm(pts[0][0])}Y{mm(pts[0][1])}D02*")
    for (x, y) in pts[1:]:
        out.append(f"X{mm(x)}Y{mm(y)}D01*")
    out.append("G37*")
    return out

def outline_rect(cx_, cy_, w, h, ap):
    """A rectangle drawn as a closed polyline with aperture `ap` (Dnn)."""
    x0, y0, x1, y1 = cx_ - w/2, cy_ - h/2, cx_ + w/2, cy_ + h/2
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    out = [f"{ap}*", f"X{mm(pts[0][0])}Y{mm(pts[0][1])}D02*"]
    for (x, y) in pts[1:]:
        out.append(f"X{mm(x)}Y{mm(y)}D01*")
    return out

def write_gerber(path, body_lines, apertures):
    lines = g_header() + apertures + body_lines + ["M02*"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

def gen():
    os.makedirs(OUT, exist_ok=True)
    drams = dram_ring()

    # ---- Edge.Cuts: board outline (0.1 mm line) ----
    ap = ["%ADD10C,0.100*%"]
    body = outline_rect(cx, cy, BOARD_W, BOARD_H, "D10")
    write_gerber(os.path.join(OUT, "board-Edge_Cuts.gbr"), body, ap)

    # ---- F.Courtyard: SoC + DRAM courtyards (0.05 mm line) ----
    ap = ["%ADD11C,0.050*%"]
    body = []
    body += outline_rect(cx, cy, SOC_W + 2*COURTYARD_EXP, SOC_H + 2*COURTYARD_EXP, "D11")
    for (x, y, o) in drams:
        w, h = (DRAM_W, DRAM_H) if o == "V" else (DRAM_H, DRAM_W)
        body += outline_rect(x, y, w + 2*COURTYARD_EXP, h + 2*COURTYARD_EXP, "D11")
    write_gerber(os.path.join(OUT, "board-F_Courtyard.gbr"), body, ap)

    # ---- F.Fab / top copper: filled package bodies (regions) ----
    ap = ["%ADD12C,0.010*%"]
    body = region_rect(cx, cy, SOC_W, SOC_H)
    for (x, y, o) in drams:
        w, h = (DRAM_W, DRAM_H) if o == "V" else (DRAM_H, DRAM_W)
        body += region_rect(x, y, w, h)
    write_gerber(os.path.join(OUT, "board-F_Fab.gbr"), body, ap)

    # ---- Excellon drill: 4 mounting holes ----
    holes = [(EDGE_CLEAR + 3, EDGE_CLEAR + 3), (BOARD_W - EDGE_CLEAR - 3, EDGE_CLEAR + 3),
             (EDGE_CLEAR + 3, BOARD_H - EDGE_CLEAR - 3),
             (BOARD_W - EDGE_CLEAR - 3, BOARD_H - EDGE_CLEAR - 3)]
    drl = ["M48", "METRIC,TZ", f"T1C{MOUNT_D:.3f}", "%", "T1"]
    for (x, y) in holes:
        drl.append(f"X{x:.3f}Y{y:.3f}")
    drl += ["T0", "M30"]
    with open(os.path.join(OUT, "board.drl"), "w") as f:
        f.write("\n".join(drl) + "\n")

    return drams

# ------------------------------------------------------------------ placement DRC --
def drc(drams):
    problems = []
    def cy_rect(x, y, o, exp=COURTYARD_EXP):
        w, h = (DRAM_W, DRAM_H) if o == "V" else (DRAM_H, DRAM_W)
        return (x - w/2 - exp, y - h/2 - exp, x + w/2 + exp, y + h/2 + exp)
    soc = (cx - SOC_W/2 - COURTYARD_EXP, cy - SOC_H/2 - COURTYARD_EXP,
           cx + SOC_W/2 + COURTYARD_EXP, cy + SOC_H/2 + COURTYARD_EXP)
    boxes = [("SoC", soc)] + [(f"DRAM{i}", cy_rect(x, y, o)) for i, (x, y, o) in enumerate(drams)]

    def overlap(a, b):
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])
    # courtyard overlaps
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if overlap(boxes[i][1], boxes[j][1]):
                problems.append(f"COURTYARD OVERLAP: {boxes[i][0]} <-> {boxes[j][0]}")
    # board-edge clearance
    for name, (x0, y0, x1, y1) in boxes:
        if x0 < EDGE_CLEAR or y0 < EDGE_CLEAR or x1 > BOARD_W - EDGE_CLEAR or y1 > BOARD_H - EDGE_CLEAR:
            problems.append(f"EDGE CLEARANCE: {name} within {EDGE_CLEAR} mm of board edge "
                            f"(bbox {x0:.1f},{y0:.1f}..{x1:.1f},{y1:.1f})")
    return problems

def svg_preview(drams, problems):
    """A to-scale SVG of the placement (1 mm = 4 px), for human review."""
    S = 4.0
    W, H = BOARD_W * S, BOARD_H * S
    def Y(y):  # SVG y is top-down; board y is bottom-up
        return H - y * S
    el = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-40 -20 {W+80} {H+70}" '
          f'font-family="monospace">']
    el.append(f'<rect x="-40" y="-20" width="{W+80}" height="{H+70}" fill="#0e1114"/>')
    # board
    el.append(f'<rect x="0" y="0" width="{W}" height="{H}" rx="{3*S}" '
              f'fill="#0d2818" stroke="#1e4a30" stroke-width="2"/>')
    # SoC
    el.append(f'<rect x="{(cx-SOC_W/2)*S}" y="{Y(cy+SOC_H/2)}" width="{SOC_W*S}" '
              f'height="{SOC_H*S}" rx="4" fill="#1a2129" stroke="#9aa8b5" stroke-width="2"/>')
    el.append(f'<rect x="{(cx-11)*S}" y="{Y(cy+11)}" width="{22*S}" height="{22*S}" '
              f'rx="3" fill="#131a21" stroke="#5fd38a" stroke-width="1.5"/>')
    el.append(f'<text x="{cx*S}" y="{Y(cy)+4}" fill="#5fd38a" font-size="11" '
              f'text-anchor="middle" font-weight="bold">WPU die</text>')
    el.append(f'<text x="{cx*S}" y="{Y(cy)-8}" fill="#9aa8b5" font-size="8" '
              f'text-anchor="middle">SoC ~50x50mm</text>')
    # DRAMs
    for (x, y, o) in drams:
        w, h = (DRAM_W, DRAM_H) if o == "V" else (DRAM_H, DRAM_W)
        el.append(f'<rect x="{(x-w/2)*S}" y="{Y(y+h/2)}" width="{w*S}" height="{h*S}" '
                  f'rx="2" fill="#101418" stroke="#d4b96a" stroke-width="1.2"/>')
        el.append(f'<text x="{x*S}" y="{Y(y)+3}" fill="#8f8360" font-size="7" '
                  f'text-anchor="middle">24G</text>')
    # dims
    el.append(f'<text x="{cx*S}" y="{H+22}" fill="#8fa0ab" font-size="11" '
              f'text-anchor="middle">{BOARD_W:.0f} mm</text>')
    el.append(f'<text x="-16" y="{H/2}" fill="#8fa0ab" font-size="11" '
              f'text-anchor="middle" transform="rotate(-90 -16 {H/2})">{BOARD_H:.0f} mm</text>')
    verdict = "DRC PASS -- 0 overlaps" if not problems else f"DRC FAIL ({len(problems)})"
    el.append(f'<text x="0" y="{H+46}" fill="{"#5fd38a" if not problems else "#e0574a"}" '
              f'font-size="10">{verdict} - 20x LPDDR5X 24GB + SoC, 12-layer HDI [study, to scale]</text>')
    el.append('</svg>')
    p = os.path.join(OUT, "..", "board_preview.svg")
    with open(p, "w") as f:
        f.write("\n".join(el))
    return os.path.normpath(p)

def main():
    drams = gen()
    print(f"emitted gerbers + drill to {OUT}/ :")
    for fn in sorted(os.listdir(OUT)):
        print(f"  {fn}")
    print(f"\nplacement: 1 SoC ({SOC_W}x{SOC_H} mm) + {len(drams)} DRAM "
          f"({DRAM_W}x{DRAM_H} mm) on {BOARD_W}x{BOARD_H} mm board")
    print("\nplacement DRC (courtyard overlaps + board-edge clearance):")
    problems = drc(drams)
    svg = svg_preview(drams, problems)
    print(f"  preview: {svg}")
    if not problems:
        util = (SOC_W*SOC_H + len(drams)*DRAM_W*DRAM_H) / (BOARD_W*BOARD_H) * 100
        print(f"  PASS: 0 overlaps, all packages >= {EDGE_CLEAR} mm from edge.")
        print(f"  component-body area utilization: {util:.0f}% of the board.")
        return 0
    for p in problems:
        print(f"  {p}")
    print(f"  FAIL: {len(problems)} placement problem(s).")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
