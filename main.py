import html
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

INPUT_PHOTO = sys.argv[1] if len(sys.argv) > 1 else "photo.png"
INNER_RADIUS_FRAC = 0.87
BG_COLOR = (3, 25, 47)
BG_TOLERANCE = 75
ROWS = 100
CELL_RATIO = 2.0
TONE_LO, TONE_HI, TONE_GAMMA = 0.08, 0.85, 1.05
UNSHARP_RADIUS, UNSHARP_PERCENT = 3, 60

CHARS = (
    " `.'-,_:;\"!~^|\\/()[]{}?*+=<>"
    "iltfjrcvxzsunyeoahkbdpqwm"
    "ZXCVJLTFYUONKAHEDRPQWMB%&#$@80"
)

CALIB_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
CALIB_SIZE = 20

SVG_FONT_SIZE = 10
SVG_CHAR_W = 5.5
SVG_LINE_H = 11.0
SVG_PAD = 12
DARK_FG, DARK_BG = "#c9d1d9", "#161b22"
LIGHT_FG, LIGHT_BG = "#24292f", "#f6f8fa"


def isolate_subject(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    cx, cy = w / 2, h / 2
    r = INNER_RADIUS_FRAC * (w / 2)

    circle = Image.new("L", (w, h), 0)
    ImageDraw.Draw(circle).ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

    px, mk = im.load(), circle.load()
    subject = Image.new("L", (w, h), 0)
    sp = subject.load()
    for y in range(h):
        for x in range(w):
            if mk[x, y] == 0:
                continue
            rr, gg, bb = px[x, y]
            dist = (abs(rr - BG_COLOR[0]) + abs(gg - BG_COLOR[1])
                    + abs(bb - BG_COLOR[2]))
            if dist > BG_TOLERANCE:
                sp[x, y] = 255

    subject = (subject.filter(ImageFilter.MinFilter(3))
                      .filter(ImageFilter.MaxFilter(5))
                      .filter(ImageFilter.MinFilter(3)))

    gray = ImageOps.grayscale(im)
    bbox = subject.getbbox()
    return gray.crop(bbox), subject.crop(bbox)


def calibrate_glyphs():
    font = ImageFont.truetype(CALIB_FONT, CALIB_SIZE)
    adv = font.getlength("M")
    asc, desc = font.getmetrics()
    cw, chh = int(np.ceil(adv)), asc + desc

    coverage = {}
    for ch in CHARS:
        tile = Image.new("L", (cw, chh), 0)
        ImageDraw.Draw(tile).text((0, 0), ch, font=font, fill=255)
        coverage[ch] = np.asarray(tile, dtype=float).mean() / 255.0

    peak = max(coverage.values())
    density = {ch: v / peak for ch, v in coverage.items()}
    ordered = sorted(density.items(), key=lambda kv: kv[1])
    dvals = np.array([v for _, v in ordered])
    dchars = [c for c, _ in ordered]
    return density, dvals, dchars


def build_ascii(gray, mask, density, dvals, dchars):
    W, H = gray.size
    cols = round(ROWS * (W / H) * CELL_RATIO)

    src = gray.filter(ImageFilter.UnsharpMask(
        radius=UNSHARP_RADIUS, percent=UNSHARP_PERCENT, threshold=2))

    arr = np.asarray(src.resize((cols, ROWS), Image.LANCZOS),
                     dtype=float) / 255.0
    marr = np.asarray(mask.resize((cols, ROWS), Image.LANCZOS),
                      dtype=float) / 255.0

    ink = np.clip(((1.0 - arr) - TONE_LO) / (TONE_HI - TONE_LO), 0, 1)
    ink = np.power(ink, TONE_GAMMA)
    ink[marr < 0.45] = 0.0

    lines, work = [], ink.copy()
    for r in range(ROWS):
        line = ""
        for c in range(cols):
            v = min(1.0, max(0.0, work[r, c]))
            ch = dchars[int(np.abs(dvals - v).argmin())]
            line += ch
            if marr[r, c] >= 0.45:
                err = v - density[ch]
                if c + 1 < cols:
                    work[r, c + 1] += err * 0.4
                if r + 1 < ROWS:
                    work[r + 1, c] += err * 0.3
                    if c + 1 < cols:
                        work[r + 1, c + 1] += err * 0.2
        lines.append(line.rstrip())
    return lines


def write_svg(lines, fg, bg, fname):
    maxlen = max(len(l) for l in lines)
    wpx = int(SVG_PAD * 2 + maxlen * SVG_CHAR_W)
    hpx = int(SVG_PAD * 2 + SVG_LINE_H * len(lines))

    tspans, y = [], SVG_PAD + SVG_LINE_H - 2
    for line in lines:
        text = html.escape(line) if line else " "
        tspans.append(f'<tspan x="{SVG_PAD}" y="{y:.1f}">{text}</tspan>')
        y += SVG_LINE_H

    svg = f"""<?xml version='1.0' encoding='UTF-8'?>
        <svg xmlns="http://www.w3.org/2000/svg" font-family="ConsolasFallback,Consolas,monospace" width="{wpx}px" height="{hpx}px" font-size="{SVG_FONT_SIZE}px">
        <style>
        @font-face {{
        src: local('Consolas'), local('Consolas Bold');
        font-family: 'ConsolasFallback';
        font-display: swap;
        -webkit-size-adjust: 109%;
        size-adjust: 109%;
        }}
        text, tspan {{white-space: pre;}}
        </style>
        <rect width="{wpx}px" height="{hpx}px" fill="{bg}" rx="15"/>
        <text x="{SVG_PAD}" y="{SVG_PAD + SVG_LINE_H - 2:.1f}" fill="{fg}">
        {chr(10).join(tspans)}
        </text>
        </svg>"""
    with open(fname, "w") as f:
        f.write(svg)
    print(f"wrote {fname}  ({wpx}x{hpx}px)")


def main():
    gray, mask = isolate_subject(INPUT_PHOTO)
    density, dvals, dchars = calibrate_glyphs()
    lines = build_ascii(gray, mask, density, dvals, dchars)

    with open("ascii_art.txt", "w") as f:
        f.write("\n".join(lines))
    print(f"grid: {max(len(l) for l in lines)} cols x {len(lines)} rows")

    write_svg(lines, DARK_FG, DARK_BG, "dark_mode.svg")
    write_svg(lines, LIGHT_FG, LIGHT_BG, "light_mode.svg")


if __name__ == "__main__":
    main()
