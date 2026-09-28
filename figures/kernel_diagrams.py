"""Generate the kernel walkthrough's SVG diagrams; Python standard library only."""

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent
INK = "#273342"
MUTED = "#5d6977"
PURPLE = "#e8dff2"
BLUE = "#dce9f5"
GREEN = "#dceee8"
ORANGE = "#fae9d5"
GRAY = "#edf0f3"


class Diagram:
    def __init__(self, title, height):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" '
            f'viewBox="0 0 960 {height}" role="img" aria-labelledby="title">',
            f'<title id="title">{escape(title)}</title>',
            '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" '
            'refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#607184"/></marker></defs>',
            '<rect width="100%" height="100%" fill="white"/>',
            '<g font-family="DejaVu Sans, sans-serif" fill="#273342">',
        ]
        self.text(28, 34, title, 22, bold=True)

    def text(self, x, y, text, size=16, color=INK, bold=False, anchor="start"):
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{"600" if bold else "400"}" text-anchor="{anchor}">{escape(text)}</text>'
        )

    def box(self, x, y, w, h, label="", fill=GRAY, size=16):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" '
            f'fill="{fill}" stroke="#c3cbd4" stroke-width="1"/>'
        )
        for i, line in enumerate(label.split("\n")):
            self.text(
                x + w / 2,
                y + h / 2 + (i - (len(label.split("\n")) - 1) / 2) * 22 + 5,
                line,
                size,
                anchor="middle",
            )

    def arrow(self, x, y, xx, yy):
        self.parts.append(
            f'<path d="M{x},{y} L{xx},{yy}" fill="none" stroke="#607184" '
            'stroke-width="1.7" marker-end="url(#arrow)"/>'
        )

    def save(self, name):
        (OUT / f"{name}.svg").write_text("\n".join(self.parts + ["</g>", "</svg>"]) + "\n")


def clusters():
    d = Diagram("Change the reuse footprint, keep the 256 × 256 MMA tile", 400)
    for x, cm, cn, title in [
        (35, 4, 1, "General: 4 × 1"),
        (310, 4, 2, "4K: 4 × 2"),
        (635, 8, 2, "8K / 16K: 8 × 2"),
    ]:
        d.text(x, 78, title, 18, bold=True)
        for m in range(cm // 2):
            for n in range(cn):
                d.box(x + n * 115, 100 + m * 48, 105, 40, "256 × 256", PURPLE, 14)
        d.text(x, 320, f"{cm * 128} rows × {cn * 256} columns", 16)
        d.text(x, 345, f"{cm * cn} CTAs; each box = 2 CTAs", 14, MUTED)
    d.text(
        28, 383, "A is shared across the N direction →     B is shared across the M direction ↓", 16
    )
    d.save("kernel_clusters")


def startup():
    d = Diagram("4K: move the first memory request into cluster setup", 320)
    d.text(28, 78, "Before", 17, bold=True)
    d.box(155, 55, 195, 52, "Cluster setup", GRAY)
    d.arrow(355, 81, 385, 81)
    d.box(392, 55, 220, 52, "TMA loads → wait", BLUE)
    d.arrow(617, 81, 647, 81)
    d.box(654, 55, 245, 52, "First K iteration", PURPLE)
    d.text(28, 156, "4K preset", 17, bold=True)
    d.box(155, 130, 330, 52, "Cluster setup", GRAY)
    d.box(155, 196, 330, 48, "Prefetch A, B, SFA, SFB into L2", GREEN, 15)
    d.arrow(492, 156, 522, 156)
    d.box(530, 130, 182, 52, "TMA loads → wait", BLUE, 15)
    d.arrow(718, 156, 748, 156)
    d.box(756, 130, 143, 52, "First K iteration", PURPLE, 14)
    d.arrow(485, 220, 530, 183)
    d.text(155, 275, "Prefetch covers the first 768 K values of each CTA’s first output tile.", 16)
    d.text(
        155, 301, "The normal TMA transfer and its readiness barrier are still required.", 15, MUTED
    )
    d.save("kernel_4k_startup")


def ownership():
    d = Diagram("8K: assign A-row stripes to two persistent-slot groups", 420)
    d.text(28, 76, "Preferred-cluster coordinates: 8 stripes × 16 cluster columns", 17)
    x0, y0, cw, ch = 86, 125, 37, 27
    d.text(235, 108, "Cluster columns 0–7", 15, anchor="middle")
    d.text(533, 108, "Cluster columns 8–15", 15, anchor="middle")
    for m in range(8):
        d.text(67, y0 + m * ch + 18, str(m), 14, anchor="end")
        for n in range(16):
            a = m < (5 if n < 8 else 4)
            d.box(
                x0 + n * cw,
                y0 + m * ch,
                cw - 3,
                ch - 3,
                "X" if a else "Y",
                PURPLE if a else BLUE,
                12,
            )
    d.text(28, 113, "M", 15, bold=True)
    d.box(707, 126, 220, 76, "Group X\nslots 0, 1, 4, 5, 8", PURPLE, 16)
    d.box(707, 224, 220, 76, "Group Y\nslots 2, 3, 6, 7", BLUE, 16)
    d.text(86, 368, "X: 5 × 8 + 4 × 8 = 72 tiles  ·  Y: 3 × 8 + 4 × 8 = 56 tiles", 17)
    d.text(
        86,
        397,
        "Colors denote slot groups. Each cell covers 1024 × 512 output elements.",
        15,
        MUTED,
    )
    d.save("kernel_8k_ownership")


def tail():
    d = Diagram("16K: keep the crossing load; remove unused handshakes", 455)
    d.text(28, 72, "Final outer K iteration: 256 live values out of 768", 17)
    x0, width = 168, 744

    def band(y, label, segments):
        d.text(28, y + 28, label, 16, bold=True)
        for start, end, text, color in segments:
            d.box(x0 + start / 768 * width, y, (end - start) / 768 * width - 3, 46, text, color, 14)

    band(
        98,
        "A/B loads",
        [
            (0, 256, "Slice 0: live", BLUE),
            (256, 512, "Slice 1: keep zero fill", ORANGE),
            (512, 768, "Slice 2: remove", GRAY),
        ],
    )
    band(
        161,
        "Scale loads",
        [
            (0, 192, "SF 0: keep", GREEN),
            (192, 384, "SF 1: keep", GREEN),
            (384, 576, "SF 2: remove", GRAY),
            (576, 768, "SF 3: remove", GRAY),
        ],
    )
    for i in range(8):
        d.box(x0 + i * width / 8, 224, width / 8 - 3, 44, f"MMA {i}", PURPLE if i < 3 else GRAY, 14)
    d.text(28, 251, "MMA issue", 16, bold=True)
    for k in [0, 192, 256, 384, 512, 768]:
        d.text(x0 + k / 768 * width, 299, str(k), 13, MUTED, anchor="middle")
    d.text(28, 299, "K offset", 15, color=MUTED)
    d.text(28, 340, "MMA 2 reads [192, 288): 64 live values + 32 zeros from A/B slice 1.", 17)
    d.text(28, 372, "Before: 3 A/B + 4 SF handshakes.  After: 2 A/B + 2 SF handshakes.", 17)
    d.text(
        28,
        404,
        "Both versions issue only MMAs 0–2. The savings are in loads, waits and releases.",
        16,
    )
    d.text(
        28,
        435,
        "Slice ranges are logical K values; packed byte offsets are half these values.",
        15,
        MUTED,
    )
    d.save("kernel_16k_tail")


def descriptors():
    d = Diagram("16K: prepare shared-memory MMA descriptors before the wait", 330)
    d.text(28, 82, "Before", 17, bold=True)
    d.box(170, 58, 150, 50, "Wait for A/B", GRAY)
    d.arrow(327, 83, 357, 83)
    d.box(365, 58, 280, 50, "Construct A + B descriptors", BLUE, 15)
    d.arrow(652, 83, 682, 83)
    d.box(690, 58, 220, 50, "Issue MMA", PURPLE)
    d.text(28, 159, "16K preset", 17, bold=True)
    d.box(170, 135, 280, 50, "Pack stage-address templates", BLUE, 15)
    d.arrow(457, 160, 487, 160)
    d.box(495, 135, 150, 50, "Wait for A/B", GRAY)
    d.arrow(652, 160, 682, 160)
    d.box(690, 135, 220, 50, "Add offset → issue MMA", PURPLE, 14)
    d.box(170, 217, 195, 54, "Within slice\n(current, current)", GREEN, 15)
    d.box(385, 217, 195, 54, "Cross slice\n(current, next)", ORANGE, 15)
    d.text(603, 236, "Layout bits stay fixed.", 16)
    d.text(603, 262, "Only addresses and offsets vary.", 15, MUTED)
    d.text(170, 307, "Address arithmetic moves; the data-readiness barrier stays in place.", 16)
    d.save("kernel_16k_descriptors")


def stores():
    d = Diagram("32K store pairing: emit adjacent row segments together", 382)
    d.text(28, 77, "Unpaired", 17, bold=True)
    d.box(170, 54, 155, 56, "Convert half 0", BLUE)
    d.arrow(332, 82, 358, 82)
    d.box(365, 54, 165, 56, "2 × 32 B stores", PURPLE)
    d.arrow(537, 82, 563, 82)
    d.box(570, 54, 160, 56, "Convert half 1", GRAY)
    d.arrow(737, 82, 763, 82)
    d.box(770, 54, 165, 56, "2 × 32 B stores", PURPLE)
    d.text(28, 170, "Paired", 17, bold=True)
    d.box(170, 142, 210, 64, "Convert half 0\nHold 64 B in registers", ORANGE, 15)
    d.arrow(387, 174, 415, 174)
    d.box(423, 142, 192, 64, "Convert half 1", GRAY, 14)
    d.arrow(622, 174, 650, 174)
    d.box(658, 142, 277, 64, "Half 0 stores → half 1 stores\n4 × 32 B, back to back", PURPLE, 15)
    d.box(170, 246, 310, 44, "64 B half 0", BLUE)
    d.box(483, 246, 310, 44, "64 B half 1", GREEN)
    d.text(807, 274, "128 B line", 15)
    d.text(170, 324, "Same bytes and store count. Shorter gap between writes to the same line.", 16)
    d.text(
        170,
        356,
        "Traversal may reverse: half 1 then half 0 still forms the same adjacent pair.",
        15,
        MUTED,
    )
    d.save("kernel_32k_stores")


def reuse_32k():
    d = Diagram("32K: exchange M sharing for a narrower retained A band", 430)
    for x, cm, cn, title in [(95, 8, 2, "Previous: 8 × 2"), (580, 4, 4, "Final: 4 × 4")]:
        d.text(x, 78, title, 18, bold=True)
        for m in range(cm // 2):
            for n in range(cn):
                d.box(x + n * 66, 104 + m * 46, 62, 42, "2 CTAs", PURPLE, 12)
        d.text(x, 322, f"{cm * 128} rows × {cn * 256} columns", 17)
        d.text(x, 353, f"Four M stripes: {cm * 128 * 4} A rows", 16)
        d.text(x, 382, f"A band: {cm * 8} MiB + {cm} MiB scales", 16)
    d.arrow(315, 191, 525, 191)
    d.text(420, 157, "A multicast: 2 → 4", 16, anchor="middle")
    d.text(420, 231, "B multicast: 4 → 2", 16, anchor="middle")
    d.text(
        28, 418, "Both use 16 CTAs and the same eight 256 × 256 MMA tiles per cluster.", 15, MUTED
    )
    d.save("kernel_32k_reuse")


def shared_release():
    d = Diagram("Shared epilogue: release before the first global store", 350)
    d.text(28, 77, "Baseline", 17, bold=True)
    steps = [
        (170, 130, "Load subtile 0", BLUE),
        (330, 145, "Convert + store 0", ORANGE),
        (505, 130, "Load subtile 1", BLUE),
        (665, 105, "Fence +\nrelease", GREEN),
        (800, 130, "Convert +\nstore 1, …", PURPLE),
    ]
    for x, w, label, color in steps:
        d.box(x, 53, w, 50, label, color, 14)
    for x, xx in [(305, 323), (480, 498), (640, 658), (775, 793)]:
        d.arrow(x, 78, xx, 78)
    d.text(28, 167, "GEMM³", 17, bold=True)
    steps = [
        (170, 130, "Load subtile 0", BLUE),
        (330, 145, "Load subtile 1", BLUE),
        (505, 130, "Fence + release", GREEN),
        (665, 265, "Convert + store 0, then 1, …", PURPLE),
    ]
    for x, w, label, color in steps:
        d.box(x, 143, w, 50, label, color, 14)
    for x, xx in [(305, 323), (480, 498), (640, 658)]:
        d.arrow(x, 168, xx, 168)
    d.text(170, 242, "Two 32-column subtiles cover the 36-column overlap.", 17)
    d.text(170, 274, "The next tile can acquire the accumulator before output traffic starts.", 16)
    d.text(170, 313, "The release still follows a TMEM-read fence in both paths.", 15, MUTED)
    d.save("kernel_shared_release")


def shared_dataflow():
    d = Diagram("The scale path used by GEMM³ presets", 370)
    d.text(28, 80, "A / B", 17, bold=True)
    d.box(180, 56, 170, 54, "Global NVFP4", BLUE)
    d.arrow(360, 83, 470, 83)
    d.text(415, 68, "TMA", 14, MUTED, anchor="middle")
    d.box(480, 56, 210, 54, "Shared-memory ring", BLUE, 15)
    d.arrow(700, 83, 840, 83)
    d.box(850, 56, 82, 164, "MMA", PURPLE)
    d.text(28, 191, "SFA / SFB", 17, bold=True)
    d.box(180, 167, 170, 54, "Global E4M3", GREEN)
    d.arrow(360, 194, 400, 194)
    d.text(380, 160, "TMA", 13, MUTED, anchor="middle")
    d.box(410, 167, 155, 54, "Shared-memory ring", GREEN, 13)
    d.arrow(575, 194, 685, 194)
    d.text(630, 177, "tcgen05.cp", 14, MUTED, anchor="middle")
    d.box(695, 167, 120, 54, "TMEM scales", GREEN, 14)
    d.arrow(823, 194, 842, 194)
    d.text(180, 276, "Operand and scale producers have separate rings and readiness barriers.", 16)
    d.text(180, 308, "The MMA warp copies live scales into TMEM before using them.", 16)
    d.text(
        180, 342, "Direct register-to-TMEM scale stores were an unselected experiment.", 15, MUTED
    )
    d.save("kernel_shared_dataflow")


def pr_overlap():
    d = Diagram("Baseline: two accumulator views in one 512-column allocation", 410)
    x, scale = 115, 1.52
    d.text(28, 78, "TMEM column", 14, MUTED)
    for col in (0, 220, 256, 476, 512):
        d.text(x + col * scale, 104, str(col), 14, anchor="middle")
    d.box(x, 119, 256 * scale, 52, "Accumulator view 0 · 256 columns", BLUE, 15)
    d.box(x + 220 * scale, 185, 256 * scale, 52, "Accumulator view 1 · 256 columns", PURPLE, 15)
    d.box(x + 220 * scale, 251, 36 * scale, 37, "36", ORANGE, 14)
    d.box(x + 476 * scale, 119, 36 * scale, 118, "SF", GREEN, 14)
    d.text(x + 238 * scale, 311, "Shared intersection", 15, anchor="middle")
    d.arrow(x + 238 * scale, 295, x + 238 * scale, 289)
    d.text(
        28, 350, "View 0 drains its high columns first; view 1 drains its low columns first.", 17
    )
    d.text(
        28,
        384,
        "Read the 36-column intersection → fence TMEM reads → release for the next tile.",
        16,
    )
    d.save("pr4866_overlap")


def pr_stores():
    d = Diagram("Baseline: layout and alignment for vectorized output stores", 335)
    for x, w, label, color in (
        (28, 275, "Compact N-contiguous layout\nN is divisible by 64", BLUE),
        (342, 275, "32-byte pointer alignment\nchecked at the FFI boundary", GREEN),
        (656, 275, "Register/output common layout\n256-bit copy-width cap", PURPLE),
    ):
        d.box(x, 72, w, 76, label, color, 15)
        d.arrow(x + w / 2, 156, 480, 203)
    d.box(246, 211, 468, 56, "CopyR2GOp: up to 16 × FP16/BF16 per store", ORANGE, 16)
    d.text(
        480,
        310,
        "Layout, stride and alignment must all support the wider copy.",
        16,
        anchor="middle",
    )
    d.save("pr4866_stores")


if __name__ == "__main__":
    for draw in (
        clusters,
        startup,
        ownership,
        tail,
        descriptors,
        stores,
        reuse_32k,
        shared_release,
        shared_dataflow,
        pr_overlap,
        pr_stores,
    ):
        draw()
    print("Wrote kernel walkthrough SVG diagrams.")
