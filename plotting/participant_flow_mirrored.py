"""
Participant flow diagram, mirrored layout.

Both groups meet on a shared centre line; the drop-out ribbons peel off
outwards (upwards for the top group, downwards for the bottom one).
One person = the same number of pixels everywhere, so all bar heights and
ribbon thicknesses are directly comparable.

Edit the CONFIG block, run, done.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

# ----------------------------------------------------------------- CONFIG ---
STAGES = ["registered\nn=87", "started\nn=74", "included\nn=53 (71.6%)", "finished\nn=51"]

GROUPS = [
    dict(label="female", side=-1, n=[23, 19, 14, 13], color="#7f7f7f",
         reasons=["DNS", "not included", "DNF"]),
    dict(label="male",   side=+1, n=[64, 55, 39, 38], color="#8ecadd",
         reasons=["DNS", "not included", "DNF"]),
]

LOST_COLOR = "#e2e2e2"
TEXT_DARK = "#3d3d3d"
FLOW_ALPHA = 0.45     # transparency of the "continuing" ribbons
LOST_ALPHA = 0.25     # transparency of the "lost" ribbons

NODE_W = 0.30         # node width (x units; stage spacing = 1.0)
STAGE_DX = 1.00
EXIT_DX = 0.55        # horizontal travel of a drop-out ribbon
EXIT_PAD = 10         # gap (in persons) between lane and first exit
EXIT_STEP = 4         # gap (in persons) between two exits
MIN_LABEL_H = 7       # min node height (persons) to print the number inside
CENTRE_GAP = 0        # persons of white space at the centre line (0 = touching)

OUT = "./participant_flow_mirrored"
# -----------------------------------------------------------------------------


def band(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha=1.0, z=1):
    """Bezier ribbon between two vertical edges."""
    cx0, cx1 = x0 + (x1 - x0) * 0.5, x1 - (x1 - x0) * 0.5
    verts = [(x0, y0a),
             (cx0, y0a), (cx1, y1a), (x1, y1a),
             (x1, y1b),
             (cx1, y1b), (cx0, y0b), (x0, y0b),
             (x0, y0a)]
    codes = [Path.MOVETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CLOSEPOLY]
    ax.add_patch(patches.PathPatch(Path(verts, codes), fc=color, ec="none",
                                   alpha=alpha, zorder=z))


def draw(groups, stages):
    span = sum(g["n"][0] + EXIT_PAD
               + max(g["n"][0] - g["n"][-1], 10)
               + EXIT_STEP * (len(stages) - 1) for g in groups)
    fig, ax = plt.subplots(figsize=(10, max(4.5, span / 26)))

    y_lo, y_hi = 0.0, 0.0          # outermost extents actually used

    for g in groups:
        n, col, s = g["n"], g["color"], g["side"]
        base = s * CENTRE_GAP / 2                       # centre line for lane
        exit_y = base + s * (n[0] + EXIT_PAD)

        for i in range(len(stages)):
            x = i * STAGE_DX
            y_top = min(base, base + s * n[i])          # rect anchor
            ax.add_patch(patches.Rectangle((x, y_top), NODE_W, n[i],
                                           fc=col, ec="none", zorder=3))

            if n[i] >= MIN_LABEL_H:
                ax.text(x + NODE_W / 2, base + s * n[i] / 2, str(n[i]),
                        ha="center", va="center", color="white",
                        fontsize=13, fontweight="bold", zorder=4)
            else:
                ax.text(x + NODE_W / 2, base + s * (n[i] + 2), str(n[i]),
                        ha="center", va="bottom" if s < 0 else "top",
                        color=col, fontsize=12, fontweight="bold", zorder=4)

            y_lo = min(y_lo, base + s * n[i])
            y_hi = max(y_hi, base + s * n[i])

            if i == len(stages) - 1:
                break

            # continuing flow: hugs the centre line
            band(ax, x + NODE_W, (i + 1) * STAGE_DX,
                 base, base + s * n[i + 1],
                 base, base + s * n[i + 1],
                 col, alpha=FLOW_ALPHA, z=2)

            # loss: the outer slice peels away
            lost = n[i] - n[i + 1]
            if lost > 0:
                band(ax, x + NODE_W, x + NODE_W + EXIT_DX,
                     base + s * n[i + 1], base + s * n[i],
                     exit_y, exit_y + s * lost,
                     col, z=1, alpha=LOST_ALPHA)
                reason = g["reasons"][i] if i < len(g.get("reasons", [])) else ""
                lab = f"\u2212{lost}" + (f"  {reason}" if reason else "")
                ax.text(x + NODE_W + EXIT_DX + 0.04, exit_y + s * lost / 2, lab,
                        ha="left", va="center", color=TEXT_DARK, fontsize=9.5)
                exit_y += s * (lost + EXIT_STEP)
                y_lo, y_hi = min(y_lo, exit_y), max(y_hi, exit_y)

        # lane label
        ax.text(-0.12, base + s * n[0] / 2, f"{g['label']}\nn = {n[0]}",
                ha="right", va="center", color=TEXT_DARK, fontsize=11,
                linespacing=1.4)

    # stage headers above everything
    for i, s_name in enumerate(stages):
        ax.text(i * STAGE_DX + NODE_W / 2, y_lo - 12, s_name,
                ha="center", va="bottom", color=TEXT_DARK, fontsize=12)

    ax.set_xlim(-1.05, (len(stages) - 1) * STAGE_DX + NODE_W + 1.15)
    ax.set_ylim(y_hi + 8, y_lo - 20)      # inverted: negative side on top
    ax.axis("off")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = draw(GROUPS, STAGES)
    fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT + ".svg", bbox_inches="tight", facecolor="white")
    print("written:", OUT + ".png / .svg")
