"""Shared figure style: palette, rcParams and saving.

Colors follow the validated categorical palette (light mode) in fixed order.
Colorblind check (adjacent pairs, lines): slots 1-6 pass; slots 3, 4, 5 sit below
3:1 contrast on white, so every figure has a legend and every number is in a table.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots, fixed order. LNS configurations take slots in config order.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
               "#6250d6", "#e34948"]
# Baselines are neutral so they never clash with an LNS hue.
CP_COLOR = "#3d3c39"         # near-black ink, solid line
HEURISTIC_COLOR = "#8a8985"  # muted gray, dashed reference line
HEURISTIC_MARK = "#e34948"   # red star where a heuristic finishes
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GRID_STRONG = "#c3c2b7"

# Time-composition components of one LNS iteration.
TIME_PARTS = [  # (column, label, color)
    ("lns_strategy_time", "Neighbourhood selection", "#e87ba4"),
    ("constr_generate_time", "Constraint generation", "#eda100"),
    ("constr_update_time", "Model update", "#008300"),
    ("solve_api_overhead", "Solver API overhead", "#8a8985"),
    ("solve_cpp_presolve", "CP extraction / presolve", "#eb6834"),
    ("solve_cpp_search", "CP search", "#2a78d6"),
    ("merge_time", "Merge", "#6250d6"),
]
OTHER_COLOR = "#d8d7d2"

# Sequential blue ramp (light -> dark) for heatmaps.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
             "#0d366b"]


def apply():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
        "lines.solid_capstyle": "round", "savefig.bbox": "tight", "savefig.dpi": 200,
    })


def save(fig, out_dir, name, formats=("pdf", "png")):
    os.makedirs(out_dir, exist_ok=True)
    for ext in formats:
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)
