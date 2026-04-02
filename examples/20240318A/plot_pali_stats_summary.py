#!/usr/bin/env python3
"""Generate a publication-style summary plot for PA-L/I statistics.

The script uses the current table values discussed in the manuscript and creates
one figure with:
  1) Spearman rho_s (with bootstrap CI) and partial Spearman rho_{s|t}
  2) Time-aware significance as -log10(p_perm)

Usage:
  python examples/20240318A/plot_pali_stats_summary.py
  python examples/20240318A/plot_pali_stats_summary.py --out output/pali_stats_summary.pdf
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Row:
    label: str
    tau_ms: str
    width_frac: str
    rho_s: float
    p_perm: float
    ci_lo: float
    ci_hi: float
    rho_partial: float
    group: str


ROWS = [
    Row("No noise", "0", "0.05-0.20", -0.302, 4.57e-1, -0.404, -0.193, -0.239, "tau0"),
    #Row("No noise", "0", "0.20-0.40", -0.746, 2.62e-3, -0.801, -0.680, -0.646, "tau0"),
    #Row("No noise", "0", "0.60-0.80", -0.796, 2.28e-3, -0.846, -0.737, -0.087, "tau0"),
    Row("No noise", "0.128", "0.05-0.20", -0.878, 4.79e-3, -0.909, -0.844, -0.881, "tau_sc"),
    Row("S/N~110", "0.128", "0.05-0.20", -0.009, 8.47e-1, -0.096, 0.079, -0.007, "tau_sc_noise"),
    Row("Real", "--", "--", 0.069, 4.29e-1, -0.101, 0.231, 0.089, "real"),
]


GROUP_COLORS = {
    # Okabe-Ito inspired, colorblind-safe palette
    "tau0": "#0072B2",          # blue
    "tau_sc": "#009E73",       # bluish green
    "tau_sc_noise": "#D55E00", # vermillion
    "real": "#CC79A7",         # reddish purple
}


def _set_pub_style(use_latex: bool = True) -> None:
    """Use conservative publication-oriented Matplotlib defaults."""
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.linewidth": 0.8,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": use_latex,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        }
    )


def _format_xtick(row: Row) -> str:
    if row.label == "No noise":
        return f"NN\n$\\tau$={row.tau_ms}"#,\n $w/W_0$={row.width_frac}"
    if row.label == "S/N~110":
        return f"S/N 110\n$\\tau$={row.tau_ms}"#,\n $w/W_0$={row.width_frac}"
    return "Real data"


def build_plot(rows: list[Row], out_path: Path, use_latex: bool = True) -> None:
    _set_pub_style(use_latex=use_latex)

    labels = [_format_xtick(r) for r in rows]
    x = np.arange(len(rows))

    rho_s = np.array([r.rho_s for r in rows])
    rho_partial = np.array([r.rho_partial for r in rows])
    ci_lo = np.array([r.ci_lo for r in rows])
    ci_hi = np.array([r.ci_hi for r in rows])
    p_perm = np.array([r.p_perm for r in rows])

    yerr = np.vstack((rho_s - ci_lo, ci_hi - rho_s))
    sig = -np.log10(p_perm)
    sig_line = -np.log10(0.05)

    colors = [GROUP_COLORS[r.group] for r in rows]

    # ~3.45 in width fits a single column in a two-column layout.
    fig = plt.figure(figsize=(3.45, 4), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.06)

    # Top panel: effect sizes
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.errorbar(
        x,
        rho_s,
        yerr=yerr,
        fmt="o",
        mfc="white",
        mec="black",
        color="black",
        ecolor="0.15",
        elinewidth=1.0,
        capsize=3,
        markersize=4.0,
        label=r"$\rho_s$ (95\% CI)",
        zorder=3,
    )
    ax0.scatter(
        x,
        rho_partial,
        s=28,
        marker="D",
        c=colors,
        edgecolors="black",
        linewidths=0.6,
        zorder=4,
    )
    ax0.axhline(0.0, color="0.25", lw=0.9, ls="--")
    ax0.set_ylabel(r"Effect size")
    ax0.set_ylim(-1.05, 0.35)
    ax0.grid(axis="y", alpha=0.18, linewidth=0.5)
    ax0.tick_params(axis="x", labelbottom=False)
    #ax0.set_title("(a) Effect Sizes", loc="left", pad=1)

    # Bottom panel: significance
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax1.bar(x, sig, color=colors, edgecolor="black", linewidth=0.7, alpha=0.9)
    ax1.axhline(sig_line, color="crimson", ls="--", lw=0.9, label=r"$p=0.05$")
    ax1.set_ylabel(r"$-\log_{10}(p_{\mathrm{perm}})$")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0, ha="center")
    ax1.grid(axis="y", alpha=0.18, linewidth=0.5)
    #ax1.set_title("(b) Time-aware Significance", loc="left", pad=1)
    ax1.set_ylim(0, max(sig) * 1.22)

    # Annotate raw p-values on bars
    for xi, pval, yi in zip(x, p_perm, sig):
        ax1.text(xi, yi + 0.03, f"{pval:.1e}", ha="center", va="bottom", fontsize=6)

    # Legend for condition groups
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markerfacecolor=c,
            markeredgecolor="black",
            markersize=6.5,
            label=lbl,
        )
        for lbl, c in [
            (r"No noise, $\tau=0$", GROUP_COLORS["tau0"]),
            (r"No noise, $\tau=0.128$", GROUP_COLORS["tau_sc"]),
            (r"S/N 110, $\tau=0.128$", GROUP_COLORS["tau_sc_noise"]),
            ("Real", GROUP_COLORS["real"]),
        ]
    ]
    top_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5,
            label=r"$\rho_s$ (95\% CI)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5,
            label=r"$\rho_{s|t}$",
        ),
    ]
    ax0.legend(handles=top_handles, loc="upper left", frameon=False, handlelength=1.0, fontsize=7)
    ax1.legend(handles=handles, loc="upper right", frameon=False, fontsize=6.5, handlelength=1.0)

    #fig.suptitle(r"PA-L/I Summary Statistics", y=0.995)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    except Exception as exc:
        if use_latex:
            print(f"LaTeX rendering failed ({exc}); retrying with Matplotlib mathtext.")
            plt.close(fig)
            build_plot(rows, out_path, use_latex=False)
            return
        raise
    print(f"Saved: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PA-L/I summary statistics.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/pali_stats_summary.pdf"),
        help="Output figure path (default: output/pali_stats_summary.pdf)",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also write a PNG with the same basename.",
    )
    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="Disable LaTeX text rendering and use Matplotlib mathtext.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_latex = not args.no_latex
    build_plot(ROWS, args.out, use_latex=use_latex)
    if args.png:
        png_path = args.out.with_suffix(".png")
        build_plot(ROWS, png_path, use_latex=use_latex)


if __name__ == "__main__":
    main()
