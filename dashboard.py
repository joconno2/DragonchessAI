"""
DragonchessAI Experiment Dashboard
====================================
Lightweight single-figure overview of current experiment state.
Works with partial results (0–30 runs in each condition).

Usage:
    python3 dashboard.py                     # save to figures/dashboard.png + .pdf
    python3 dashboard.py --show              # display interactively
    python3 dashboard.py --mono results/monolithic/ --cc results/cc/
    python3 dashboard.py --out figures/      # custom output dir
"""

import json
import os
import argparse
import datetime
import numpy as np
from scipy import stats
import matplotlib
matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        120,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "lines.linewidth":   1.4,
})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Wong (2011) color-blind safe
C_MONO = "#0072B2"   # blue
C_CC   = "#D55E00"   # vermilion
C_REF  = "#009E73"   # green

PIECE_NAMES = [
    "Sylph", "Griffin", "Dragon", "Oliphant", "Unicorn",
    "Hero", "Thief", "Cleric", "Mage",
    "Paladin", "Warrior", "Basilisk", "Elemental", "Dwarf"
]
JACKMAN = [1.0, 2.0, 9.0, 5.0, 8.0, 5.0, 2.5, 4.5, 4.0, 8.0, 1.0, 3.0, 4.0, 2.0]
TARGET_RUNS = 30


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_runs(directory):
    runs = []
    if not os.path.isdir(directory):
        return runs
    for fname in sorted(os.listdir(directory)):
        if fname.startswith("run_") and fname.endswith(".json"):
            with open(os.path.join(directory, fname)) as f:
                runs.append(json.load(f))
    return runs


def pad_logs_nan(runs, max_gen=None):
    """Stack fitness logs as matrix; shorter runs stay NaN after they end."""
    logs = [np.array(r["fitness_log"]) for r in runs]
    if max_gen is None:
        max_gen = max(len(l) for l in logs)
    mat = np.full((len(logs), max_gen), np.nan)
    for i, log in enumerate(logs):
        n = min(len(log), max_gen)
        mat[i, :n] = log[:n]
    return mat


def ci95(arr, axis=0):
    n = arr.shape[axis]
    if n < 2:
        return np.zeros(arr.shape[1] if axis == 0 else arr.shape[0])
    se = stats.sem(arr, axis=axis, nan_policy="omit")
    t  = stats.t.ppf(0.975, df=n - 1)
    return t * se


# ── Panel helpers ─────────────────────────────────────────────────────────────

def panel_status(ax, mono_runs, cc_runs):
    """Left panel: text summary of current experiment state."""
    ax.axis("off")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"Dashboard  {now}",
        "",
        f"{'Condition':<14}{'n':>4}{'Mean':>7}{'Std':>7}{'Best':>7}",
        "─" * 42,
    ]

    for label, runs, color in [("Monolithic", mono_runs, C_MONO), ("CC-CMA-ES", cc_runs, C_CC)]:
        n = len(runs)
        if n > 0:
            wrs = np.array([r["best_win_rate"] for r in runs])
            lines.append(
                f"  {label:<12}{n:>4}{wrs.mean():>7.3f}{wrs.std():>7.3f}{wrs.max():>7.3f}"
            )
        else:
            lines.append(f"  {label:<12}{n:>4}{'—':>7}{'—':>7}{'—':>7}")

    # Progress bars
    lines += ["", "Progress (out of 30 target runs):"]
    for label, n in [("Mono", len(mono_runs)), ("CC  ", len(cc_runs))]:
        filled  = int(round(20 * n / TARGET_RUNS))
        empty   = 20 - filled
        pct     = 100 * n / TARGET_RUNS
        lines.append(f"  {label} [{'█'*filled}{'░'*empty}] {n}/{TARGET_RUNS} ({pct:.0f}%)")

    # Stat test if enough data
    if len(mono_runs) >= 5 and len(cc_runs) >= 5:
        mono_wr = np.array([r["best_win_rate"] for r in mono_runs])
        cc_wr   = np.array([r["best_win_rate"] for r in cc_runs])
        _, p    = stats.mannwhitneyu(mono_wr, cc_wr, alternative="two-sided")
        d       = cohens_d(mono_wr, cc_wr)
        sig     = "✓" if p < 0.05 else "○"
        lines  += ["", f"Mann-Whitney p={p:.3f}  Cohen's d={d:.2f}  {sig}"]

    # Generation info
    if mono_runs:
        gens = [r["generations"] for r in mono_runs]
        lines += ["", f"Mono gens: mean={np.mean(gens):.0f} [{min(gens)}, {max(gens)}]"]
    if cc_runs:
        gens = [r["generations"] for r in cc_runs]
        lines.append(f"CC   gens: mean={np.mean(gens):.0f} [{min(gens)}, {max(gens)}]")

    text = "\n".join(lines)
    ax.text(0.04, 0.97, text, transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="0.7", lw=0.8))
    ax.set_title("Experiment Status", fontweight="bold")


def panel_convergence(ax, mono_runs, cc_runs):
    """Middle panel: convergence curves with 95% CI shading."""
    has_mono = len(mono_runs) >= 2
    has_cc   = len(cc_runs)   >= 2

    if not has_mono and not has_cc:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="0.5")
        ax.set_title("Convergence")
        return

    max_gen = max(
        (max(r["generations"] for r in mono_runs) if mono_runs else 0),
        (max(r["generations"] for r in cc_runs)   if cc_runs   else 0),
    )

    if has_mono:
        mat  = pad_logs_nan(mono_runs, max_gen)
        mean = np.nanmean(mat, axis=0)
        ci   = ci95(mat)
        gens = np.arange(1, max_gen + 1)
        ax.plot(gens, mean, color=C_MONO, label=f"Monolithic (n={len(mono_runs)})")
        ax.fill_between(gens, mean - ci, mean + ci, color=C_MONO, alpha=0.18)

    if has_cc:
        mat  = pad_logs_nan(cc_runs, max_gen)
        mean = np.nanmean(mat, axis=0)
        ci   = ci95(mat)
        gens = np.arange(1, max_gen + 1)
        ax.plot(gens, mean, color=C_CC, label=f"CC-CMA-ES (n={len(cc_runs)})")
        ax.fill_between(gens, mean - ci, mean + ci, color=C_CC, alpha=0.18)

    # Infer opponent label from run metadata
    all_runs = list(mono_runs) + list(cc_runs)
    if all_runs:
        r0 = all_runs[0]
        opp = r0.get("opponent", "")
        depth = r0.get("opponent_depth", None)
        if opp == "alphabeta" and depth is not None:
            opp_label = f"AlphaBeta-D{depth}"
        elif opp == "greedyvalue":
            opp_label = "GreedyValue"
        else:
            opp_label = opp.capitalize() if opp else "Opponent"
    else:
        opp_label = "Opponent"

    ax.set_xlabel("Generation")
    ax.set_ylabel(f"Win Rate vs. {opp_label}")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, max_gen)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Convergence (mean ± 95% CI)")


def panel_distribution(ax, mono_runs, cc_runs):
    """Right panel: final win rate distributions."""
    mono_wr = np.array([r["best_win_rate"] for r in mono_runs]) if mono_runs else np.array([])
    cc_wr   = np.array([r["best_win_rate"] for r in cc_runs])   if cc_runs   else np.array([])

    has_violin = len(mono_wr) >= 5 or len(cc_wr) >= 5

    data, colors, labels, pos = [], [], [], []
    if len(mono_wr) > 0:
        data.append(mono_wr); colors.append(C_MONO); labels.append("Mono"); pos.append(1)
    if len(cc_wr) > 0:
        data.append(cc_wr);   colors.append(C_CC);   labels.append("CC");   pos.append(2)

    if not data:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="0.5")
        ax.set_title("Win Rate Distribution")
        return

    if has_violin:
        parts = ax.violinplot(data, positions=pos, showmedians=False,
                              showextrema=False, widths=0.55)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c); pc.set_alpha(0.30); pc.set_edgecolor(c)

    ax.boxplot(data, positions=pos, widths=0.20,
               medianprops=dict(color="black", linewidth=1.5),
               whiskerprops=dict(linewidth=0.8),
               capprops=dict(linewidth=0.8),
               flierprops=dict(marker="o", markersize=3, alpha=0.5))

    rng = np.random.default_rng(42)
    for d_arr, p_pos, c in zip(data, pos, colors):
        jitter = rng.uniform(-0.07, 0.07, size=len(d_arr))
        ax.scatter(p_pos + jitter, d_arr, color=c, s=14, alpha=0.65, zorder=3)

    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Best Win Rate")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("Win Rate Distribution")


def panel_weights(ax, mono_runs, cc_runs):
    """Bottom panel: mean evolved piece values vs Jackman baseline."""
    jackman = np.array(JACKMAN)
    n = len(PIECE_NAMES)
    x = np.arange(n)
    w = 0.25

    # Tier background
    tier_ranges = [(0, 3), (3, 11), (11, 14)]
    tier_colors = ["#e8f4f8", "#f8f0e8", "#f0e8f8"]
    tier_labels = ["Sky", "Ground", "Underworld"]
    for (lo, hi), tc, tl in zip(tier_ranges, tier_colors, tier_labels):
        ax.axvspan(lo - 0.5, hi - 0.5, alpha=0.30, color=tc, zorder=0)

    ax.bar(x - w, jackman, width=w, color=C_REF, alpha=0.70, label="Jackman (1997)")

    if mono_runs:
        mono_w = np.array([r["best_weights"] for r in mono_runs])
        ax.bar(x, mono_w.mean(axis=0), width=w, color=C_MONO, alpha=0.85,
               label=f"Monolithic (n={len(mono_runs)})",
               yerr=stats.sem(mono_w, axis=0) if len(mono_runs) > 1 else None,
               error_kw=dict(elinewidth=0.8, capsize=2))

    if cc_runs:
        cc_w = np.array([r["best_weights"] for r in cc_runs])
        ax.bar(x + w, cc_w.mean(axis=0), width=w, color=C_CC, alpha=0.85,
               label=f"CC-CMA-ES (n={len(cc_runs)})",
               yerr=stats.sem(cc_w, axis=0) if len(cc_runs) > 1 else None,
               error_kw=dict(elinewidth=0.8, capsize=2))

    ax.set_xticks(x)
    ax.set_xticklabels(PIECE_NAMES, rotation=35, ha="right")
    ax.set_ylabel("Piece Value")
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    ax.set_title("Evolved Piece Values vs. Jackman Baseline (mean ± SE)")

    # Tier labels at top
    ylim_top = ax.get_ylim()[1]
    for (lo, hi), tl in zip(tier_ranges, tier_labels):
        ax.text((lo + hi) / 2 - 0.5, ylim_top * 0.95, tl,
                ha="center", va="top", fontsize=7, color="0.45")


def cohens_d(a, b):
    pooled = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def build_dashboard(mono_dir, cc_dir, out_dir=None, show=False):
    mono_runs = load_runs(mono_dir)
    cc_runs   = load_runs(cc_dir)

    print(f"Loaded: {len(mono_runs)} monolithic, {len(cc_runs)} CC runs")

    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            left=0.05, right=0.97,
                            top=0.94, bottom=0.10,
                            hspace=0.38, wspace=0.30)

    ax_status = fig.add_subplot(gs[0, 0])
    ax_conv   = fig.add_subplot(gs[0, 1])
    ax_dist   = fig.add_subplot(gs[0, 2])
    ax_wts    = fig.add_subplot(gs[1, :])

    panel_status(ax_status,  mono_runs, cc_runs)
    panel_convergence(ax_conv, mono_runs, cc_runs)
    panel_distribution(ax_dist, mono_runs, cc_runs)
    panel_weights(ax_wts, mono_runs, cc_runs)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.suptitle(
        f"DragonchessAI — Monolithic vs CC-CMA-ES Experiment  |  {now}",
        fontsize=11, fontweight="bold", y=0.99
    )

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for ext in ("png", "pdf"):
            path = os.path.join(out_dir, f"dashboard.{ext}")
            fig.savefig(path)
            print(f"Saved {path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="DragonchessAI experiment dashboard")
    parser.add_argument("--mono", default="results/monolithic/", help="Monolithic results dir")
    parser.add_argument("--cc",   default="results/cc/",         help="CC results dir")
    parser.add_argument("--out",  default="figures/",            help="Output dir")
    parser.add_argument("--show", action="store_true",           help="Display interactively")
    args = parser.parse_args()

    build_dashboard(args.mono, args.cc, out_dir=args.out, show=args.show)


if __name__ == "__main__":
    main()
