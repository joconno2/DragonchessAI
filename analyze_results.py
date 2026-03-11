"""
Analysis and figure generation for Monolithic vs CC-CMA-ES experiment.

Produces paper-ready figures:
  1. Convergence curves (win rate vs generation, mean ± 95% CI)
  2. Final win rate distribution (violin + box)
  3. Evolved piece value comparison (bar chart with error bars)

Usage:
    python3 analyze_results.py                          # default dirs
    python3 analyze_results.py --mono results/monolithic/ --cc results/cc/
    python3 analyze_results.py --out figures/           # save to dir
"""

import json
import os
import argparse
import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---- Aesthetics -------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "lines.linewidth":  1.5,
    "patch.linewidth":  0.8,
})

# Color-blind safe palette (Wong 2011)
C_MONO = "#0072B2"   # blue
C_CC   = "#D55E00"   # vermilion
C_REF  = "#009E73"   # green (Jackman reference)

PIECE_NAMES = [
    "Sylph", "Griffin", "Dragon", "Oliphant", "Unicorn",
    "Hero", "Thief", "Cleric", "Mage",
    "Paladin", "Warrior", "Basilisk", "Elemental", "Dwarf"
]
# Jackman (1997) piece values in same order
JACKMAN_VALUES = [1.0, 2.0, 9.0, 5.0, 8.0, 5.0, 2.5, 4.5, 4.0, 8.0, 1.0, 3.0, 4.0, 2.0]
TIER_BOUNDARIES = [3, 11]  # indices where Ground and Underworld start


# ---- Data loading -----------------------------------------------------

def load_runs(directory):
    """Load all run_*.json files from a directory. Returns list of dicts."""
    runs = []
    if not os.path.isdir(directory):
        return runs
    for fname in sorted(os.listdir(directory)):
        if fname.startswith("run_") and fname.endswith(".json"):
            with open(os.path.join(directory, fname)) as f:
                runs.append(json.load(f))
    return runs


def pad_logs(runs, max_gen=None):
    """Stack fitness_log arrays; shorter runs leave NaN after they stop."""
    logs = [np.array(r["fitness_log"]) for r in runs]
    if max_gen is None:
        max_gen = max(len(l) for l in logs)
    padded = np.full((len(logs), max_gen), np.nan)
    for i, log in enumerate(logs):
        n = min(len(log), max_gen)
        padded[i, :n] = log[:n]
    return padded


# ---- Statistical tests ------------------------------------------------

def wilcoxon_test(a, b):
    """Two-sided Wilcoxon rank-sum (Mann-Whitney U) test + rank-biserial r."""
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    # Rank-biserial correlation as effect size (positive = a > b)
    r = 1 - (2 * stat) / (n1 * n2)
    return p, r


def wilcoxon_directional(a, b, alternative="greater"):
    """One-sided Mann-Whitney U: tests whether a > b (default)."""
    stat, p = stats.mannwhitneyu(a, b, alternative=alternative)
    n1, n2 = len(a), len(b)
    r = 1 - (2 * stat) / (n1 * n2)
    return p, r


def cohens_d(a, b):
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0


def ci95(arr, axis=0):
    """95% confidence interval half-width via t-distribution."""
    n = arr.shape[axis]
    se = stats.sem(arr, axis=axis, nan_policy="omit")
    t = stats.t.ppf(0.975, df=n - 1)
    return t * se


# ---- Figures ----------------------------------------------------------

def _opponent_label(runs):
    """Infer opponent description from run metadata for axis labels."""
    if not runs:
        return "Opponent"
    r = runs[0]
    opp = r.get("opponent", "unknown")
    depth = r.get("opponent_depth", None)
    if opp == "alphabeta" and depth is not None:
        return f"AlphaBeta-D{depth}"
    if opp == "greedyvalue":
        return "GreedyValue"
    return opp.capitalize()


def fig_convergence(mono_runs, cc_runs, out_dir=None):
    """Figure 1: Mean win rate vs generation with 95% CI shading."""
    max_gen = max(
        max(r["generations"] for r in mono_runs),
        max(r["generations"] for r in cc_runs),
    )

    mono_mat = pad_logs(mono_runs, max_gen)
    cc_mat   = pad_logs(cc_runs,   max_gen)

    gens = np.arange(1, max_gen + 1)
    mono_mean = np.nanmean(mono_mat, axis=0)
    mono_ci   = ci95(mono_mat)
    cc_mean   = np.nanmean(cc_mat, axis=0)
    cc_ci     = ci95(cc_mat)

    opp_label = _opponent_label(mono_runs or cc_runs)
    n_mono, n_cc = len(mono_runs), len(cc_runs)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    ax.plot(gens, mono_mean, color=C_MONO, label=f"Monolithic (n={n_mono})")
    ax.fill_between(gens, mono_mean - mono_ci, mono_mean + mono_ci,
                    color=C_MONO, alpha=0.18)

    ax.plot(gens, cc_mean, color=C_CC, label=f"CC-CMA-ES (n={n_cc})")
    ax.fill_between(gens, cc_mean - cc_ci, cc_mean + cc_ci,
                    color=C_CC, alpha=0.18)

    ax.set_xlabel("Generation")
    ax.set_ylabel(f"Win Rate vs. {opp_label}")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, max_gen)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Convergence of Evolved Evaluation Functions")
    fig.tight_layout()

    _save(fig, "fig1_convergence", out_dir)
    return fig


def fig_final_distribution(mono_runs, cc_runs, out_dir=None):
    """Figure 2: Final win rate distributions — violin + swarm."""
    mono_wr = np.array([r["best_win_rate"] for r in mono_runs])
    cc_wr   = np.array([r["best_win_rate"] for r in cc_runs])

    p, r_rb = wilcoxon_test(mono_wr, cc_wr)
    d = cohens_d(mono_wr, cc_wr)

    fig, ax = plt.subplots(figsize=(2.8, 2.8))

    data   = [mono_wr, cc_wr]
    colors = [C_MONO, C_CC]
    labels = ["Monolithic", "CC-CMA-ES"]
    pos    = [1, 2]

    parts = ax.violinplot(data, positions=pos, showmedians=False,
                          showextrema=False, widths=0.6)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.35)
        pc.set_edgecolor(c)

    ax.boxplot(data, positions=pos, widths=0.18,
               medianprops=dict(color="black", linewidth=1.5),
               whiskerprops=dict(linewidth=0.8),
               capprops=dict(linewidth=0.8),
               flierprops=dict(marker="o", markersize=3, alpha=0.5))

    # Individual points (jittered)
    rng = np.random.default_rng(42)
    for d_arr, p_pos, c in zip(data, pos, colors):
        jitter = rng.uniform(-0.08, 0.08, size=len(d_arr))
        ax.scatter(p_pos + jitter, d_arr, color=c, s=12, alpha=0.6, zorder=3)

    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    opp_label = _opponent_label(mono_runs or cc_runs)
    ax.set_ylabel(f"Best Win Rate vs. {opp_label}")
    ax.set_ylim(-0.05, 1.1)

    # Significance annotation
    sig_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
    r_str   = f"r = {r_rb:.2f}"
    ax.text(1.5, 1.06, f"{sig_str}, {r_str}",
            ha="center", va="center", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.7))
    ax.plot([1, 1, 2, 2], [1.01, 1.03, 1.03, 1.01], lw=0.8, color="0.3")

    ax.set_title("Final Win Rate Distribution")
    fig.tight_layout()

    _save(fig, "fig2_distribution", out_dir)

    print(f"\n=== Statistical Tests ===")
    print(f"Monolithic: mean={mono_wr.mean():.3f}  std={mono_wr.std():.3f}  "
          f"median={np.median(mono_wr):.3f}")
    print(f"CC-CMA-ES:  mean={cc_wr.mean():.3f}  std={cc_wr.std():.3f}  "
          f"median={np.median(cc_wr):.3f}")
    print(f"Wilcoxon rank-sum: {sig_str}  rank-biserial r = {r_rb:.3f}")
    print(f"Cohen's d = {d:.3f}")

    return fig


def fig_piece_values(mono_runs, cc_runs, out_dir=None):
    """Figure 3: Mean evolved piece values vs Jackman baseline."""
    mono_weights = np.array([r["best_weights"] for r in mono_runs])
    cc_weights   = np.array([r["best_weights"] for r in cc_runs])

    mono_mean = mono_weights.mean(axis=0)
    mono_sem  = stats.sem(mono_weights, axis=0)
    cc_mean   = cc_weights.mean(axis=0)
    cc_sem    = stats.sem(cc_weights, axis=0)
    jackman   = np.array(JACKMAN_VALUES)

    n = len(PIECE_NAMES)
    x = np.arange(n)
    w = 0.25

    fig, ax = plt.subplots(figsize=(7.16, 2.8))

    ax.bar(x - w,   jackman,   width=w, color=C_REF,  alpha=0.75, label="Jackman (1997)")
    ax.bar(x,       mono_mean, width=w, color=C_MONO, alpha=0.85, label="Monolithic",
           yerr=mono_sem, error_kw=dict(elinewidth=0.8, capsize=2))
    ax.bar(x + w,   cc_mean,   width=w, color=C_CC,   alpha=0.85, label="CC-CMA-ES",
           yerr=cc_sem, error_kw=dict(elinewidth=0.8, capsize=2))

    ax.set_xticks(x)
    ax.set_xticklabels(PIECE_NAMES, rotation=35, ha="right")
    ax.set_ylabel("Piece Value")
    n_str = f"mono n={len(mono_runs)}, cc n={len(cc_runs)}"
    ax.set_title(f"Evolved Piece Values vs. Jackman Baseline (mean ± SE, {n_str})")
    ax.legend(frameon=False, loc="upper right")

    # Tier boundary shading
    tier_labels = ["Sky", "Ground", "Underworld"]
    tier_ranges = [(0, 3), (3, 11), (11, 14)]
    tier_colors = ["#e8f4f8", "#f8f0e8", "#f0e8f8"]
    for (lo, hi), tc, tl in zip(tier_ranges, tier_colors, tier_labels):
        ax.axvspan(lo - 0.5, hi - 0.5, alpha=0.35, color=tc, zorder=0)
        ax.text((lo + hi) / 2 - 0.5, ax.get_ylim()[1] * 0.96, tl,
                ha="center", va="top", fontsize=7, color="0.5")

    fig.tight_layout()
    _save(fig, "fig3_piece_values", out_dir)
    return fig


# ---- Helpers ----------------------------------------------------------

def _save(fig, name, out_dir):
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for ext in ("pdf", "png"):
            path = os.path.join(out_dir, f"{name}.{ext}")
            fig.savefig(path)
            print(f"Saved {path}")


def print_summary(mono_runs, cc_runs):
    print(f"\n=== Run Summary ===")
    print(f"Monolithic: {len(mono_runs)} runs completed")
    print(f"CC-CMA-ES:  {len(cc_runs)} runs completed")

    if mono_runs:
        mono_wr = [r["best_win_rate"] for r in mono_runs]
        mono_gen = [r["generations"] for r in mono_runs]
        print(f"  Win rate: mean={np.mean(mono_wr):.3f}  std={np.std(mono_wr):.3f}  "
              f"[{min(mono_wr):.3f}, {max(mono_wr):.3f}]")
        print(f"  Generations: mean={np.mean(mono_gen):.1f}  "
              f"[{min(mono_gen)}, {max(mono_gen)}]")

    if cc_runs:
        cc_wr = [r["best_win_rate"] for r in cc_runs]
        cc_gen = [r["generations"] for r in cc_runs]
        print(f"  Win rate: mean={np.mean(cc_wr):.3f}  std={np.std(cc_wr):.3f}  "
              f"[{min(cc_wr):.3f}, {max(cc_wr):.3f}]")
        print(f"  Generations: mean={np.mean(cc_gen):.1f}  "
              f"[{min(cc_gen)}, {max(cc_gen)}]")


# ---- Main -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze monolithic vs CC-CMA-ES results")
    parser.add_argument("--mono", default="results/monolithic/", help="Monolithic results dir")
    parser.add_argument("--cc",   default="results/cc/",         help="CC results dir")
    parser.add_argument("--out",  default="figures/",            help="Output dir for figures")
    parser.add_argument("--show", action="store_true",           help="Display figures interactively")
    args = parser.parse_args()

    mono_runs = load_runs(args.mono)
    cc_runs   = load_runs(args.cc)

    print_summary(mono_runs, cc_runs)

    if not mono_runs and not cc_runs:
        print("No results found. Check --mono and --cc paths.")
        return

    figs = []
    if mono_runs and cc_runs:
        figs.append(fig_convergence(mono_runs, cc_runs, args.out))
        figs.append(fig_final_distribution(mono_runs, cc_runs, args.out))
        figs.append(fig_piece_values(mono_runs, cc_runs, args.out))
    elif mono_runs:
        print("Only monolithic results available — skipping comparative figures.")
    elif cc_runs:
        print("Only CC results available — skipping comparative figures.")

    if args.show:
        plt.show()
    else:
        plt.close("all")
        print(f"\nFigures saved to {args.out}")
        print("Run with --show to display interactively.")


if __name__ == "__main__":
    main()
