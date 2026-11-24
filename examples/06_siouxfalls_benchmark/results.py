"""Side-by-side table of the 24 cases, per-intervention summary and the predicted-vs-reference figure."""
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from reference import RUNS  # noqa: E402

MODELS = [("cubic", "Cubic (city configuration)", "#2a78d6", "o", 4.6),
          ("bpr4", "Quartic BPR", "#eb6834", "s", 4.2)]
PANELS = [("cap+25%", "(a) Capacity +25%"), ("cap-50%", "(b) Capacity $-$50%"), ("closure", "(c) Full closure"),
          ("toll", "(d) Toll")]
ORDER = {k: i for i, (k, _) in enumerate(PANELS)}
INK, MUTED, GRID = "#222222", "#555555", "#e6e6e6"


def side_by_side(data):
    key = ["case", "level"]
    first = data[MODELS[0][0]]
    out = first[key + ["link_index", "intervention", "sttt_ref_base", "dsttt_ref"]].copy()
    out["dsttt_ref_pct_of_base"] = 100 * out.dsttt_ref / out.sttt_ref_base
    for m, *_ in MODELS:
        d = data[m]
        assert (d[key].values == out[key].values).all() and (d.dsttt_ref.values == out.dsttt_ref.values).all()
        out[f"{m}_dsttt_pred"] = d.dsttt_pred.values
        out[f"{m}_rel_error"] = d.rel_error.values
        out[f"{m}_error_pct_of_base"] = 100 * d.error_over_base.values
        out[f"{m}_sign_ok"] = d.sign_ok.values
    return out


def summary(cases):
    rows = []
    groups = sorted(cases.groupby("intervention"), key=lambda g: ORDER[g[0]]) + [("all 24", cases)]
    for name, g in groups:
        for m, label, *_ in MODELS:
            pred, ref = g[f"{m}_dsttt_pred"], g.dsttt_ref
            r, e = g[f"{m}_rel_error"], g[f"{m}_error_pct_of_base"]
            rows.append({"intervention": name, "model": label, "n": len(g), "sign": f"{int(g[f'{m}_sign_ok'].sum())}/{len(g)}",
                         "median pred/ref": (pred / ref).median(), "median |rel. error|": r.abs().median(),
                         "max |rel. error|": r.abs().max(),
                         "|effect| under-predicted": f"{int((pred.abs() < ref.abs()).sum())}/{len(g)}",
                         "median |error| (% base)": e.abs().median(), "max |error| (% base)": e.abs().max()})
    return pd.DataFrame(rows)


def figure(data, stem):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Nimbus Roman", "STIXGeneral", "Liberation Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "out", "ytick.direction": "out",
        "axes.unicode_minus": True, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 600})
    fig, axs = plt.subplots(1, 4, figsize=(6.5, 2.05))
    for ax, (key, title) in zip(axs, PANELS):
        vals = []
        for m, _, col, mk, ms in MODELS:
            g = data[m][data[m].intervention == key]
            # both axes in % of the reference baseline STTT, so the distance from the diagonal is the error
            x = 100 * g.dsttt_ref / g.sttt_ref_base; y = 100 * g.dsttt_pred / g.sttt_ref_base
            vals += list(x) + list(y)
            for lvl, face in [(1.0, col), (0.8, "white")]:
                sel = (g.level == lvl).values
                ax.plot(x[sel], y[sel], ls="", marker=mk, ms=ms, mfc=face, mec=col, mew=0.9, zorder=3)
        lo, hi = min(0.0, min(vals)), max(0.0, max(vals))
        pad = 0.07 * (hi - lo); lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK)
        ax.tick_params(colors=INK)
        ax.set_title(title, loc="left", pad=3, color=INK)
    axs[0].set_ylabel("Predicted $\\Delta$STTT (% of baseline)", color=INK)
    fig.supxlabel("Reference $\\Delta$STTT (% of baseline)", fontsize=8, color=INK, y=0.06)
    handles = [Line2D([], [], ls="", marker=mk, ms=ms, mfc=col, mec=col, mew=0.9, label=lab) for _, lab, col, mk, ms in MODELS]
    handles += [Line2D([], [], ls="", marker="o", ms=4.4, mfc=MUTED, mec=MUTED, mew=0.9, label="Demand $q$"),
                Line2D([], [], ls="", marker="o", ms=4.4, mfc="white", mec=MUTED, mew=0.9, label="Demand $0.8q$"),
                Line2D([], [], color=MUTED, lw=0.7, ls=(0, (3, 2)), label="Predicted = reference")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.3, columnspacing=1.1, borderaxespad=0.1)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.2, top=0.8, wspace=0.36)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.02)


def main():
    data = {m: pd.read_csv(os.path.join(RUNS, m, "cases.csv")) for m, *_ in MODELS}
    cases = side_by_side(data)
    cases.to_csv(os.path.join(RUNS, "cases_side_by_side.csv"), index=False)
    summ = summary(cases)
    summ.to_csv(os.path.join(RUNS, "summary_by_intervention.csv"), index=False)
    figure(data, os.path.join(RUNS, "pred_vs_ref"))
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    show = cases[["case", "level", "dsttt_ref", "dsttt_ref_pct_of_base"]
                 + [f"{m}_{c}" for m, *_ in MODELS for c in ("dsttt_pred", "rel_error", "error_pct_of_base")]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:,.3f}" if abs(v) < 1e3 else f"{v:,.0f}"))
    print()
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {os.path.join('runs', 'cases_side_by_side.csv')}, {os.path.join('runs', 'summary_by_intervention.csv')} "
          f"and {os.path.join('runs', 'pred_vs_ref')}.{{pdf,png,svg}}")


if __name__ == "__main__":
    main()
