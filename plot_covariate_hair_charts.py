############################################################################################
#   Chronos2 : hair charts des prévisions avec covariables vs sans covariables
#                                   - Anna Petronevich
#                         -----
#    Reads the forecasts saved by covariate_selection.py (no model is run) and draws, for
#    each run (level, qoq, qoq_cov), one panel per covariate set on shared axes: one hair
#    per cut-off (median forecast 1..12 quarters ahead, starting from the observed value
#    at the cut-off) against the observed target. The vertical line marks the start of
#    the evaluation period; the shaded area, the quarters not final yet.
#    Settings (input file, periods, output folder) come from covariate_selection.py.
#
#    SETS_TO_PLOT : "none" (pas de covariables), "selected" (choisi sur la période de
#    sélection), "best" (meilleur sur la période d'évaluation : choisi a posteriori), ou le
#    nom exact d'un ensemble (colonne "set" de la feuille summary, sans "[selected]").
#    Outputs : OUTPUT_DIR/<ASSUMPTIONS_MODE>/<run>/hair_charts_covariates.png
#                         -----
#  A.Petronevich et AI
############################################################################################


############### Settings ##################################################################

SETS_TO_PLOT = ["none", "selected", "best"]    # e.g. ["none", "fr_wdr (assumptions past-only)"]
RUNS_TO_PLOT = None             # None = every run of covariate_selection.py; e.g. ["qoq_cov"]
# Folder holding the <run>/covariate_selection.xlsx files. None = the folder of the current
# ASSUMPTIONS_MODE of covariate_selection.py (Outputs/covariate_selection/<mode>). Results
# made before the assumption modes existed are in "Outputs/covariate_selection".
RESULTS_FOLDER = None
HAIR_EVERY = 1                  # 4 = one hair per year, easier to read
START_YEARS_BEFORE = 2          # observed series shown from this many years before the first cut-off


############### Get installed #####################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import covariate_selection as cs    # same data and settings as the selection test

COLORS = {"none": "#8c8c8c"}        # zero-shot without covariates in grey
COVARIATE_COLOR = "#2a6fdb"


#####################  Get the data  ######################################################

def run_dir(run):
    return os.path.join(RESULTS_FOLDER, run) if RESULTS_FOLDER else cs.results_dir(run)


def load_results(run):
    path = os.path.join(run_dir(run), "covariate_selection.xlsx")
    if not os.path.exists(path):
        return None, None
    forecasts = pd.read_excel(path, sheet_name="forecasts", parse_dates=["cutoff"])
    summary = pd.read_excel(path, sheet_name="summary")
    summary["name"] = summary["set"].str.replace(r"\s*\[selected\]", "", regex=True)
    return forecasts, summary


def resolve_sets(summary):
    """(set name in the forecasts sheet, panel title) for each entry of SETS_TO_PLOT."""
    ratio = dict(zip(summary["name"], summary["average MAFE ratio, evaluation"]))
    out = []
    for wanted in SETS_TO_PLOT:
        if wanted == "selected":
            rows = summary[summary["set"].str.contains(r"\[selected\]")]
            if rows.empty:
                continue
            name, how = rows["name"].iloc[0], "selected on the selection period"
        elif wanted == "best":
            others = summary[summary["name"] != "none"]
            name = others.loc[others["average MAFE ratio, evaluation"].idxmin(), "name"]
            how = "best on the evaluation period (chosen after the fact)"
        elif wanted == "none":
            name, how = "none", "no covariates"
        else:
            name, how = wanted, "chosen by hand"
        if name not in ratio:
            print(f"  {name!r} is not among the tested sets, skipped")
            continue
        if any(name == n for n, _ in out):
            continue    # e.g. the selected set is also the best one
        title = f"{name}: {how}" if name != "none" else "No covariates (zero-shot)"
        if name != "none":
            title += f" - MAFE ratio to no covariates, evaluation: {ratio[name]:.2f}"
        out.append((name, title))
    return out


def observed_series(run):
    """The target as the selection test saw it, dated at the start of each quarter."""
    data = cs.load_data()
    data = data[data["DATE"].dt.to_period("Q") <= cs.quarter(cs.LAST_OBSERVED)]
    y = data[cs.TARGET_COLUMN]
    if run in ("qoq", "qoq_cov"):
        y = 100 * y.pct_change()
    return pd.Series(y.to_numpy(), index=data["DATE"].dt.to_period("Q")).dropna()


################## Plot #####################################

def plot_run(run, forecasts, sets, observed, path):
    cutoffs = sorted(forecasts["cutoff"].unique())[::HAIR_EVERY]
    forecasts = forecasts[forecasts["cutoff"].isin(cutoffs)].copy()
    forecasts["cutoff_q"] = forecasts["cutoff"].dt.to_period("Q")
    forecasts["x"] = (forecasts["cutoff_q"] + forecasts["horizon"]).dt.to_timestamp()

    start = forecasts["cutoff_q"].min() - 4 * START_YEARS_BEFORE
    shown = observed[observed.index >= start]
    x_obs = shown.index.to_timestamp()
    split = (cs.quarter(cs.SELECTION_END) + 1).to_timestamp()
    not_final = (cs.quarter(cs.LAST_FINAL) + 1).to_timestamp()

    fig, axes = plt.subplots(len(sets), 1, figsize=(12, 3.6 * len(sets) + 1), sharex=True, sharey=True,
                             squeeze=False)
    for ax, (name, title) in zip(axes[:, 0], sets):
        color = COLORS.get(name, COVARIATE_COLOR)
        for cutoff_q, hair in forecasts[forecasts["set"] == name].groupby("cutoff_q"):
            hair = hair.sort_values("horizon")
            x = [cutoff_q.to_timestamp()] + list(hair["x"])
            y = [observed.get(cutoff_q, np.nan)] + list(hair["predictions"])
            ax.plot(x, y, color=color, alpha=0.5, linewidth=1)
        ax.plot([], [], color=color, linewidth=1.5, label="Median forecast from each cut-off")
        ax.plot(x_obs, shown.to_numpy(), color="#1f1f1f", linewidth=2, label="Observed")
        ax.axvline(split, color="#595959", linestyle="--", linewidth=1, label="Start of the evaluation period")
        ax.axvspan(not_final, forecasts["x"].max(), color="#d9d9d9", alpha=0.5, linewidth=0,
                   label="Not final yet: not scored")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(cs.TARGET_COLUMN, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper left", fontsize=8)

    # Growth rates: keep the bulk readable when a few quarters (e.g. Covid) are extreme.
    # The range comes from the observed series only: forecasts cluster near the mean.
    if run in ("qoq", "qoq_cov"):
        lo, hi = np.nanpercentile(shown.to_numpy(), [1, 99])
        everything = np.concatenate([shown.to_numpy(), forecasts["predictions"].to_numpy()])
        if np.nanmin(everything) < lo or np.nanmax(everything) > hi:
            pad = 0.15 * (hi - lo)
            axes[0, 0].set_ylim(lo - pad, hi + pad)
            axes[0, 0].text(0.99, 0.98, "y-axis limited to the 1st-99th percentile range of the observed series",
                            transform=axes[0, 0].transAxes, ha="right", va="top", fontsize=8, color="#595959")

    every = "" if HAIR_EVERY == 1 else f", one cut-off in {HAIR_EVERY}"
    fig.suptitle(f"Chronos2 zero-shot forecasts with and without covariates{every}\n"
                 f"{cs.TARGET_COLUMN}, {cs.unit(run)}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


################## Main #####################################

def main():
    for run in RUNS_TO_PLOT or cs.TRANSFORMS:
        forecasts, summary = load_results(run)
        if forecasts is None:
            print(f"{run}: no results in {run_dir(run)}, run covariate_selection.py first")
            continue
        print(f"{cs.TARGET_COLUMN}, {cs.unit(run)}", flush=True)
        plot_run(run, forecasts, resolve_sets(summary), observed_series(run),
                 os.path.join(run_dir(run), "hair_charts_covariates.png"))


if __name__ == "__main__":
    main()
