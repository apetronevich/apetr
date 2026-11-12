############################################################################################
#   Chronos2 : hair charts des prévisions LoRA et zero-shot - Anna Petronevich
#                         -----
#    Reads the forecasts saved by finetune_lora_backtest.py (no model is run) and draws,
#    for the target in level and in q-o-q growth, one hair per cut-off: the zero-shot
#    forecast path and the LoRA forecast path, against the observed BMPE series. The two
#    panels share the same axes so they can be compared directly.
#    Settings (BMPE file, dates, output folder) are taken from finetune_lora_backtest.py,
#    so both scripts always use the same data.
#
#    Inputs:
#      - OUTPUT_DIR/<level|qoq>/lora_vs_zeroshot.xlsx, written by finetune_lora_backtest.py
#      - HAIR_EVERY : une prévision tracée toutes les HAIR_EVERY coupures (1 = toutes)
#    Outputs (OUTPUT_DIR/<level|qoq>) :
#      - hair_charts.png : zero-shot (haut) et LoRA (bas)
#                         -----
#  A.Petronevich et AI
############################################################################################


############### Settings ##################################################################

HAIR_EVERY = 1                  # 4 = one hair per year, easier to read
START_YEARS_BEFORE = 2          # realised series shown from this many years before the first cut-off


############### Get installed #####################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import finetune_lora_backtest as ft    # same data and settings as the backtest

MODELS = {"zero_shot": ("Zero-shot", "#8c8c8c"), "lora": ("LoRA fine-tuned", "#2a6fdb")}


#####################  Get the data  ######################################################

def load_forecasts(transform):
    path = os.path.join(ft.OUTPUT_DIR, transform, "lora_vs_zeroshot.xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} introuvable : lancer d'abord finetune_lora_backtest.py")
    forecasts = pd.read_excel(path, sheet_name="forecasts", parse_dates=["cutoff", "target_date"])
    # target dates past the last observed quarter are left empty by the backtest
    missing = forecasts["target_date"].isna()
    forecasts.loc[missing, "target_date"] = [c + pd.DateOffset(months=3 * h) for c, h in
                                             zip(forecasts.loc[missing, "cutoff"], forecasts.loc[missing, "horizon"])]
    cutoffs = sorted(forecasts["cutoff"].unique())[::HAIR_EVERY]
    return forecasts[forecasts["cutoff"].isin(cutoffs)]


def load_observed(transform):
    """The target as the backtest saw it: BMPE up to LAST_OBSERVED, same transformation."""
    panel = ft.transform_panel(ft.load_bmpe(), transform)
    return panel[["DATE", ft.TARGET_COLUMN]].dropna().reset_index(drop=True)


################## Plot #####################################

def plot_hair_charts(forecasts, observed, transform, path):
    start = forecasts["cutoff"].min() - pd.DateOffset(years=START_YEARS_BEFORE)
    shown = observed[observed["DATE"] >= start]
    last_observed = observed.set_index("DATE")[ft.TARGET_COLUMN]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, sharey=True)
    for ax, (model, (label, color)) in zip(axes, MODELS.items()):
        for cutoff, hair in forecasts.groupby("cutoff"):
            hair = hair.sort_values("horizon")
            # each hair starts at the last observed value of its cut-off
            x = [cutoff] + list(hair["target_date"])
            y = [last_observed.get(cutoff, np.nan)] + list(hair[model])
            ax.plot(x, y, color=color, alpha=0.55, linewidth=1)
        ax.plot([], [], color=color, linewidth=1.5, label=f"{label}, median forecast from each cut-off")
        ax.plot(shown["DATE"], shown[ft.TARGET_COLUMN], color="#1f1f1f", linewidth=2, label="Realised (BMPE)")

        provisional = observed["DATE"][observed["DATE"].dt.to_period("Q") > ft.quarter(ft.LAST_FINAL)] \
            if ft.LAST_FINAL else pd.Series(dtype="datetime64[ns]")
        if len(provisional):
            ax.axvspan(provisional.min(), forecasts["target_date"].max(), color="#d9d9d9", alpha=0.5, linewidth=0,
                       label=f"Not final yet (from {ft.quarter_label(provisional.min())}): not scored")
        ax.set_title(label, fontsize=11)
        ax.set_ylabel(f"{ft.TARGET_COLUMN}\n({ft.unit(transform)})")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    # Growth rates: keep the bulk readable when a few quarters (e.g. Covid) are extreme.
    # The range comes from the realised series only: forecasts cluster near the mean.
    everything = np.concatenate([shown[ft.TARGET_COLUMN].to_numpy(),
                                 forecasts[list(MODELS)].to_numpy().ravel()])
    lo, hi = np.nanpercentile(shown[ft.TARGET_COLUMN], [1, 99])
    if transform == "qoq" and (np.nanmin(everything) < lo or np.nanmax(everything) > hi):
        pad = 0.15 * (hi - lo)
        axes[0].set_ylim(lo - pad, hi + pad)
        axes[0].text(0.99, 0.98, "y-axis limited to the 1st-99th percentile range of the realised series",
                     transform=axes[0].transAxes, ha="right", va="top", fontsize=8, color="#595959")

    every = "" if HAIR_EVERY == 1 else f", one cut-off in {HAIR_EVERY}"
    fig.suptitle(f"Chronos2 forecasts from each cut-off{every}: zero-shot vs LoRA fine-tuned, "
                 f"{ft.TARGET_COLUMN} ({ft.unit(transform)})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


################## Main #####################################

def main():
    for transform in ft.TRANSFORMS:
        print(f"{ft.TARGET_COLUMN}, {ft.unit(transform)}", flush=True)
        plot_hair_charts(load_forecasts(transform), load_observed(transform), transform,
                         os.path.join(ft.OUTPUT_DIR, transform, "hair_charts.png"))


if __name__ == "__main__":
    main()
