############################################################################################
#   Chronos2 zero-shot : précision selon la date de coupure, la transformation de la
#   cible et l'horizon - Anna Petronevich
#                         -----
#    For every cut-off date from FIRST_CUTOFF on, Chronos2 forecasts TARGET_COLUMN
#    1..PREDICTION_LENGTH quarters ahead, zero-shot (no covariates, no future covariates),
#    from all the history up to the cut-off (expanding window). The exercise is run for
#    the target in level and in q-o-q growth rate. The BMPE series is both the model
#    input and the ground truth.
#
#    Inputs:
#      - chemin vers stockage modèle MODEL_PATH
#      - BMPE_FILE : export excel du BMPE de juin 2026 (colonne DATE + TARGET_COLUMN)
#      - LAST_OBSERVED : dernier trimestre observé du BMPE ; les trimestres suivants sont
#        des projections et sont exclus (ni contexte, ni réalisé)
#    Outputs (OUTPUT_DIR/<level|qoq>) :
#      - zero_shot_cutoff.xlsx : prévisions, erreurs coupure x horizon, métriques
#      - heatmap_<métrique>.png : coupure en vertical, horizon en horizontal
#      - hair_chart.png : réalisé + une prévision ("cheveu") par date de coupure
#                         -----
#  A.Petronevich et AI
############################################################################################


############### Settings ##################################################################

MODEL_PATH = "C:\\TSFM"

BMPE_FILE = "Inputs/BMPE_Juin2026.xlsx"
BMPE_SHEET = 0                  # sheet name or index
DATE_COLUMN = "DATE"            # dates or quarters like "2000Q1"
TARGET_COLUMN = "P51_S11S12S14A_7CH"   # matched ignoring case
LAST_OBSERVED = "2026-01-01"    # 2026Q1 ; None = keep every non-empty row

FIRST_CUTOFF = "2000-01-01"     # 2000Q1, last quarter in the context of the first forecast
PREDICTION_LENGTH = 12          # horizons 1..12
TRANSFORMS = ["level", "qoq"]   # target in level, then in q-o-q growth rate (%)
QUANTILE_LEVELS = [0.1, 0.5, 0.9]

# Colour scales stop at this percentile so a few extreme errors (2008-09, Covid)
# do not wash out the rest of the heatmap; larger values get the end colour.
ROBUST_PERCENTILE = 95

OUTPUT_DIR = "Outputs/zero_shot_cutoff"


############### Get installed #####################

print("Step 1 - Get installed", flush=True)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_model(model_path):
    import torch
    from chronos import Chronos2Pipeline

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    return Chronos2Pipeline.from_pretrained(
        model_path,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        local_files_only=True
    )


#####################  Get the data  ######################################################

def find_column(df, name):
    matches = [c for c in df.columns if str(c).strip().lower() == name.lower()]
    if not matches:
        raise KeyError(f"Colonne {name} introuvable dans {BMPE_FILE}. Colonnes : {list(df.columns)}")
    return matches[0]


def to_dates(values):
    try:
        return pd.to_datetime(values)
    except (ValueError, TypeError):
        return pd.PeriodIndex(values.astype(str), freq="Q").to_timestamp()


def load_ground_truth():
    raw = pd.read_excel(BMPE_FILE, sheet_name=BMPE_SHEET)
    data = pd.DataFrame({
        "DATE": to_dates(raw[find_column(raw, DATE_COLUMN)]),
        "level": pd.to_numeric(raw[find_column(raw, TARGET_COLUMN)], errors="coerce"),
    }).dropna().sort_values("DATE")
    if LAST_OBSERVED is not None:
        data = data[data["DATE"] <= pd.Timestamp(LAST_OBSERVED)]
    data["qoq"] = 100 * data["level"].pct_change()
    return data.reset_index(drop=True)


def quarter_label(date):
    return f"{date.year}Q{date.quarter}"


def unit(transform):
    return "q-o-q growth, %" if transform == "qoq" else "level"


################## Generate predictions ####################

def run_forecasts(pipeline, series):
    """Zero-shot forecasts from every cut-off, expanding window. series has DATE and
    TARGET_COLUMN, one row per quarter."""
    last = len(series) - 1
    cutoffs = series.index[(series["DATE"] >= pd.Timestamp(FIRST_CUTOFF)) & (series.index < last)]
    contexts = pd.concat(
        [series.iloc[:pos + 1].assign(item_id=series["DATE"].iloc[pos].strftime("%Y-%m-%d")) for pos in cutoffs],
        ignore_index=True,
    )
    print(f"  {len(cutoffs)} cut-offs, {quarter_label(series['DATE'].iloc[cutoffs[0]])} "
          f"to {quarter_label(series['DATE'].iloc[cutoffs[-1]])}", flush=True)

    # All cut-offs in a single call, one item_id each
    pred = pipeline.predict_df(
        contexts,
        future_df=None,
        prediction_length=PREDICTION_LENGTH,
        quantile_levels=QUANTILE_LEVELS,
        id_column="item_id",
        timestamp_column="DATE",
        target=TARGET_COLUMN,
    )
    pred = pred.sort_values(["item_id", "DATE"]).reset_index(drop=True)
    pred["horizon"] = pred.groupby("item_id").cumcount() + 1

    # Ground truth matched by position (cut-off + h), not by forecast date
    position_of = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(series["DATE"])}
    cutoff_pos = pred["item_id"].map(position_of).to_numpy()
    truth_pos = cutoff_pos + pred["horizon"].to_numpy()
    target = series[TARGET_COLUMN].to_numpy()
    pred["cutoff"] = pd.to_datetime(pred["item_id"])
    pred["target_date"] = series["DATE"].reindex(truth_pos).to_numpy()
    pred["target_date"] = pred["target_date"].fillna(pd.to_datetime(pred["DATE"]))    # beyond LAST_OBSERVED
    pred["actual"] = np.where(truth_pos <= last, target[np.minimum(truth_pos, last)], np.nan)
    pred["last_observed"] = target[cutoff_pos]    # no-change benchmark

    pred["error"] = pred["predictions"] - pred["actual"]
    pred["abs_error"] = pred["error"].abs()
    pred["sq_error"] = pred["error"] ** 2
    pred["abs_error_naive"] = (pred["last_observed"] - pred["actual"]).abs()
    return pred.drop(columns="item_id")


################## Forecast evaluation ######################

def evaluate(results, transform):
    scored = results.dropna(subset=["actual"])
    if transform == "level":
        # a percentage of a growth rate close to zero is meaningless
        scored = scored.assign(ape=100 * scored["abs_error"] / scored["actual"].abs())

    def by_cutoff(value):
        table = scored.pivot_table(index="cutoff", columns="horizon", values=value)
        table.index = [quarter_label(d) for d in table.index]
        return table

    def by_year(value, how="mean"):
        return scored.assign(year=scored["cutoff"].dt.year).pivot_table(
            index="year", columns="horizon", values=value, aggfunc=how)

    def summary(frame):
        out = {"MAFE": frame["abs_error"].mean(),
               "RMSE": np.sqrt(frame["sq_error"].mean()),
               "Bias (mean error)": frame["error"].mean(),
               "MAFE / MAFE no-change": frame["abs_error"].mean() / frame["abs_error_naive"].mean(),
               "N": len(frame)}
        if "ape" in frame:
            out["MAPE"] = frame["ape"].mean()
        return pd.Series(out)

    per_cutoff = {"abs_error": by_cutoff("abs_error"), "error": by_cutoff("error")}
    per_year = {"MAFE": by_year("abs_error"), "RMSE": np.sqrt(by_year("sq_error")), "Bias": by_year("error")}
    if transform == "level":
        per_cutoff["ape"] = by_cutoff("ape")
        per_year["MAPE"] = by_year("ape")
    by_horizon = scored.groupby("horizon").apply(summary, include_groups=False)
    return per_cutoff, per_year, by_horizon


################## Plot #####################################

def plot_heatmap(table, title, colorbar_label, path, diverging=False, annotate=False):
    values = table.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if diverging:
        limit = np.percentile(np.abs(finite), ROBUST_PERCENTILE)
        vmin, vmax, cmap = -limit, limit, plt.get_cmap("RdBu_r").copy()
    else:
        vmin, vmax, cmap = 0, np.percentile(finite, ROBUST_PERCENTILE), plt.get_cmap("Blues").copy()
    cmap.set_bad("#e6e6e6")    # horizons beyond the last observed quarter

    n_rows = len(table)
    row_height = 0.11 if n_rows > 40 else 0.3
    fig, ax = plt.subplots(figsize=(10, max(row_height * n_rows, 4) + 1.5))
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(table.shape[1]), table.columns)
    step = 4 if n_rows > 40 else 1
    ax.set_yticks(range(0, n_rows, step), table.index[::step])
    ax.set_xlabel("Forecast horizon (quarters)")
    ax.set_ylabel("Forecast cut-off")
    ax.set_title(title)

    if annotate:
        decimals = 0 if np.nanmax(np.abs(finite)) >= 100 else 2
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                if np.isfinite(v):
                    dark = abs(v) > 0.6 * vmax if diverging else v > 0.6 * vmax
                    ax.text(j, i, f"{v:,.{decimals}f}", ha="center", va="center", fontsize=6.5,
                            color="white" if dark else "#1f1f1f")

    extend = "both" if diverging else "max"
    fig.colorbar(im, ax=ax, label=colorbar_label, extend=extend, shrink=0.6 if n_rows > 40 else 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_hair_chart(results, series, transform, path):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for _, hair in results.groupby("cutoff"):
        # each hair starts at the last observed value of its cut-off
        x = [hair["cutoff"].iloc[0]] + list(hair["target_date"])
        y = [hair["last_observed"].iloc[0]] + list(hair["predictions"])
        ax.plot(x, y, color="#2a6fdb", alpha=0.5, linewidth=1)
    ax.plot([], [], color="#2a6fdb", linewidth=1, label="Chronos2 median forecast, one per cut-off")

    start = pd.Timestamp(FIRST_CUTOFF) - pd.DateOffset(years=2)
    shown = series[series["DATE"] >= start]
    ax.plot(shown["DATE"], shown[TARGET_COLUMN], color="#1f1f1f", linewidth=2, label="Realised (BMPE)")

    # Keep the bulk readable when a few quarters (e.g. Covid) are extreme. The range comes
    # from the realised series only: forecasts cluster near the mean and would shrink it.
    everything = np.concatenate([shown[TARGET_COLUMN].to_numpy(), results["predictions"].to_numpy()])
    lo, hi = np.nanpercentile(shown[TARGET_COLUMN], [1, 99])
    if transform == "qoq" and (np.nanmin(everything) < lo or np.nanmax(everything) > hi):
        pad = 0.15 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.text(0.99, 0.98, "y-axis limited to the 1st-99th percentile range of the realised series", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="#595959")

    ax.set_title(f"Chronos2 zero-shot forecasts from each cut-off, {TARGET_COLUMN} ({unit(transform)})")
    ax.set_ylabel(f"{TARGET_COLUMN} ({unit(transform)})")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


################## Save #####################################

def save_excel(path, results, per_cutoff, per_year, by_horizon):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        by_horizon.to_excel(writer, sheet_name="by_horizon")
        for name, table in per_cutoff.items():
            table.to_excel(writer, sheet_name=f"{name}_by_cutoff")
        for name, table in per_year.items():
            table.to_excel(writer, sheet_name=f"{name}_by_year")
        results.to_excel(writer, sheet_name="forecasts", index=False)


def save_excel_unlocked(path, *tables):
    try:
        save_excel(path, *tables)
    except PermissionError:
        # Windows locks a workbook that is open in Excel: keep this run's results anyway
        locked = path
        path = path.replace(".xlsx", f"_{pd.Timestamp.now():%Y%m%d_%H%M%S}.xlsx")
        print(f"  {locked} is locked (open in Excel?), saving to a new file instead")
        save_excel(path, *tables)
    print(f"  Saved {path}")


################## Main #####################################

HEATMAPS_PER_CUTOFF = {
    "abs_error": ("Absolute error", False),
    "error": ("Error (forecast - realised)", True),
    "ape": ("Absolute percentage error, %", False),
}
HEATMAPS_PER_YEAR = {
    "MAFE": ("MAFE", False),
    "RMSE": ("RMSE", False),
    "Bias": ("Mean error (forecast - realised)", True),
    "MAPE": ("MAPE, %", False),
}


def run_exercise(pipeline, data, transform):
    print(f"\nTarget: {TARGET_COLUMN}, {unit(transform)}", flush=True)
    output_dir = os.path.join(OUTPUT_DIR, transform)
    os.makedirs(output_dir, exist_ok=True)

    series = data[["DATE", transform]].rename(columns={transform: TARGET_COLUMN}).dropna().reset_index(drop=True)
    results = run_forecasts(pipeline, series)
    per_cutoff, per_year, by_horizon = evaluate(results, transform)

    print("\n  Metrics by horizon, all cut-offs:")
    print(by_horizon.round(2).to_string())
    save_excel_unlocked(os.path.join(output_dir, "zero_shot_cutoff.xlsx"), results, per_cutoff, per_year, by_horizon)

    for name, table in per_cutoff.items():
        label, diverging = HEATMAPS_PER_CUTOFF[name]
        plot_heatmap(table, f"Chronos2 zero-shot, {unit(transform)}: {label.lower()} of each forecast",
                     f"{label} ({unit(transform)})" if name != "ape" else label,
                     os.path.join(output_dir, f"heatmap_{name}_by_cutoff.png"), diverging=diverging)
    for name, table in per_year.items():
        label, diverging = HEATMAPS_PER_YEAR[name]
        plot_heatmap(table, f"Chronos2 zero-shot, {unit(transform)}: {label} over the cut-offs of each year",
                     f"{label} ({unit(transform)})" if name != "MAPE" else label,
                     os.path.join(output_dir, f"heatmap_{name}_by_year.png"), diverging=diverging, annotate=True)
    plot_hair_chart(results, series, transform, os.path.join(output_dir, "hair_chart.png"))


def main(pipeline=None):
    if pipeline is None:
        pipeline = load_model(MODEL_PATH)

    print("Step 2 - Load data", flush=True)
    data = load_ground_truth()
    print(f"  {len(data)} quarters, {quarter_label(data['DATE'].iloc[0])} to {quarter_label(data['DATE'].iloc[-1])}",
          flush=True)

    print("Step 3 - Forecasts and evaluation", flush=True)
    for transform in TRANSFORMS:
        run_exercise(pipeline, data, transform)


if __name__ == "__main__":
    main()
