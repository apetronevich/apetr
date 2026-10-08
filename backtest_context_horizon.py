############################################################################################
#   Chronos2 : impact de la longueur du contexte et de l'horizon - Anna Petronevich
#                         -----
#    Pseudo out-of-sample rolling-origin backtest. For every forecast origin since
#    FIRST_ORIGIN and every context length in CONTEXT_LENGTHS, Chronos2 forecasts
#    PREDICTION_LENGTH periods ahead using only the last L observations up to the
#    origin. Forecasts are compared with the realised values of TARGET_COLUMN.
#
#    Inputs:
#      - chemin vers stockage modèle MODEL_PATH
#      - INPUT_FILE : fichier excel de sortie eviews (colonne DATE + TARGET_COLUMN)
#      - TARGET_COLUMN : variable à prévoir (sert aussi de réalisé / ground truth)
#      - CONTEXT_LENGTHS : longueurs de contexte testées (None = tout l'historique)
#      - PREDICTION_LENGTH : horizon maximal
#    Outputs (OUTPUT_DIR):
#      - backtest_context_horizon.xlsx : prévisions + grilles d'erreur contexte x horizon
#      - mafe_heatmap.png, mafe_by_horizon.png
#      - fan_chart_context_<L>.png : forecasts at FAN_HORIZONS against the observed series
#                         -----
#  A.Petronevich et AI
############################################################################################


############### Settings ##################################################################

MODEL_PATH = "C:\\TSFM"
INPUT_FILE = "Inputs/RawDataInvMPEMars2026_wocovariates.xlsx"
TARGET_COLUMN = "P51_S11S12S14A_7CH"

# False: target only (zero-shot). True: every other column of INPUT_FILE is used as a
# past covariate (no future covariates, they would not be known at the origin).
USE_COVARIATES = False

FIRST_ORIGIN = "2000-01-01"     # last observed date of the first forecast
PREDICTION_LENGTH = 11          # horizons 1..PREDICTION_LENGTH
CONTEXT_LENGTHS = [8, 12, 16, 20, 28, 40, 60, None]   # in periods, None = all history
# Symmetric pairs give the fan chart bands (90%, 80%, 50%); the widest pair is used for coverage
QUANTILE_LEVELS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
FAN_HORIZONS = [1, 4, 8]       # one panel per horizon in the fan charts

OUTPUT_DIR = "Outputs/backtest_context_horizon"


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

def load_data(path):
    df = pd.read_excel(path)
    df["DATE"] = pd.to_datetime(df["DATE"])
    if not USE_COVARIATES:
        df = df[["DATE", TARGET_COLUMN]]
    # eviews exports can carry empty rows after the last observation
    df = df.dropna(subset=[TARGET_COLUMN])
    return df.sort_values("DATE").reset_index(drop=True)


def context_label(context_length):
    return "all" if context_length is None else str(context_length)


################## Generate predictions ####################

def build_contexts(df, origin_positions, context_length):
    """One series per origin (item_id = origin date), holding the last
    context_length observations up to and including the origin."""
    frames = []
    for pos in origin_positions:
        start = 0 if context_length is None else pos + 1 - context_length
        if start < 0:
            continue    # not enough history at this origin for this context length
        window = df.iloc[start:pos + 1].copy()
        window["item_id"] = df["DATE"].iloc[pos].strftime("%Y-%m-%d")
        frames.append(window)
    return pd.concat(frames, ignore_index=True) if frames else None


def run_backtest(pipeline, df):
    origin_positions = df.index[df["DATE"] >= pd.Timestamp(FIRST_ORIGIN)][:-1]
    position_of = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(df["DATE"])}
    target = df[TARGET_COLUMN].to_numpy()

    results = []
    for context_length in CONTEXT_LENGTHS:
        label = context_label(context_length)
        contexts = build_contexts(df, origin_positions, context_length)
        if contexts is None:
            print(f"  context {label}: no origin with enough history, skipped", flush=True)
            continue
        print(f"  context {label}: {contexts['item_id'].nunique()} origins", flush=True)

        # All origins go in a single call, one item_id each
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
        pred["context_length"] = label
        pred["horizon"] = pred.groupby("item_id").cumcount() + 1

        # Ground truth matched by position (origin + h), not by forecast date,
        # so it does not depend on how the quarterly dates are stamped
        origin_pos = pred["item_id"].map(position_of).to_numpy()
        truth_pos = origin_pos + pred["horizon"].to_numpy()
        in_sample = truth_pos < len(target)
        pred["origin"] = pd.to_datetime(pred["item_id"])
        pred["target_date"] = df["DATE"].reindex(truth_pos).to_numpy()
        pred["actual"] = np.where(in_sample, target[np.minimum(truth_pos, len(target) - 1)], np.nan)
        pred["naive"] = target[origin_pos]    # random walk: last observed value
        results.append(pred)

    results = pd.concat(results, ignore_index=True).dropna(subset=["actual"])
    results["error"] = results["predictions"] - results["actual"]
    results["abs_error"] = results["error"].abs()
    results["abs_pct_error"] = 100 * results["abs_error"] / results["actual"].abs()
    results["abs_error_naive"] = (results["naive"] - results["actual"]).abs()

    quantile_cols = [c for c in results.columns if _is_number(c)]
    if len(quantile_cols) >= 2:
        lo = min(quantile_cols, key=float)
        hi = max(quantile_cols, key=float)
        results["inside_interval"] = (results["actual"] >= results[lo]) & (results["actual"] <= results[hi])
    return results


def _is_number(name):
    try:
        float(name)
        return True
    except (TypeError, ValueError):
        return False


################## Forecast evaluation ######################

def grid(results, value, how="mean"):
    order = [context_label(c) for c in CONTEXT_LENGTHS if context_label(c) in set(results["context_length"])]
    table = results.pivot_table(index="context_length", columns="horizon", values=value, aggfunc=how)
    return table.reindex(order)


def evaluate(results):
    # Long contexts only exist for late origins, so the full-sample grids compare
    # different periods. The common sample keeps origins where every context exists.
    n_contexts = results.groupby("origin")["context_length"].nunique()
    common_origins = n_contexts.index[n_contexts == results["context_length"].nunique()]
    common = results[results["origin"].isin(common_origins)]

    squared = results.assign(sq=results["error"] ** 2)
    common_sq = common.assign(sq=common["error"] ** 2)

    tables = {
        "MAFE": grid(results, "abs_error"),
        "RMSE": np.sqrt(grid(squared, "sq")),
        "MAPE": grid(results, "abs_pct_error"),
        "MAFE_vs_RW": grid(results, "abs_error") / grid(results, "abs_error_naive"),
        "MAFE_common": grid(common, "abs_error"),
        "RMSE_common": np.sqrt(grid(common_sq, "sq")),
        "MAFE_vs_RW_common": grid(common, "abs_error") / grid(common, "abs_error_naive"),
        "N_forecasts": grid(results, "abs_error", how="count"),
    }
    if "inside_interval" in results:
        tables["Coverage_widest"] = grid(results.assign(cov=results["inside_interval"].astype(float)), "cov")
    return tables, common_origins


################## Plot #####################################

def plot_heatmap(table, title, path):
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(table) + 2))
    im = ax.imshow(table.to_numpy(dtype=float), cmap="Blues", aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Context length")
    ax.set_title(title)

    values = table.to_numpy(dtype=float)
    threshold = np.nanmin(values) + 0.6 * (np.nanmax(values) - np.nanmin(values))
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.isnan(values[i, j]):
                ax.text(j, i, f"{values[i, j]:,.0f}", ha="center", va="center", fontsize=8,
                        color="white" if values[i, j] > threshold else "#1f1f1f")
    fig.colorbar(im, ax=ax, label="MAFE")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_by_horizon(table, title, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    # Context length is ordinal: one hue, light (short) to dark (long)
    colors = plt.cm.Blues(np.linspace(0.35, 1.0, len(table)))
    for color, (label, row) in zip(colors, table.iterrows()):
        ax.plot(row.index, row.values, marker="o", markersize=4, linewidth=2, color=color, label=label)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("MAFE")
    ax.set_title(title)
    ax.set_xticks(table.columns)
    ax.grid(alpha=0.3)
    ax.legend(title="Context length")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fan_chart(results, df, context, path):
    """For each horizon h, the forecast made h periods before each date, plotted at
    that date against the observed series."""
    sub = results[results["context_length"] == context]
    quantiles = {float(c): c for c in sub.columns if _is_number(c)}
    # Symmetric quantile pairs, widest first so narrower bands are painted on top
    bands = sorted((q, 1 - q) for q in quantiles if q < 0.5 and any(abs(1 - q - p) < 1e-9 for p in quantiles))
    band_colors = plt.cm.Blues(np.linspace(0.25, 0.55, max(len(bands), 1)))
    horizons = [h for h in FAN_HORIZONS if h <= PREDICTION_LENGTH]
    observed = df[df["DATE"] >= sub["target_date"].min() - pd.DateOffset(years=1)]

    fig, axes = plt.subplots(len(horizons), 1, figsize=(11, 3.4 * len(horizons)), sharex=True, squeeze=False)
    for ax, h in zip(axes[:, 0], horizons):
        s = sub[sub["horizon"] == h].sort_values("target_date")
        for color, (lo, hi) in zip(band_colors, bands):
            hi_col = next(c for q, c in quantiles.items() if abs(q - hi) < 1e-9)
            ax.fill_between(s["target_date"], s[quantiles[lo]], s[hi_col], color=color, linewidth=0,
                            label=f"{round(100 * (hi - lo))}% interval")
        ax.plot(s["target_date"], s["predictions"], color="#0b3d91", linewidth=1.5, label="Median forecast")
        ax.plot(observed["DATE"], observed[TARGET_COLUMN], color="#1f1f1f", linewidth=2, label="Observed")
        ax.set_title(f"{h}-step-ahead forecasts")
        ax.set_ylabel(TARGET_COLUMN)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Chronos2 forecasts vs observed, context length {context}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


################## Main #####################################

def main(pipeline=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if pipeline is None:
        pipeline = load_model(MODEL_PATH)

    print("Step 2 - Load data", flush=True)
    df = load_data(INPUT_FILE)
    print(f"  {len(df)} observations, {df['DATE'].min():%Y-%m-%d} to {df['DATE'].max():%Y-%m-%d}", flush=True)

    print("Step 3 - Rolling forecasts", flush=True)
    results = run_backtest(pipeline, df)

    print("Step 4 - Evaluation", flush=True)
    tables, common_origins = evaluate(results)
    print("\nMAFE, full sample (context length x horizon):")
    print(tables["MAFE"].round(0).to_string())
    if len(common_origins):
        print(f"\nMAFE, common sample ({common_origins.min():%Y-%m-%d} to {common_origins.max():%Y-%m-%d} origins):")
        print(tables["MAFE_common"].round(0).to_string())

    output_file = os.path.join(OUTPUT_DIR, "backtest_context_horizon.xlsx")
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name)
        results.drop(columns="item_id").to_excel(writer, sheet_name="forecasts", index=False)
    print(f"\nSaved {output_file}")

    plot_heatmap(tables["MAFE"], "Chronos2 MAFE by context length and horizon (all origins)",
                 os.path.join(OUTPUT_DIR, "mafe_heatmap.png"))
    if len(common_origins):
        plot_heatmap(tables["MAFE_common"], "Chronos2 MAFE by context length and horizon (common sample)",
                     os.path.join(OUTPUT_DIR, "mafe_heatmap_common.png"))
        plot_by_horizon(tables["MAFE_common"], "Chronos2 MAFE by horizon (common sample)",
                        os.path.join(OUTPUT_DIR, "mafe_by_horizon.png"))
    for context in tables["MAFE"].index:
        plot_fan_chart(results, df, context, os.path.join(OUTPUT_DIR, f"fan_chart_context_{context}.png"))
    return results, tables


if __name__ == "__main__":
    main()
