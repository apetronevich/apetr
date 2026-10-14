############################################################################################
#   Chronos2 zero-shot : la prévision à l'horizon h dépend-elle de PREDICTION_LENGTH ?
#                                   - Anna Petronevich
#                         -----
#    For the same contexts (same cut-offs, same context length), Chronos2 forecasts
#    TARGET_COLUMN zero-shot with each value of PREDICTION_LENGTHS. The forecast at horizon
#    h requested with prediction_length = L is compared with the forecast at the same h
#    from the reference run (the longest L). If the model's output does not depend on L,
#    every difference is zero.
#    The reference run is done twice: the gap between the two runs is the numerical
#    noise of the hardware (exactly zero on CPU, sometimes tiny on GPU), against which
#    the other gaps are judged.
#
#    Inputs:
#      - chemin vers stockage modèle MODEL_PATH
#      - BMPE_FILE : export excel du BMPE (colonne DATE + TARGET_COLUMN)
#      - PREDICTION_LENGTHS : horizons demandés ; inclure des valeurs de part et d'autre
#        de 16 (taille d'un bloc de sortie de Chronos2)
#      - CONTEXT_LENGTHS : longueurs de contexte testées (None = tout l'historique)
#    Outputs (OUTPUT_DIR/<level|qoq>) :
#      - prediction_length_invariance.xlsx : écarts par longueur demandée x horizon,
#        prévisions, MAFE par longueur demandée
#      - max_gap_heatmap.png : plus grand écart |prévision(L) - prévision(référence)|
#      - pairwise_heatmap.png : plus grand écart entre chaque paire de longueurs demandées,
#        sur les horizons communs ; des blocs de zéros montrent quelles longueurs donnent
#        exactement les mêmes prévisions
#      - example_cutoff.png : trajectoires d'une même coupure pour chaque L
#                         -----
#  A.Petronevich et AI
############################################################################################


############### Settings ##################################################################

MODEL_PATH = "C:\\TSFM"

BMPE_FILE = "Inputs/BMPE_Juin2026.xlsx"
BMPE_SHEET = 0                  # sheet name or index
DATE_COLUMN = "DATE"            # dates or quarters like "2000Q1"
TARGET_COLUMN = "P51_S11S12S14A_7CH"   # matched ignoring case
LAST_OBSERVED = "2026Q1"        # later quarters are BMPE projections: dropped
LAST_FINAL = "2022Q4"           # ground truth for the MAFE table only; None = up to LAST_OBSERVED

FIRST_CUTOFF = "2000Q1"
CONTEXT_LENGTHS = [20, 40, None]    # quarters; None = all history up to the cut-off
PREDICTION_LENGTHS = [1, 2, 3, 4, 6, 8, 12, 16, 17, 20, 24]
TRANSFORMS = ["level", "qoq"]
QUANTILE_LEVELS = [0.1, 0.5, 0.9]
EXAMPLE_CUTOFF = "2019Q4"       # cut-off drawn in example_cutoff.png

# Two forecasts count as identical when they differ by less than this share of the mean
# absolute reference forecast (float32 arithmetic is good to about 1e-7)
RELATIVE_TOLERANCE = 1e-6

OUTPUT_DIR = "Outputs/prediction_length_invariance"


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


def quarter(value):
    return pd.Period(value, freq="Q")


def quarter_label(date):
    return f"{date.year}Q{date.quarter}"


def load_ground_truth():
    raw = pd.read_excel(BMPE_FILE, sheet_name=BMPE_SHEET)
    data = pd.DataFrame({
        "DATE": to_dates(raw[find_column(raw, DATE_COLUMN)]),
        "level": pd.to_numeric(raw[find_column(raw, TARGET_COLUMN)], errors="coerce"),
    }).dropna().sort_values("DATE")
    if LAST_OBSERVED is not None:
        data = data[data["DATE"].dt.to_period("Q") <= quarter(LAST_OBSERVED)]
    data["qoq"] = 100 * data["level"].pct_change()
    return data.reset_index(drop=True)


def unit(transform):
    return "q-o-q growth, %" if transform == "qoq" else "level"


def context_name(context_length):
    return "all history" if context_length is None else f"{context_length} quarters"


################## Generate predictions ####################

def build_contexts(series, context_length):
    """One item per cut-off, holding the context_length quarters up to the cut-off.
    Cut-offs without enough history are skipped."""
    last = len(series) - 1
    cutoffs = series.index[(series["DATE"].dt.to_period("Q") >= quarter(FIRST_CUTOFF)) & (series.index < last)]
    if context_length is not None:
        cutoffs = cutoffs[cutoffs + 1 >= context_length]
    start = (lambda pos: 0) if context_length is None else (lambda pos: pos + 1 - context_length)
    return pd.concat(
        [series.iloc[start(pos):pos + 1].assign(item_id=series["DATE"].iloc[pos].strftime("%Y-%m-%d"))
         for pos in cutoffs],
        ignore_index=True,
    )


def forecast(pipeline, contexts, prediction_length):
    pred = pipeline.predict_df(
        contexts,
        future_df=None,
        prediction_length=prediction_length,
        quantile_levels=QUANTILE_LEVELS,
        id_column="item_id",
        timestamp_column="DATE",
        target=TARGET_COLUMN,
    )
    pred = pred.sort_values(["item_id", "DATE"]).reset_index(drop=True)
    pred["horizon"] = pred.groupby("item_id").cumcount() + 1
    pred["cutoff"] = pd.to_datetime(pred["item_id"])
    return pred


def run_all(pipeline, series):
    """Every context length x requested length, plus a second reference run."""
    reference = max(PREDICTION_LENGTHS)
    runs = []
    for context_length in CONTEXT_LENGTHS:
        contexts = build_contexts(series, context_length)
        n_items = contexts["item_id"].nunique()
        print(f"  context {context_name(context_length)}: {n_items} cut-offs", flush=True)
        for prediction_length, run in [(L, "first") for L in PREDICTION_LENGTHS] + [(reference, "rerun")]:
            pred = forecast(pipeline, contexts, prediction_length)
            pred["context_length"] = context_name(context_length)
            pred["prediction_length"] = prediction_length
            pred["run"] = run
            runs.append(pred)
    return pd.concat(runs, ignore_index=True), reference


################## Comparison ###############################

def compare(forecasts, reference, quantile_cols):
    """Gap between each run and the first reference run at the same context length,
    cut-off and horizon."""
    keys = ["context_length", "cutoff", "horizon"]
    values = ["predictions"] + quantile_cols
    ref = forecasts[(forecasts["prediction_length"] == reference) & (forecasts["run"] == "first")]
    others = forecasts[~((forecasts["prediction_length"] == reference) & (forecasts["run"] == "first"))]
    merged = others.merge(ref[keys + values], on=keys, suffixes=("", "_ref"))

    merged["gap_median"] = (merged["predictions"] - merged["predictions_ref"]).abs()
    merged["gap_any_quantile"] = np.max(
        [(merged[c] - merged[f"{c}_ref"]).abs() for c in quantile_cols] + [merged["gap_median"]], axis=0)
    merged["label"] = np.where(merged["run"] == "rerun", f"{reference} (rerun)",
                               merged["prediction_length"].astype(str))
    scale = ref.groupby("context_length")["predictions"].apply(lambda s: s.abs().mean())
    merged["tolerance"] = merged["context_length"].map(scale) * RELATIVE_TOLERANCE
    merged["identical"] = merged["gap_any_quantile"] <= merged["tolerance"]
    return merged


def gap_tables(merged, reference):
    """Largest gap over all cut-offs, by requested length (rows) and horizon (columns)."""
    order = [f"{reference} (rerun)"] + [str(L) for L in sorted(PREDICTION_LENGTHS) if L != reference]
    tables = {}
    for context_length, sub in merged.groupby("context_length", sort=False):
        table = sub.pivot_table(index="label", columns="horizon", values="gap_any_quantile", aggfunc="max")
        tables[context_length] = table.reindex([o for o in order if o in table.index])
    return tables


def pairwise_gaps(forecasts, quantile_cols):
    """Largest gap between every pair of requested lengths, over the horizons they share
    (1..min(L1, L2)), all cut-offs, median and quantiles."""
    first = forecasts[forecasts["run"] == "first"]
    lengths = sorted(PREDICTION_LENGTHS)
    tables = {}
    for context_length, sub in first.groupby("context_length", sort=False):
        gap = pd.DataFrame(0.0, index=lengths, columns=lengths)
        wide = {c: sub.pivot_table(index=["cutoff", "horizon"], columns="prediction_length", values=c)
                for c in ["predictions"] + quantile_cols}
        for a in lengths:
            for b in lengths:
                if a < b:
                    g = max(np.nanmax((w[a] - w[b]).abs().to_numpy()) for w in wide.values())
                    gap.loc[a, b] = gap.loc[b, a] = g
        gap.index.name = "requested length"
        tables[context_length] = gap
    return tables


def summary(merged):
    rows = []
    for (context_length, label), sub in merged.groupby(["context_length", "label"], sort=False):
        rows.append({"context length": context_length, "requested length": label,
                     "forecasts compared": len(sub),
                     "share identical": sub["identical"].mean(),
                     "max gap, median": sub["gap_median"].max(),
                     "max gap, any quantile": sub["gap_any_quantile"].max(),
                     "tolerance": sub["tolerance"].iloc[0]})
    return pd.DataFrame(rows)


def mafe_by_length(forecasts, series):
    """MAFE of each requested length at each horizon, final quarters only (first runs)."""
    position_of = {d: i for i, d in enumerate(series["DATE"])}
    first = forecasts[forecasts["run"] == "first"].copy()
    truth_pos = first["cutoff"].map(position_of).to_numpy() + first["horizon"].to_numpy()
    ok = truth_pos < len(series)
    if LAST_FINAL is not None:
        target_quarter = series["DATE"].dt.to_period("Q").reindex(truth_pos)
        ok &= (target_quarter <= quarter(LAST_FINAL)).to_numpy()
    actual = np.where(ok, series[TARGET_COLUMN].to_numpy()[np.minimum(truth_pos, len(series) - 1)], np.nan)
    first["abs_error"] = (first["predictions"] - actual).abs()
    return first.dropna(subset=["abs_error"]).pivot_table(
        index=["context_length", "prediction_length"], columns="horizon", values="abs_error")


################## Plot #####################################

def plot_gap_heatmaps(tables, transform, path):
    fig, axes = plt.subplots(len(tables), 1, figsize=(12, 3.2 * len(tables) + 1), squeeze=False)
    for ax, (context_length, table) in zip(axes[:, 0], tables.items()):
        values = table.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        vmax = finite.max() if finite.size and finite.max() > 0 else 1.0
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("#e6e6e6")    # horizon beyond the requested length
        ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=vmax, aspect="auto",
                  interpolation="nearest")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                if np.isfinite(v):
                    text = "0" if v == 0 else f"{v:.1e}"
                    ax.text(j, i, text, ha="center", va="center", fontsize=6,
                            color="white" if v > 0.6 * vmax else "#1f1f1f")
        ax.set_xticks(range(table.shape[1]), table.columns)
        ax.set_yticks(range(table.shape[0]), table.index)
        ax.set_ylabel("Requested length")
        ax.set_title(f"Context: {context_length}", fontsize=10)
    axes[-1, 0].set_xlabel("Forecast horizon (quarters)")
    fig.suptitle(f"Largest |forecast(L) - forecast(reference)| over all cut-offs, median and quantiles "
                 f"({unit(transform)})\n0 = identical; grey = horizon beyond the requested length")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pairwise(tables, transform, path):
    fig, axes = plt.subplots(1, len(tables), figsize=(5.2 * len(tables), 5), squeeze=False)
    for ax, (context_length, gap) in zip(axes[0], tables.items()):
        values = gap.to_numpy(dtype=float)
        vmax = values.max() if values.max() > 0 else 1.0
        ax.imshow(values, cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                ax.text(j, i, "0" if v == 0 else f"{v:.1g}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.6 * vmax else "#1f1f1f")
        ax.set_xticks(range(len(gap)), gap.columns)
        ax.set_yticks(range(len(gap)), gap.index)
        ax.set_xlabel("Requested length")
        ax.set_title(f"Context: {context_length}", fontsize=10)
    axes[0, 0].set_ylabel("Requested length")
    fig.suptitle(f"Largest gap between two requested lengths, on the horizons they share ({unit(transform)})\n"
                 f"0 = the two lengths give exactly the same forecasts")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_example(forecasts, series, transform, reference, path):
    cutoff = quarter(EXAMPLE_CUTOFF).start_time
    sub = forecasts[(forecasts["cutoff"].dt.to_period("Q") == quarter(EXAMPLE_CUTOFF)) & (forecasts["run"] == "first")]
    if sub.empty:
        print(f"  {EXAMPLE_CUTOFF} is not a cut-off of this run, example chart skipped")
        return
    contexts = list(dict.fromkeys(sub["context_length"]))
    fig, axes = plt.subplots(1, len(contexts), figsize=(5 * len(contexts), 4.5), sharey=True, squeeze=False)
    history = series[(series["DATE"] <= cutoff + pd.DateOffset(years=7)) & (series["DATE"] >= cutoff - pd.DateOffset(years=5))]
    lengths = sorted(PREDICTION_LENGTHS)
    colors = plt.cm.Blues(np.linspace(0.35, 1.0, len(lengths)))
    for ax, context_length in zip(axes[0], contexts):
        ax.plot(history["DATE"], history[TARGET_COLUMN], color="#1f1f1f", linewidth=2, label="BMPE")
        for color, L in zip(colors, lengths):
            path_ = sub[(sub["context_length"] == context_length) & (sub["prediction_length"] == L)]
            ax.plot(path_["horizon"].map(lambda h: cutoff + pd.DateOffset(months=3 * h)), path_["predictions"],
                    color=color, linewidth=1.2, marker="o", markersize=2.5, label=f"L = {L}")
        ax.set_title(f"Context: {context_length}", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel(f"{TARGET_COLUMN} ({unit(transform)})")
    axes[0, -1].legend(fontsize=7, loc="best")
    fig.suptitle(f"Median forecasts from {EXAMPLE_CUTOFF} for each requested length L "
                 f"(overlapping lines = identical forecasts)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


################## Save #####################################

def save_excel(path, summary_table, tables, pairwise, mafe, forecasts):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary_table.to_excel(writer, sheet_name="summary", index=False)
        for i, (context_length, table) in enumerate(pairwise.items()):
            table.to_excel(writer, sheet_name=f"pairwise_ctx{i + 1}")
            writer.sheets[f"pairwise_ctx{i + 1}"].cell(row=1, column=1, value=f"context: {context_length}")
        for i, (context_length, table) in enumerate(tables.items()):
            table.to_excel(writer, sheet_name=f"max_gap_ctx{i + 1}")
            writer.sheets[f"max_gap_ctx{i + 1}"].cell(row=1, column=1, value=f"context: {context_length}")
        mafe.to_excel(writer, sheet_name="MAFE_by_length")
        forecasts.drop(columns="item_id").to_excel(writer, sheet_name="forecasts", index=False)


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

def run_exercise(pipeline, data, transform):
    print(f"\nTarget: {TARGET_COLUMN}, {unit(transform)}", flush=True)
    output_dir = os.path.join(OUTPUT_DIR, transform)
    os.makedirs(output_dir, exist_ok=True)

    series = data[["DATE", transform]].rename(columns={transform: TARGET_COLUMN}).dropna().reset_index(drop=True)
    forecasts, reference = run_all(pipeline, series)
    quantile_cols = [c for c in forecasts.columns if _is_number(c)]
    merged = compare(forecasts, reference, quantile_cols)
    tables = gap_tables(merged, reference)
    summary_table = summary(merged)
    mafe = mafe_by_length(forecasts, series)
    pairwise = pairwise_gaps(forecasts, quantile_cols)

    print(f"\n  Reference: prediction_length = {reference}. Share of forecasts identical to it "
          f"(gap <= {RELATIVE_TOLERANCE:g} x mean forecast), and largest gap:")
    print(summary_table.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    not_identical = summary_table[summary_table["share identical"] < 1]
    if not_identical.empty:
        print("\n  -> Every forecast is identical whatever the requested length.")
    else:
        print("\n  -> The requested length changes the forecasts (compared with L = %d):" % reference)
        for context_length, rows in not_identical.groupby("context length", sort=False):
            print(f"     context {context_length}: L = {', '.join(rows['requested length'].astype(str))}")

    print("\n  Requested lengths giving exactly the same forecasts on their shared horizons:")
    for context_length, gap in pairwise.items():
        tolerance = merged.loc[merged["context_length"] == context_length, "tolerance"].iloc[0]
        groups, seen = [], set()
        for a in gap.index:
            if a not in seen:
                group = [b for b in gap.index if gap.loc[a, b] <= tolerance]
                seen.update(group)
                groups.append(group)
        print(f"     context {context_length}: " + " | ".join("{" + ", ".join(map(str, g)) + "}" for g in groups))

    save_excel_unlocked(os.path.join(output_dir, "prediction_length_invariance.xlsx"),
                        summary_table, tables, pairwise, mafe, forecasts)
    plot_gap_heatmaps(tables, transform, os.path.join(output_dir, "max_gap_heatmap.png"))
    plot_pairwise(pairwise, transform, os.path.join(output_dir, "pairwise_heatmap.png"))
    plot_example(forecasts, series, transform, reference, os.path.join(output_dir, "example_cutoff.png"))


def _is_number(name):
    try:
        float(name)
        return True
    except (TypeError, ValueError):
        return False


def main(pipeline=None):
    if pipeline is None:
        pipeline = load_model(MODEL_PATH)

    print("Step 2 - Load data", flush=True)
    data = load_ground_truth()
    print(f"  {len(data)} quarters, {quarter_label(data['DATE'].iloc[0])} to {quarter_label(data['DATE'].iloc[-1])}",
          flush=True)

    print("Step 3 - Forecasts with each requested length", flush=True)
    for transform in TRANSFORMS:
        run_exercise(pipeline, data, transform)


if __name__ == "__main__":
    main()
