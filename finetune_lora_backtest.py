############################################################################################
#   Chronos2 : fine-tuning LoRA contre zero-shot, en pseudo temps réel - Anna Petronevich
#                         -----
#    For every cut-off from FIRST_CUTOFF on, TARGET_COLUMN is forecast 1..PREDICTION_LENGTH
#    quarters ahead twice from the same context (all history up to the cut-off):
#      - zero-shot, with the pretrained Chronos2;
#      - with a LoRA adapter fine-tuned ONLY on data up to the cut-off. Chronos2.fit()
#        samples training windows anywhere in the series it receives, so giving it data
#        after the cut-off would leak the future into the forecast. The adapter is
#        re-trained every REFIT_EVERY quarters (on the data up to that refit date) and
#        used for the cut-offs until the next refit.
#    Both are scored on final BMPE quarters only (up to LAST_FINAL).
#
#    Inputs:
#      - chemin vers stockage modèle MODEL_PATH ; le package peft doit être installé
#      - BMPE_FILE : export excel du BMPE (colonne DATE + TARGET_COLUMN [+ autres séries])
#      - TRAIN_SERIES : "all" = toutes les séries du fichier servent au fine-tuning (panel),
#        "target" = la cible seule
#    Outputs (OUTPUT_DIR/<level|qoq>) :
#      - lora_vs_zeroshot.xlsx : métriques par horizon, ratio LoRA / zero-shot par année,
#        historique d'entraînement, prévisions
#        + test de Diebold-Mariano LoRA vs zero-shot par horizon (feuille DM_test)
#      - mafe_by_horizon.png, relative_mafe_by_year.png
#    Outputs (OUTPUT_DIR) : si les deux transformations sont lancées,
#      - level_comparison.xlsx / .png : niveaux prévus directement, ou reconstruits à partir
#        des taux de croissance prévus, zero-shot et LoRA, sur les mêmes coupures
#      - checkpoints/ : un adaptateur LoRA par date de ré-estimation
#                         -----
#  A.Petronevich et AI
############################################################################################


############### Settings ##################################################################

MODEL_PATH = "C:\\TSFM"

BMPE_FILE = "Inputs/BMPE_Juin2026.xlsx"
BMPE_SHEET = 0                  # sheet name or index
DATE_COLUMN = "DATE"            # dates or quarters like "2000Q1"
TARGET_COLUMN = "P51_S11S12S14A_7CH"   # matched ignoring case
LAST_OBSERVED = "2026Q1"        # later quarters are BMPE projections: dropped everywhere
LAST_FINAL = "2022Q4"           # later quarters are still revised: not used as ground truth

FIRST_CUTOFF = "2000Q1"
PREDICTION_LENGTH = 12
TRANSFORMS = ["level", "qoq"]   # the training series get the same transformation as the target
QUANTILE_LEVELS = [0.1, 0.5, 0.9]

# --- Fine-tuning -----------------------------------------------------------------------
# "all": every numeric column of BMPE_FILE is a training series (a panel of related macro
# series; fine-tuning on one short series tends to overfit). "target": the target only.
TRAIN_SERIES = "all"
REFIT_EVERY = 4                 # quarters between two re-trainings of the adapter
LORA_CONFIG = {                 # Chronos2's default LoRA configuration
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["self_attention.q", "self_attention.v", "self_attention.k",
                       "self_attention.o", "output_patch_embedding.output_layer"],
}
LEARNING_RATE = 1e-5            # the value Chronos2 recommends for LoRA
NUM_STEPS = 200                 # few steps: little data, overfitting comes fast
BATCH_SIZE = 64                 # training windows per step
# True: hold out the last PREDICTION_LENGTH quarters before each refit date, train on the
# rest and keep the step with the lowest loss on the held-out quarters (early stopping).
# The adapter then never sees those most recent quarters.
USE_VALIDATION = False
SEED = 0

OUTPUT_DIR = "Outputs/finetune_lora"


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
    try:
        import peft  # noqa: F401
    except ImportError:
        # without peft, Chronos2.fit() silently falls back to full fine-tuning
        raise ImportError("Le package peft est nécessaire pour LoRA : pip install peft")

    return Chronos2Pipeline.from_pretrained(
        model_path,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        local_files_only=True
    )


def loss_recorder():
    """Trainer callback keeping the training (and validation) losses of a fit."""
    from transformers import TrainerCallback

    class LossRecorder(TrainerCallback):
        def __init__(self):
            self.history = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                self.history.append({"step": state.global_step, **logs})

    return LossRecorder()


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


def unit(transform):
    return "q-o-q growth, %" if transform == "qoq" else "level"


def load_bmpe():
    """All numeric series of the BMPE file, target first, one row per quarter."""
    raw = pd.read_excel(BMPE_FILE, sheet_name=BMPE_SHEET)
    date_col, target_col = find_column(raw, DATE_COLUMN), find_column(raw, TARGET_COLUMN)
    data = raw.drop(columns=date_col).apply(pd.to_numeric, errors="coerce")
    data = data[[target_col] + [c for c in data.columns if c != target_col]].rename(columns={target_col: TARGET_COLUMN})
    data.insert(0, "DATE", to_dates(raw[date_col]))
    data = data.dropna(subset=["DATE", TARGET_COLUMN]).sort_values("DATE")
    if LAST_OBSERVED is not None:
        data = data[data["DATE"].dt.to_period("Q") <= quarter(LAST_OBSERVED)]
    keep = [TARGET_COLUMN] if TRAIN_SERIES == "target" else [c for c in data.columns if c != "DATE"]
    return data[["DATE"] + [c for c in keep if data[c].notna().any()]].reset_index(drop=True)


def transform_panel(data, transform):
    if transform == "level":
        return data
    out = data[["DATE"]].copy()
    for c in data.columns[1:]:
        s = data[c]
        if (s.dropna() <= 0).any():
            # a growth rate needs a positive level (balances, rates...): left out of training
            if c == TARGET_COLUMN:
                raise ValueError(f"{TARGET_COLUMN} has non-positive values: no q-o-q growth rate")
            continue
        out[c] = 100 * s.pct_change()
    return out


################## Fine-tuning ##############################

def training_arrays(panel, last_pos):
    """Every series cut at the refit date (row last_pos included): nothing after it."""
    min_length = 2 * PREDICTION_LENGTH + (PREDICTION_LENGTH if USE_VALIDATION else 0)
    arrays = []
    for c in panel.columns[1:]:
        s = panel[c].iloc[:last_pos + 1].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        # the unbroken stretch ending at the refit date: no jump over missing quarters
        missing = np.flatnonzero(np.isnan(s))
        s = s[missing[-1] + 1:] if len(missing) else s
        if len(s) >= min_length:
            arrays.append(s)
    return arrays


def fine_tune(pipeline, panel, last_pos, transform, refit_label):
    arrays = training_arrays(panel, last_pos)
    if not arrays:
        return None, [], 0
    np.random.seed(SEED + last_pos)    # the training windows are drawn with numpy
    recorder = loss_recorder()
    kwargs = dict(
        prediction_length=PREDICTION_LENGTH,
        finetune_mode="lora",
        lora_config=dict(LORA_CONFIG),
        learning_rate=LEARNING_RATE,
        num_steps=NUM_STEPS,
        batch_size=BATCH_SIZE,
        output_dir=os.path.join(OUTPUT_DIR, transform, "checkpoints", refit_label),
        callbacks=[recorder],
        remove_printer_callback=True,
        seed=SEED,
        logging_steps=max(NUM_STEPS // 10, 1),
    )
    if USE_VALIDATION:
        every = max(NUM_STEPS // 10, 1)
        kwargs.update(inputs=[a[:-PREDICTION_LENGTH] for a in arrays], validation_inputs=arrays,
                      eval_steps=every, save_steps=every)
    else:
        kwargs.update(inputs=arrays)
    return pipeline.fit(**kwargs), recorder.history, len(arrays)


################## Generate predictions ####################

def forecast(pipeline, series, positions):
    contexts = pd.concat(
        [series.iloc[:pos + 1].assign(item_id=series["DATE"].iloc[pos].strftime("%Y-%m-%d")) for pos in positions],
        ignore_index=True,
    )
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
    pred["cutoff"] = pd.to_datetime(pred["item_id"])
    return pred[["cutoff", "horizon", "predictions"]]


def run(pipeline, panel, transform):
    series = panel[["DATE", TARGET_COLUMN]].dropna().reset_index(drop=True)
    panel = panel.loc[panel["DATE"].isin(series["DATE"])].reset_index(drop=True)    # same rows
    last = len(series) - 1
    cutoffs = list(series.index[(series["DATE"].dt.to_period("Q") >= quarter(FIRST_CUTOFF)) & (series.index < last)])
    refits = cutoffs[::REFIT_EVERY]
    print(f"  {len(cutoffs)} cut-offs {quarter_label(series['DATE'].iloc[cutoffs[0]])} to "
          f"{quarter_label(series['DATE'].iloc[cutoffs[-1]])}, {len(refits)} fine-tunings", flush=True)

    zero_shot = forecast(pipeline, series, cutoffs).rename(columns={"predictions": "zero_shot"})

    lora, training = [], []
    for k, refit_pos in enumerate(refits):
        block = [c for c in cutoffs if refit_pos <= c < (refits[k + 1] if k + 1 < len(refits) else last)]
        label = quarter_label(series["DATE"].iloc[refit_pos])
        finetuned, history, n_series = fine_tune(pipeline, panel, refit_pos, transform, label)
        if finetuned is None:
            print(f"  {label}: not enough data to fine-tune, cut-offs left out", flush=True)
            continue
        losses = [h["loss"] for h in history if "loss" in h]
        print(f"  {label}: LoRA on {n_series} series, loss {losses[0]:.3g} -> {losses[-1]:.3g}"
              if losses else f"  {label}: LoRA on {n_series} series", flush=True)
        training += [{"refit": label, "series": n_series, **h} for h in history]
        pred = forecast(finetuned, series, block).rename(columns={"predictions": "lora"})
        lora.append(pred.assign(refit=label))

    results = zero_shot.merge(pd.concat(lora), on=["cutoff", "horizon"], how="inner")

    # Ground truth by position (cut-off + h), final quarters only
    position_of = {d: i for i, d in enumerate(series["DATE"])}
    truth_pos = results["cutoff"].map(position_of).to_numpy() + results["horizon"].to_numpy()
    target_quarter = series["DATE"].dt.to_period("Q").reindex(truth_pos)
    ok = truth_pos <= last
    if LAST_FINAL is not None:
        ok &= (target_quarter <= quarter(LAST_FINAL)).to_numpy()
    values = series[TARGET_COLUMN].to_numpy()
    results["target_date"] = series["DATE"].reindex(truth_pos).to_numpy()
    results["actual"] = np.where(ok, values[np.minimum(truth_pos, last)], np.nan)
    return results, pd.DataFrame(training)


################## Forecast evaluation ######################

def evaluate(results):
    scored = results.dropna(subset=["actual"]).copy()
    for model in ["zero_shot", "lora"]:
        scored[f"ae_{model}"] = (scored[model] - scored["actual"]).abs()
        scored[f"se_{model}"] = (scored[model] - scored["actual"]) ** 2

    def summary(g):
        return pd.Series({
            "MAFE zero-shot": g["ae_zero_shot"].mean(),
            "MAFE LoRA": g["ae_lora"].mean(),
            "MAFE LoRA / zero-shot": g["ae_lora"].mean() / g["ae_zero_shot"].mean(),
            "RMSE zero-shot": np.sqrt(g["se_zero_shot"].mean()),
            "RMSE LoRA": np.sqrt(g["se_lora"].mean()),
            "RMSE LoRA / zero-shot": np.sqrt(g["se_lora"].mean() / g["se_zero_shot"].mean()),
            "share of forecasts where LoRA is closer": (g["ae_lora"] < g["ae_zero_shot"]).mean(),
            "N": len(g),
        })

    by_horizon = scored.groupby("horizon").apply(summary, include_groups=False)
    by_year = scored.assign(year=scored["cutoff"].dt.year).groupby(["year", "horizon"])[["ae_lora", "ae_zero_shot"]].mean()
    relative_by_year = (by_year["ae_lora"] / by_year["ae_zero_shot"]).unstack("horizon")
    relative_by_year.index = relative_by_year.index.astype(str)
    relative_by_year.loc["All years"] = by_horizon["MAFE LoRA / zero-shot"]
    return by_horizon, relative_by_year


def diebold_mariano(loss_a, loss_b, horizon):
    """Diebold-Mariano test of equal accuracy on loss_a - loss_b, ordered by cut-off.
    h-step forecasts made in consecutive quarters overlap, so their errors are correlated
    up to lag h-1: Newey-West variance with h-1 lags (Bartlett weights) and the
    Harvey-Leybourne-Newbold small-sample correction. A negative statistic means model a
    is more accurate. Returns (statistic, two-sided p-value).
    With about 100 cut-offs the test rejects too often beyond h = 1: in simulations of
    equally accurate overlapping forecasts, about 10% rejections at the 5% level for
    h = 4 to 12 (4% for h = 1). Read p < 0.01 as clear evidence, 0.01-0.10 as weak."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    dc = d - d.mean()
    variance = dc @ dc / n
    for k in range(1, horizon):
        variance += 2 * (1 - k / horizon) * (dc[k:] @ dc[:-k]) / n
    if variance <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(variance / n)
    stat *= np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    try:
        from scipy.stats import t
        p_value = 2 * t.sf(abs(stat), df=n - 1)
    except ImportError:    # normal approximation, close to Student's t for n around 100
        from math import erf, sqrt
        p_value = 2 * (1 - 0.5 * (1 + erf(abs(stat) / sqrt(2))))
    return stat, p_value


def dm_table(scored, model_a, model_b, actual):
    """DM tests of model_a against model_b by horizon, on absolute and squared errors."""
    rows = {}
    for h, g in scored.sort_values("cutoff").groupby("horizon"):
        e_a, e_b = g[model_a] - g[actual], g[model_b] - g[actual]
        dm_abs, p_abs = diebold_mariano(e_a.abs(), e_b.abs(), h)
        dm_sq, p_sq = diebold_mariano(e_a ** 2, e_b ** 2, h)
        rows[h] = {"DM stat, absolute errors": dm_abs, "p-value, absolute errors": p_abs,
                   "DM stat, squared errors": dm_sq, "p-value, squared errors": p_sq, "N": len(g)}
    table = pd.DataFrame(rows).T
    table.index.name = "horizon"
    return table


def implied_levels(results, levels):
    """Level paths implied by the growth forecasts: level at the cut-off compounded by the
    forecast q-o-q growth rates up to each horizon. Ground truth: the realised level,
    final quarters only, as for the other scores."""
    level_at = levels.set_index("DATE")[TARGET_COLUMN]
    out = results.sort_values(["cutoff", "horizon"]).copy()
    base = out["cutoff"].map(level_at).to_numpy()
    for model in ["zero_shot", "lora"]:
        growth = 1 + out[model] / 100
        out[model] = base * growth.groupby(out["cutoff"]).cumprod().to_numpy()
    out["actual"] = np.where(out["actual"].notna(), out["target_date"].map(level_at), np.nan)
    return out


def compare_levels(level_results, implied):
    """MAFE / RMSE of the four ways of forecasting the level, on the cut-offs and horizons
    where all four are scored, and DM tests of each against direct zero-shot."""
    keys = ["cutoff", "horizon"]
    merged = level_results[keys + ["zero_shot", "lora", "actual"]].merge(
        implied[keys + ["zero_shot", "lora"]].rename(columns={"zero_shot": "growth_zero_shot", "lora": "growth_lora"}),
        on=keys).dropna(subset=["actual"])
    models = {"zero_shot": "Level, zero-shot", "lora": "Level, LoRA",
              "growth_zero_shot": "From growth, zero-shot", "growth_lora": "From growth, LoRA"}
    mafe = pd.DataFrame({label: (merged[m] - merged["actual"]).abs().groupby(merged["horizon"]).mean()
                         for m, label in models.items()})
    rmse = pd.DataFrame({label: np.sqrt(((merged[m] - merged["actual"]) ** 2).groupby(merged["horizon"]).mean())
                         for m, label in models.items()})
    dm = {label: dm_table(merged, m, "zero_shot", "actual")
          for m, label in models.items() if m != "zero_shot"}
    return mafe, rmse, dm


################## Plot #####################################

def plot_by_horizon(by_horizon, transform, path):
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, height_ratios=[2, 1])
    ax.plot(by_horizon.index, by_horizon["MAFE zero-shot"], color="#8c8c8c", marker="o", linewidth=2,
            label="Zero-shot")
    ax.plot(by_horizon.index, by_horizon["MAFE LoRA"], color="#0b3d91", marker="o", linewidth=2,
            label="LoRA fine-tuned")
    ax.set_ylabel(f"MAFE ({unit(transform)})")
    ax.set_title(f"Chronos2, {TARGET_COLUMN} ({unit(transform)}): LoRA fine-tuning vs zero-shot")
    ax.grid(alpha=0.3)
    ax.legend()
    ax_ratio.axhline(1, color="#8c8c8c", linewidth=1, linestyle="--")
    ax_ratio.plot(by_horizon.index, by_horizon["MAFE LoRA / zero-shot"], color="#0b3d91", marker="o", linewidth=2)
    ax_ratio.set_ylabel("MAFE LoRA / zero-shot\n(< 1: LoRA better)")
    ax_ratio.set_xlabel("Forecast horizon (quarters)")
    ax_ratio.set_xticks(by_horizon.index)
    ax_ratio.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_relative_by_year(table, transform, path):
    values = table.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    spread = max(np.percentile(np.abs(np.log(finite)), 95), 0.05) if finite.size else 0.5
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e6e6e6")
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(table) + 2))
    # log scale: a ratio of 2 and of 0.5 get the same intensity, red = LoRA worse
    im = ax.imshow(np.ma.masked_invalid(np.log(values)), cmap=cmap, vmin=-spread, vmax=spread,
                   aspect="auto", interpolation="nearest")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(np.log(values[i, j])) > 0.6 * spread else "#1f1f1f",
                        fontweight="bold" if i == len(table) - 1 else "normal")
    ax.axhline(len(table) - 1.5, color="white", linewidth=3)
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(len(table)), table.index)
    ax.set_xlabel("Forecast horizon (quarters)")
    ax.set_ylabel("Forecast cut-off year")
    ax.set_title(f"MAFE LoRA / MAFE zero-shot, {unit(transform)} (blue: LoRA better, red: worse)")
    cbar = fig.colorbar(im, ax=ax, label="Ratio")
    ticks = np.array([0.5, 0.75, 1, 1.33, 2])
    ticks = ticks[np.abs(np.log(ticks)) <= spread]
    cbar.set_ticks(np.log(ticks), labels=[f"{t:g}" for t in ticks])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_level_comparison(mafe, path):
    styles = {"Level, zero-shot": ("#8c8c8c", "-"), "Level, LoRA": ("#0b3d91", "-"),
              "From growth, zero-shot": ("#8c8c8c", "--"), "From growth, LoRA": ("#0b3d91", "--")}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, (color, style) in styles.items():
        ax.plot(mafe.index, mafe[label], color=color, linestyle=style, marker="o", markersize=4, linewidth=2,
                label=label)
    ax.set_xlabel("Forecast horizon (quarters)")
    ax.set_ylabel("MAFE on the level")
    ax.set_xticks(mafe.index)
    ax.set_title(f"{TARGET_COLUMN}: level forecast directly, or rebuilt from growth forecasts")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


################## Save #####################################

def save_excel(path, by_horizon, relative_by_year, dm, training, results):
    settings = pd.DataFrame({"setting": ["TRAIN_SERIES", "REFIT_EVERY", "LORA_CONFIG", "LEARNING_RATE", "NUM_STEPS",
                                         "BATCH_SIZE", "USE_VALIDATION", "SEED", "LAST_FINAL"],
                             "value": [str(v) for v in [TRAIN_SERIES, REFIT_EVERY, LORA_CONFIG, LEARNING_RATE, NUM_STEPS,
                                                        BATCH_SIZE, USE_VALIDATION, SEED, LAST_FINAL]]})
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        by_horizon.to_excel(writer, sheet_name="by_horizon")
        relative_by_year.to_excel(writer, sheet_name="MAFE_ratio_by_year")
        dm.to_excel(writer, sheet_name="DM_test")
        training.to_excel(writer, sheet_name="training", index=False)
        settings.to_excel(writer, sheet_name="settings", index=False)
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

def main(pipeline=None):
    if pipeline is None:
        pipeline = load_model(MODEL_PATH)

    print("Step 2 - Load data", flush=True)
    data = load_bmpe()
    print(f"  {len(data)} quarters, {len(data.columns) - 1} series for fine-tuning ({TRAIN_SERIES})", flush=True)

    all_results = {}
    for transform in TRANSFORMS:
        print(f"\nStep 3 - {TARGET_COLUMN}, {unit(transform)}: zero-shot and LoRA forecasts", flush=True)
        output_dir = os.path.join(OUTPUT_DIR, transform)
        os.makedirs(output_dir, exist_ok=True)
        results, training = run(pipeline, transform_panel(data, transform), transform)
        all_results[transform] = results
        by_horizon, relative_by_year = evaluate(results)
        dm = dm_table(results.dropna(subset=["actual"]), "lora", "zero_shot", "actual")

        print("\n  Scored on final quarters (up to %s):" % LAST_FINAL)
        print(by_horizon.round(3).to_string())
        print("\n  Diebold-Mariano, LoRA vs zero-shot (negative: LoRA more accurate; beyond h = 1 the test "
              "rejects too often with ~100 cut-offs, so read p < 0.01 as clear, 0.01-0.10 as weak):")
        print(dm.round(3).to_string())
        save_excel_unlocked(os.path.join(output_dir, "lora_vs_zeroshot.xlsx"),
                            by_horizon, relative_by_year, dm, training, results)
        plot_by_horizon(by_horizon, transform, os.path.join(output_dir, "mafe_by_horizon.png"))
        plot_relative_by_year(relative_by_year, transform, os.path.join(output_dir, "relative_mafe_by_year.png"))

    if {"level", "qoq"} <= set(all_results):
        print(f"\nStep 4 - {TARGET_COLUMN} level: forecast directly, or rebuilt from the growth forecasts", flush=True)
        implied = implied_levels(all_results["qoq"], data[["DATE", TARGET_COLUMN]])
        mafe, rmse, dm = compare_levels(all_results["level"], implied)
        print("\n  MAFE on the level, same cut-offs and horizons for all four:")
        print(mafe.round(0).to_string())
        path = os.path.join(OUTPUT_DIR, "level_comparison.xlsx")
        try:
            writer = pd.ExcelWriter(path, engine="openpyxl")
        except PermissionError:
            path = path.replace(".xlsx", f"_{pd.Timestamp.now():%Y%m%d_%H%M%S}.xlsx")
            writer = pd.ExcelWriter(path, engine="openpyxl")
        with writer:
            mafe.to_excel(writer, sheet_name="MAFE")
            rmse.to_excel(writer, sheet_name="RMSE")
            for i, (label, table) in enumerate(dm.items()):
                sheet = f"DM_{i + 1}"
                table.to_excel(writer, sheet_name=sheet, startrow=1)
                writer.sheets[sheet].cell(row=1, column=1, value=f"{label} vs Level, zero-shot (negative: more accurate)")
            implied.to_excel(writer, sheet_name="implied_levels", index=False)
        print(f"  Saved {path}")
        plot_level_comparison(mafe, os.path.join(OUTPUT_DIR, "level_comparison.png"))


if __name__ == "__main__":
    main()
