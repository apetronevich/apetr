############################################################################################
#   Chronos2 zero-shot avec covariables : quel groupe de covariables choisir ?
#                                   - Anna Petronevich
#                         -----
#    Expanding-window zero-shot forecasts of TARGET_COLUMN, 1..PREDICTION_LENGTH quarters
#    ahead, from every cut-off, with different sets of covariates:
#      - the projection assumptions (HYPOTHESES) are known over the forecast horizon: their
#        future values are given to the model (future_df);
#      - the investment-equation variables (EQUATION) are past-only;
#      - the Covid dummies (DUMMIES) are past-only: before 2020 nobody knew a shock was
#        coming, so giving their future values would announce Covid in advance.
#    Three runs: target in level; target in q-o-q growth with the covariates as they are
#    ("qoq"); target in q-o-q growth with the covariates transformed too ("qoq_cov": growth
#    rates, differences or unchanged, see COVARIATE_TRANSFORM_OVERRIDES).
#    Choosing the covariates on the whole sample and then reporting their score would be
#    look-ahead. The sample is split:
#      - selection period (cut-offs up to SELECTION_END, errors observable by then):
#        forward selection adds one group at a time while it lowers the MAFE, averaged
#        over horizons, by at least MIN_IMPROVEMENT;
#      - evaluation period (cut-offs after SELECTION_END, final quarters up to LAST_FINAL):
#        the selected set and fixed reference sets are scored out of sample against the
#        forecast without covariates (MAFE ratio, RMSE ratio, Diebold-Mariano test).
#    Caveat: the future values of the assumptions in the file are the realised ones, not
#    the assumptions available at each cut-off: with them the forecast is conditional on
#    a perfect foresight of the assumptions (an upper bound). The past-only run of the
#    same assumptions shows how much of the gain comes from that.
#
#    Inputs:
#      - MODEL_PATH ; INPUT_FILE (colonne DATE, la cible et les covariables)
#    Outputs (OUTPUT_DIR/<level|qoq|qoq_cov>) :
#      - covariate_selection.xlsx : ensembles testés, sélection pas à pas, scores sur la
#        période de sélection et d'évaluation, tests de Diebold-Mariano, prévisions
#      - mafe_ratio_evaluation.png, mafe_ratio_selection.png
#                         -----
#  A.Petronevich et AI
############################################################################################


############### Settings ##################################################################

MODEL_PATH = "C:\\TSFM"
INPUT_FILE = "Inputs/RawDataInvMPEMars2026_19992026q1_covstruct.xlsx"
SHEET = 0
DATE_COLUMN = "DATE"
TARGET_COLUMN = "P51_S11S12S14A_7CH"

LAST_OBSERVED = "2026Q1"        # last quarter of the target used (context and ground truth)
LAST_FINAL = "2022Q4"           # later quarters are still revised: not used as ground truth

# Projection assumptions, known over the forecast horizon, by group (one group = one block
# that forward selection can add)
HYPOTHESES = {
    "fr_cmd": ["fr_cmd", "fr_cmdex", "fr_cmdin"],
    "fr_cxd": ["fr_cxd", "fr_cxdex", "fr_cxdin"],
    "fr_een": ["fr_eenm", "fr_eenx"],
    "fr_ltn": ["fr_ltn", "fr_ltn_2y", "fr_ltn_3y", "fr_ltn_5y"],
    "fr_wdr": ["fr_wdr", "fr_wdrex", "fr_wdrin"],
    "BRENT": ["BRENT"],
    "TEURIBOR": ["TEURIBOR"],
    "USDEXR": ["USDEXR"],
}
# Variables of the investment equation: past-only
EQUATION = ["DFHP51S11S12S14A_7CH", "HTL_B1_SM_7CH",
            "HTL_CKR", "HTL_DF_HP51_7CH", "L_C_P51_S11S12S14A_7CH",
            "VP_D_ECART_L_B1_SM_7CH", "VP_D_ECART_L_CKR_S11S12S14A", "VP_DHTL_B1_SM_7CH", "VP_DHTL_CKR"]
# Covid dummies: past-only (their future values were not known before 2020)
DUMMIES = ["DU20Q1", "DU20Q2", "DU20Q3"]

FIRST_CUTOFF = "2000Q1"
MIN_CONTEXT = 20                # quarters of history required at a cut-off (the file starts in 1999)
SELECTION_END = "2012Q4"        # last cut-off (and last observed quarter) of the selection period
PREDICTION_LENGTH = 12
# "level": target in level, covariates as they are. "qoq": target in q-o-q growth,
# covariates as they are. "qoq_cov": target in q-o-q growth, covariates transformed too.
TRANSFORMS = ["level", "qoq", "qoq_cov"]
# How each covariate is transformed in the "qoq_cov" run: "growth" = q-o-q % change,
# "diff" = first difference, "none" = unchanged. Covariates not listed follow the rule:
# dummies unchanged; logs (names starting with L_) and series with zero or negative
# values -> "diff"; otherwise "growth". Rates are differenced (a % change of a rate near
# zero is meaningless); error-correction gaps enter an ECM in level.
COVARIATE_TRANSFORM_OVERRIDES = {
    "fr_ltn": "diff", "fr_ltn_2y": "diff", "fr_ltn_3y": "diff", "fr_ltn_5y": "diff",
    "TEURIBOR": "diff", "HTL_CKR": "diff",                       # rates
    "DFHP51S11S12S14A_7CH": "growth", "HTL_B1_SM_7CH": "growth",  # levels
    "HTL_DF_HP51_7CH": "growth",
    "VP_D_ECART_L_B1_SM_7CH": "none", "VP_D_ECART_L_CKR_S11S12S14A": "none",   # gaps
    "VP_DHTL_B1_SM_7CH": "none", "VP_DHTL_CKR": "none",          # already first differences
}
QUANTILE_LEVELS = [0.1, 0.5, 0.9]

MIN_IMPROVEMENT = 0.01          # forward selection: a group must lower the average MAFE ratio by 1%
MAX_GROUPS = 5                  # forward selection: at most this many groups

OUTPUT_DIR = "Outputs/covariate_selection"


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
    return {"level": "level", "qoq": "q-o-q growth, %, covariates as they are",
            "qoq_cov": "q-o-q growth, %, covariates transformed"}[transform]


def covariate_transform_rules(data):
    """Transformation of each covariate in the qoq_cov run."""
    rules = {}
    for c in data.columns:
        if c in ("DATE", TARGET_COLUMN):
            continue
        if c in COVARIATE_TRANSFORM_OVERRIDES:
            rules[c] = COVARIATE_TRANSFORM_OVERRIDES[c]
        elif c in DUMMIES:
            rules[c] = "none"
        elif c.upper().startswith("L_") or (data[c].dropna() <= 0).any():
            rules[c] = "diff"
        else:
            rules[c] = "growth"
    return rules


def transform_covariates(data, rules):
    """Applied to the whole file, so the future values continue the past ones (the first
    future growth rate uses the last observed value)."""
    out = data.copy()
    for c, rule in rules.items():
        if rule == "growth":
            out[c] = 100 * data[c].pct_change()
        elif rule == "diff":
            out[c] = data[c].diff() * (100 if c.upper().startswith("L_") else 1)
    return out.replace([np.inf, -np.inf], np.nan)


def load_data():
    """The whole file (covariates may extend past LAST_OBSERVED: future values of the
    assumptions) with column names matched ignoring case. Unknown covariates are dropped."""
    raw = pd.read_excel(INPUT_FILE, sheet_name=SHEET)
    by_lower = {str(c).strip().lower(): c for c in raw.columns}

    def column(name):
        return by_lower.get(name.lower())

    if column(DATE_COLUMN) is None or column(TARGET_COLUMN) is None:
        raise KeyError(f"{DATE_COLUMN} or {TARGET_COLUMN} not found in {INPUT_FILE}")
    wanted = [c for cols in HYPOTHESES.values() for c in cols] + EQUATION + DUMMIES
    missing = [c for c in wanted if column(c) is None]
    if missing:
        print(f"  Not in {INPUT_FILE}, left out: {missing}", flush=True)

    data = pd.DataFrame({"DATE": to_dates(raw[column(DATE_COLUMN)])})
    data[TARGET_COLUMN] = pd.to_numeric(raw[column(TARGET_COLUMN)], errors="coerce")
    for c in wanted:
        if column(c) is not None and c.lower() != TARGET_COLUMN.lower():
            data[c] = pd.to_numeric(raw[column(c)], errors="coerce")
    return data.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)


def covariate_groups(data):
    """Unit of forward selection -> (past-only columns, known-future columns)."""
    present = set(data.columns)
    groups = {name: ([], [c for c in cols if c in present]) for name, cols in HYPOTHESES.items()}
    groups["equation"] = ([c for c in EQUATION if c in present], [])
    groups["dummies"] = ([c for c in DUMMIES if c in present], [])
    return {name: g for name, g in groups.items() if g[0] or g[1]}


################## Generate predictions ####################

def future_timestamps(contexts):
    """The forecast timestamps exactly as Chronos2 builds them, so future_df matches."""
    try:
        from chronos.df_utils import make_future_df
        return make_future_df(contexts, PREDICTION_LENGTH, id_column="item_id", timestamp_column="DATE")
    except ImportError:    # without chronos (tests): same construction with pandas
        dates = contexts.loc[contexts["item_id"] == contexts["item_id"].iloc[0], "DATE"]
        offset = pd.tseries.frequencies.to_offset(pd.infer_freq(dates))
        last = contexts.groupby("item_id", sort=False)["DATE"].last()
        return pd.DataFrame({"item_id": np.repeat(last.index, PREDICTION_LENGTH),
                             "DATE": [d + k * offset for d in last for k in range(1, PREDICTION_LENGTH + 1)]})


def forecast(pipeline, series, full, past_cols, future_cols, cutoffs):
    """Forecasts from every cut-off: context = all quarters up to the cut-off (target and
    covariates); future covariates = their values over the next PREDICTION_LENGTH quarters
    (missing beyond the end of the file, which Chronos2 masks)."""
    columns = ["DATE", TARGET_COLUMN] + past_cols + future_cols
    contexts = pd.concat(
        [series.iloc[:pos + 1][columns].assign(item_id=series["DATE"].iloc[pos].strftime("%Y-%m-%d"))
         for pos in cutoffs],
        ignore_index=True,
    )
    future_df = None
    if future_cols:
        future_df = future_timestamps(contexts)
        future_quarter = future_df["DATE"].dt.to_period("Q")
        by_quarter = full.assign(q=full["DATE"].dt.to_period("Q")).drop_duplicates("q").set_index("q")
        for c in future_cols:
            future_df[c] = future_quarter.map(by_quarter[c]).to_numpy()
    pred = pipeline.predict_df(
        contexts,
        future_df=future_df,
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


class Experiment:
    """Runs and caches the forecasts of each covariate set for one target transformation."""

    def __init__(self, pipeline, data, transform):
        self.rules = covariate_transform_rules(data) if transform == "qoq_cov" else {}
        if self.rules:
            data = transform_covariates(data, self.rules)
        self.pipeline, self.full, self.transform = pipeline, data, transform
        observed = data[data["DATE"].dt.to_period("Q") <= quarter(LAST_OBSERVED)].copy()
        if transform in ("qoq", "qoq_cov"):
            observed[TARGET_COLUMN] = 100 * observed[TARGET_COLUMN].pct_change()
        self.series = observed.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
        last = len(self.series) - 1
        quarters = self.series["DATE"].dt.to_period("Q")
        self.cutoffs = [pos for pos in range(MIN_CONTEXT - 1, last) if quarters.iloc[pos] >= quarter(FIRST_CUTOFF)]
        self.groups = covariate_groups(data)
        self.cache = {}
        self.truth = self._truth()

    def _truth(self):
        """Actual value and target quarter for each cut-off and horizon (by position)."""
        rows = [(self.series["DATE"].iloc[c], h, c + h) for c in self.cutoffs for h in range(1, PREDICTION_LENGTH + 1)]
        truth = pd.DataFrame(rows, columns=["cutoff", "horizon", "pos"])
        inside = truth["pos"] < len(self.series)
        pos = truth["pos"].clip(upper=len(self.series) - 1)
        truth["target_quarter"] = np.where(inside, self.series["DATE"].dt.to_period("Q").to_numpy()[pos], pd.NaT)
        truth["actual"] = np.where(inside, self.series[TARGET_COLUMN].to_numpy()[pos], np.nan)
        return truth.drop(columns="pos")

    def columns(self, names, hypotheses_future=True):
        past, future = [], []
        for name in names:
            p, f = self.groups[name]
            past += p
            if name in HYPOTHESES and not hypotheses_future:
                past += f
            else:
                future += f
        return past, future

    def run(self, names, hypotheses_future=True):
        key = (tuple(sorted(names)), hypotheses_future)
        if key not in self.cache:
            past, future = self.columns(key[0], hypotheses_future)
            pred = forecast(self.pipeline, self.series, self.full, past, future, self.cutoffs)
            self.cache[key] = pred.merge(self.truth, on=["cutoff", "horizon"])
        return self.cache[key]


################## Forecast evaluation ######################

def set_name(names, hypotheses_future=True):
    if not names:
        return "none"
    name = " + ".join(sorted(names))
    return name if hypotheses_future or not any(n in HYPOTHESES for n in names) else name + " (assumptions past-only)"


def sample(results, period):
    """Selection: cut-offs and target quarters up to SELECTION_END (what was known then).
    Evaluation: cut-offs after SELECTION_END, final target quarters (up to LAST_FINAL)."""
    q_cut = results["cutoff"].dt.to_period("Q")
    final = results["target_quarter"] <= quarter(LAST_FINAL)
    if period == "selection":
        keep = (q_cut <= quarter(SELECTION_END)) & (results["target_quarter"] <= quarter(SELECTION_END))
    elif period == "evaluation":
        keep = (q_cut > quarter(SELECTION_END)) & final
    else:
        keep = final
    return results[keep & results["actual"].notna()]


def mafe(results, period):
    s = sample(results, period)
    return (s["predictions"] - s["actual"]).abs().groupby(s["horizon"]).mean()


def rmse(results, period):
    s = sample(results, period)
    return np.sqrt(((s["predictions"] - s["actual"]) ** 2).groupby(s["horizon"]).mean())


def criterion(results, baseline):
    """Average over horizons of MAFE(set) / MAFE(no covariates), selection period."""
    return (mafe(results, "selection") / mafe(baseline, "selection")).mean()


def diebold_mariano(loss_a, loss_b, horizon):
    """Diebold-Mariano test of equal accuracy on loss_a - loss_b, ordered by cut-off.
    h-step forecasts made in consecutive quarters overlap, so their errors are correlated
    up to lag h-1: Newey-West variance with h-1 lags (Bartlett weights) and the
    Harvey-Leybourne-Newbold correction. Negative statistic: a more accurate.
    With small samples the test rejects too often beyond h = 1 (about 10% at the 5% level
    with 100 cut-offs, more with fewer): read p < 0.01 as clear, 0.01-0.10 as weak."""
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
    stat *= np.sqrt(max(n + 1 - 2 * horizon + horizon * (horizon - 1) / n, 1e-12) / n)
    try:
        from scipy.stats import t
        p_value = 2 * t.sf(abs(stat), df=n - 1)
    except ImportError:
        from math import erf, sqrt
        p_value = 2 * (1 - 0.5 * (1 + erf(abs(stat) / sqrt(2))))
    return stat, p_value


def dm_vs_baseline(results, baseline, period="evaluation"):
    keys = ["cutoff", "horizon"]
    both = sample(results, period)[keys + ["predictions", "actual"]].merge(
        sample(baseline, period)[keys + ["predictions"]], on=keys, suffixes=("", "_base"))
    rows = {}
    for h, g in both.sort_values("cutoff").groupby("horizon"):
        rows[h] = diebold_mariano((g["predictions"] - g["actual"]).abs(),
                                  (g["predictions_base"] - g["actual"]).abs(), h)
    return pd.DataFrame(rows, index=["DM stat", "p-value"]).T


################## Selection ################################

def forward_selection(exp):
    """Greedy forward selection of groups on the selection period."""
    baseline = exp.run([])
    chosen, best, steps = [], 1.0, []
    while len(chosen) < MAX_GROUPS:
        trials = {g: criterion(exp.run(chosen + [g]), baseline) for g in exp.groups if g not in chosen}
        if not trials:
            break
        for g, value in trials.items():
            steps.append({"step": len(chosen) + 1, "candidate set": set_name(chosen + [g]),
                          "average MAFE ratio, selection period": value})
        g_best = min(trials, key=trials.get)
        if trials[g_best] > best * (1 - MIN_IMPROVEMENT):
            break
        chosen.append(g_best)
        best = trials[g_best]
        steps.append({"step": len(chosen), "candidate set": f"-> kept: {set_name(chosen)}",
                      "average MAFE ratio, selection period": best})
    return chosen, pd.DataFrame(steps)


def candidate_sets(exp, selected):
    """Sets scored on the evaluation period: (names, hypotheses known in the future)."""
    hyps = [g for g in exp.groups if g in HYPOTHESES]
    sets = [([], True), (selected, True)]
    if any(g in HYPOTHESES for g in selected):
        sets.append((selected, False))
    sets += [([g], True) for g in exp.groups]
    if hyps:
        sets += [(hyps, True), (hyps, False)]
    if "equation" in exp.groups:
        sets.append((["equation"] + (["dummies"] if "dummies" in exp.groups else []), True))
    sets.append((list(exp.groups), True))
    unique = {}
    for names, fut in sets:
        unique.setdefault((tuple(sorted(names)), fut), (names, fut))
    return list(unique.values())


def score_sets(exp, sets, selected):
    baseline = exp.run([])
    rows, mafe_ratio, sel_ratio, dms, definitions = [], {}, {}, {}, []
    for names, fut in sets:
        res = exp.run(names, fut)
        label = set_name(names, fut) + ("  [selected]" if sorted(names) == sorted(selected) and fut else "")
        ratio = mafe(res, "evaluation") / mafe(baseline, "evaluation")
        mafe_ratio[label] = ratio
        sel_ratio[label] = mafe(res, "selection") / mafe(baseline, "selection")
        dm = dm_vs_baseline(res, baseline)
        dms[label] = dm
        rmse_ratio = rmse(res, "evaluation") / rmse(baseline, "evaluation")
        rows.append({"set": label,
                     "average MAFE ratio, evaluation": ratio.mean(),
                     "average RMSE ratio, evaluation": rmse_ratio.mean(),
                     "average MAFE ratio, selection": sel_ratio[label].mean(),
                     "horizons better at 5% (DM)": int(((dm["DM stat"] < 0) & (dm["p-value"] < 0.05)).sum()),
                     "horizons worse at 5% (DM)": int(((dm["DM stat"] > 0) & (dm["p-value"] < 0.05)).sum())})
        past, future = exp.columns(names, fut)
        definitions.append({"set": label, "past-only covariates": ", ".join(past),
                            "known-future covariates": ", ".join(future)})
    summary = pd.DataFrame(rows).sort_values("average MAFE ratio, evaluation").reset_index(drop=True)
    order = list(summary["set"])
    return (summary, pd.DataFrame(mafe_ratio).T.reindex(order), pd.DataFrame(sel_ratio).T.reindex(order),
            dms, pd.DataFrame(definitions))


################## Plot #####################################

def plot_ratio_heatmap(table, title, path):
    values = table.to_numpy(dtype=float)
    finite = values[np.isfinite(values) & (values > 0)]
    spread = max(np.percentile(np.abs(np.log(finite)), 95), 0.05) if finite.size else 0.5
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e6e6e6")
    fig, ax = plt.subplots(figsize=(11, 0.4 * len(table) + 2))
    # log scale: ratios of 2 and 0.5 get the same intensity; blue = better than no covariates
    im = ax.imshow(np.ma.masked_invalid(np.log(values)), cmap=cmap, vmin=-spread, vmax=spread,
                   aspect="auto", interpolation="nearest")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(np.log(values[i, j])) > 0.6 * spread else "#1f1f1f")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(len(table)), table.index, fontsize=8)
    for tick in ax.get_yticklabels():
        if "[selected]" in tick.get_text():
            tick.set_fontweight("bold")
    ax.set_xlabel("Forecast horizon (quarters)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, label="MAFE ratio to no covariates (blue: better)")
    ticks = np.array([0.5, 0.67, 0.8, 1, 1.25, 1.5, 2])
    ticks = ticks[np.abs(np.log(ticks)) <= spread]
    cbar.set_ticks(np.log(ticks), labels=[f"{t:g}" for t in ticks])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


################## Save #####################################

def save_excel(path, summary, definitions, steps, mafe_eval, mafe_sel, dms, forecasts, rules):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        if rules:
            pd.Series(rules, name="transformation").rename_axis("covariate").to_excel(
                writer, sheet_name="covariate_transforms")
        definitions.to_excel(writer, sheet_name="sets", index=False)
        steps.to_excel(writer, sheet_name="forward_selection", index=False)
        mafe_eval.to_excel(writer, sheet_name="MAFE_ratio_evaluation")
        mafe_sel.to_excel(writer, sheet_name="MAFE_ratio_selection")
        pd.concat(dms, names=["set", "horizon"]).to_excel(writer, sheet_name="DM_tests")
        forecasts.to_excel(writer, sheet_name="forecasts", index=False)


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
    data = load_data()
    print(f"  {quarter_label(data['DATE'].iloc[0])} to {quarter_label(data['DATE'].iloc[-1])}, "
          f"{len(data.columns) - 2} covariates", flush=True)

    for transform in TRANSFORMS:
        print(f"\nStep 3 - {TARGET_COLUMN}, {unit(transform)}", flush=True)
        output_dir = os.path.join(OUTPUT_DIR, transform)
        os.makedirs(output_dir, exist_ok=True)
        exp = Experiment(pipeline, data, transform)
        if exp.rules:
            by_rule = pd.Series(exp.rules).groupby(lambda c: exp.rules[c]).apply(lambda s: ", ".join(s.index))
            print("  covariate transformations: " + "; ".join(f"{r}: {cols}" for r, cols in by_rule.items()), flush=True)
        cut = exp.series["DATE"]
        print(f"  cut-offs {quarter_label(cut.iloc[exp.cutoffs[0]])} to {quarter_label(cut.iloc[exp.cutoffs[-1]])}; "
              f"selection up to {SELECTION_END}, evaluation after; groups: {', '.join(exp.groups)}", flush=True)

        selected, steps = forward_selection(exp)
        print(f"  Forward selection on the selection period: {set_name(selected)}", flush=True)
        summary, mafe_eval, mafe_sel, dms, definitions = score_sets(exp, candidate_sets(exp, selected), selected)

        print(f"\n  Evaluation period (cut-offs after {SELECTION_END}, final quarters up to {LAST_FINAL}), "
              f"ratios to no covariates (< 1: better):")
        print(summary.round(3).to_string(index=False))

        forecasts = pd.concat([res.assign(set=set_name(list(k[0]), k[1])) for k, res in exp.cache.items()])
        save_excel_unlocked(os.path.join(output_dir, "covariate_selection.xlsx"),
                            summary, definitions, steps, mafe_eval, mafe_sel, dms, forecasts, exp.rules)
        plot_ratio_heatmap(mafe_eval, f"Evaluation period (cut-offs after {SELECTION_END})\nMAFE ratio to no "
                                      f"covariates, {unit(transform)}", os.path.join(output_dir, "mafe_ratio_evaluation.png"))
        plot_ratio_heatmap(mafe_sel, f"Selection period (up to {SELECTION_END})\nMAFE ratio to no covariates, "
                                     f"{unit(transform)}", os.path.join(output_dir, "mafe_ratio_selection.png"))


if __name__ == "__main__":
    main()
