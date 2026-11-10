# =============================================================================
# STATISTICAL ANALYSIS: FOUR CSV INPUT FILES
#
# Input files in BASE_FOLDER:
#   All_Participants_Results_ProtocolI_independent.csv
#   All_Participants_Results_ProtocolII_independent.csv
#   All_Participants_Results_ProtocolI_matched.csv
#   All_Participants_Results_ProtocolII_matched.csv
#
# Each CSV must contain:
# Participant, Accuracy_RG, Kappa_RG, Precision_RG, Recall_RG,
# Specificity_RG, Accuracy_FBCSP, Kappa_FBCSP, Precision_FBCSP,
# Recall_FBCSP, Specificity_FBCSP
#
# Analyses:
#   1. Mann-Whitney U: independent participants, Protocol II > Protocol I
#   2. Wilcoxon signed-rank: matched participants:
#        a. Protocol II > Protocol I, separately for RG and FBCSP
#        b. FBCSP vs RG within each protocol
#   3. Factorial linear mixed-effects model:
#        Protocol * Method * Cohort, participant random intercept
#   4. Holm-Bonferroni correction and publication-quality figures
# Designed by: Aura Ximena Gonzalez Cely
# v1.0
# Python v.3.10.9
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon
import statsmodels.formula.api as smf
from scipy.stats import norm

BASE_FOLDER = Path(r"A:\ARTICLES\Frontiers2026orMDPI2026\GithubFiles")
OUTPUT_FOLDER = BASE_FOLDER / "Statistical_Analysis_Results"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
N_INDEPENDENT = 7
N_MATCHED = 6

METRICS = ["Accuracy", "Kappa", "Precision", "Recall", "Specificity"]
METHODS = ["RG", "FBCSP"]
PROTOCOLS = ["I", "II"]
COHORTS = ["independent", "matched"]

INPUT_FILES = {
    ("I", "independent"): BASE_FOLDER / "All_Participants_Results_ProtocolI_independent.csv",
    ("II", "independent"): BASE_FOLDER / "All_Participants_Results_ProtocolII_independent.csv",
    ("I", "matched"): BASE_FOLDER / "All_Participants_Results_ProtocolI_matched.csv",
    ("II", "matched"): BASE_FOLDER / "All_Participants_Results_ProtocolII_matched.csv",
}
REQUIRED_COLUMNS = (["Participant"] + [f"{metric}_{method}" for metric in METRICS for method in METHODS])

##################################################################### Functions ###################################################################################################

def holm_correction(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan)

    valid = np.flatnonzero(np.isfinite(p_values))
    if valid.size == 0:
        return adjusted

    order = valid[np.argsort(p_values[valid])]
    sorted_p = p_values[order]
    n = len(sorted_p)
    adjusted_sorted = np.maximum.accumulate((n - np.arange(n)) * sorted_p)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted

def add_holm(results, p_column="P_value"):
    if not results:
        return results
    adjusted = holm_correction([row[p_column] for row in results])
    for row, p_adjusted in zip(results, adjusted):
        row["Holm_P_value"] = p_adjusted
    return results

def safe_wilcoxon(x, y, alternative="two-sided"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    differences = x - y
    if np.allclose(differences, 0):
        return 0.0, 1.0
    result = wilcoxon( x, y, alternative=alternative, zero_method="wilcox", method="auto",)
    return float(result.statistic), float(result.pvalue)

def safe_mannwhitney(x, y, alternative="two-sided"):
    result = mannwhitneyu( np.asarray(x, dtype=float), np.asarray(y, dtype=float), alternative=alternative, method="auto",)
    return float(result.statistic), float(result.pvalue)

def read_results_csv(protocol, cohort, file_path):
    if not file_path.is_file():
        raise FileNotFoundError(f"Input CSV not found:\n{file_path}")
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {file_path.name}: {missing}")
    df = df[REQUIRED_COLUMNS].copy()
    if df["Participant"].isna().any():
        raise ValueError(f"Missing participant ID in {file_path.name}")
    df["Participant"] = df["Participant"].astype("string").str.strip()
    if df["Participant"].eq("").any():
        raise ValueError(f"Empty participant ID in {file_path.name}")
    if df["Participant"].duplicated().any():
        duplicates = df.loc[df["Participant"].duplicated(keep=False), "Participant"].tolist()
        raise ValueError(f"Duplicate participant IDs in {file_path.name}: {duplicates}")
    for column in REQUIRED_COLUMNS[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    numeric = df[REQUIRED_COLUMNS[1:]]
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        bad_columns = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"Missing or non-numeric metric values in {file_path.name}. " f"Columns with missing values: {bad_columns}")
    results = {}
    for row in df.itertuples(index=False):
        participant = row[0]
        results[participant] = { metric: { method: float(getattr(row, f"{metric}_{method}")) for method in METHODS} for metric in METRICS}
    print(f"Loaded Protocol {protocol} / {cohort}: " f"{len(results)} participants ({file_path.name})")
    return results

def append_cohort(cohort_data, id_map, cohort_name, protocol):
    for original_id, participant_data in cohort_data.items():
        anonymized_id = id_map[original_id]
        for method in METHODS:
            row = {
                "Participant": anonymized_id,
                "Cohort": cohort_name,
                "Protocol": protocol,
                "Method": method,
            }
            for metric in METRICS:
                row[metric] = participant_data[metric][method]
            combined_rows.append(row)

def plot_planned_contrasts(data, metrics, comparison_order, output_name, xlabel="Estimated difference (95% confidence interval)", figsize=(13, 15)):
    fig, axes = plt.subplots(nrows=len(metrics), ncols=1, figsize=figsize, sharex=False)
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        metric_df = data[data["Metric"] == metric].copy()
        metric_df["Comparison"] = pd.Categorical( metric_df["Comparison"], categories=comparison_order, ordered=True)
        metric_df = metric_df.sort_values("Comparison")
        y_positions = np.arange(len(metric_df))
        ax.axvline( 0, color="black", linestyle="--", linewidth=1)
        for y, (_, row) in enumerate(metric_df.iterrows()):
            ax.errorbar(row["Estimate"], y, xerr=[[row["Estimate"] - row["CI_lower"]], [row["CI_upper"] - row["Estimate"]]], fmt="o", color="black", ecolor="black",  capsize=4, markersize=6, linewidth=1.5)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(metric_df["Comparison"])
        ax.set_ylabel(metric, fontsize=20)
        ax.tick_params(axis="both", labelsize=20)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        ax.invert_yaxis()
    axes[-1].set_xlabel(xlabel, fontsize=18)
    plt.tight_layout()
    fig.savefig( output_name, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


data = { key: read_results_csv(key[0], key[1], path) for key, path in INPUT_FILES.items()}
# Labels
independent_I = data[("I", "independent")]
independent_II = data[("II", "independent")]
paired_I = data[("I", "matched")]
paired_II = data[("II", "matched")]

participants_independent_I = set(independent_I)
participants_independent_II = set(independent_II)
participants_paired_I = set(paired_I)
participants_paired_II = set(paired_II)

# Sample-size and participant-identity checks.
for label, cohort_data, expected_n in [
    ("Protocol I independent", independent_I, N_INDEPENDENT),
    ("Protocol II independent", independent_II, N_INDEPENDENT),
    ("Protocol I matched", paired_I, N_MATCHED),
    ("Protocol II matched", paired_II, N_MATCHED),
]:
    if len(cohort_data) != expected_n:
        raise ValueError( f"{label}: expected {expected_n} participants, " f"found {len(cohort_data)}.")

if not participants_independent_I.isdisjoint(participants_independent_II):
    raise ValueError("Independent participant IDs must be different between protocols.")

if participants_paired_I != participants_paired_II:
    raise ValueError(
        "Matched participant IDs must be identical in both protocol files. "
        f"Only in Protocol I: {participants_paired_I - participants_paired_II}; "
        f"only in Protocol II: {participants_paired_II - participants_paired_I}"
    )

participants_paired = sorted(participants_paired_I)
if not participants_independent_I.isdisjoint(participants_paired):
    raise ValueError("Protocol I independent and matched IDs overlap.")

if not participants_independent_II.isdisjoint(participants_paired):
    raise ValueError("Protocol II independent and matched IDs overlap.")

print("\nAll four CSV files loaded and validated.")
print("Independent Protocol I:", sorted(participants_independent_I))
print("Independent Protocol II:", sorted(participants_independent_II))
print("Matched participants:", participants_paired)


#############################################################################################################################################################################
# ANALYSIS 1: MANN-WHITNEY U TEST (INDEPENDENT COHORT, 7 vs 7)
# Alternative hypothesis: Protocol II > Protocol I
# Holm correction across 10 tests (5 metrics x 2 methods).

fitted_models = {}
mannwhitney_results = []
for metric in METRICS:
    for method in METHODS:
        values_I = np.array([ independent_I[p][metric][method] for p in sorted(independent_I)])
        values_II = np.array([ independent_II[p][metric][method] for p in sorted(independent_II)])
        statistic, p_value = safe_mannwhitney(values_II, values_I, alternative="greater")
        mannwhitney_results.append({
            "Metric": metric,
            "Method": method,
            "Comparison": "Protocol II > Protocol I",
            "Test": "Mann-Whitney U",
            "N_Protocol_I": len(values_I),
            "N_Protocol_II": len(values_II),
            "Mean_Protocol_I": np.mean(values_I),
            "Mean_Protocol_II": np.mean(values_II),
            "Median_Protocol_I": np.median(values_I),
            "Median_Protocol_II": np.median(values_II),
            "U_statistic": statistic,
            "P_value": p_value,
        })

add_holm(mannwhitney_results)
mannwhitney_df = pd.DataFrame(mannwhitney_results)
mannwhitney_df.to_csv(OUTPUT_FOLDER / "Mann_Whitney_Independent_group.csv", index=False)
print("\nMann-Whitney results:")
print(mannwhitney_df.to_string(index=False))

##########################################################################################################################################################################
# ANALYSIS 2: WILCOXON SIGNED-RANK TEST (MATCHED COHORT, n = 6)
# Comparisons per metric:
# a. Protocol II > Protocol I for RG
# b. Protocol II > Protocol I for FBCSP
# c. FBCSP vs RG within Protocol I
# d. FBCSP vs RG within Protocol II
# Holm correction across all 20 tests (5 metrics x 4 comparisons).

wilcoxon_results = []
for metric in METRICS:
    for method in METHODS:
        values_I = np.array([paired_I[p][metric][method] for p in participants_paired])
        values_II = np.array([paired_II[p][metric][method] for p in participants_paired])
        statistic, p_value = safe_wilcoxon(values_II, values_I, alternative="greater")
        wilcoxon_results.append({
            "Metric": metric,
            "Method": method,
            "Comparison": "Protocol II > Protocol I",
            "Test": "Wilcoxon signed-rank",
            "N_Paired": len(participants_paired),
            "Median_Protocol_I": np.median(values_I),
            "Median_Protocol_II": np.median(values_II),
            "W_statistic": statistic,
            "P_value": p_value,
        })

    # FBCSP vs RG within each protocol.
    for protocol, protocol_data in [("I", paired_I), ("II", paired_II)]:
        values_RG = np.array([ protocol_data[p][metric]["RG"] for p in participants_paired])
        values_FBCSP = np.array([ protocol_data[p][metric]["FBCSP"] for p in participants_paired])
        statistic, p_value = safe_wilcoxon(values_FBCSP, values_RG, alternative="two-sided")
        wilcoxon_results.append({
            "Metric": metric,
            "Method": "FBCSP vs RG",
            "Comparison": f"Protocol {protocol}",
            "Test": "Wilcoxon signed-rank",
            "N_Paired": len(participants_paired),
            "Median_RG": np.median(values_RG),
            "Median_FBCSP": np.median(values_FBCSP),
            "W_statistic": statistic,
            "P_value": p_value,
        })

add_holm(wilcoxon_results)
wilcoxon_df = pd.DataFrame(wilcoxon_results)
wilcoxon_df.to_csv(OUTPUT_FOLDER / "Wilcoxon_Matched_group.csv", index=False)
print("\nWilcoxon results:")
print(wilcoxon_df.to_string(index=False))

########################################################################################################################################################################
# LONG-FORMAT DATASET#
# Matched participants: P01-P06, consistent across protocols.
# Protocol I independent: P07-P13.
# Protocol II independent: P14-P20.
# Two rows per participant/protocol (RG and FBCSP): 52 rows.

matched_id_map = {participant: f"P{i:02d}" for i, participant in enumerate(participants_paired, start=1)}
independent_I_id_map = {participant: f"P{i:02d}" for i, participant in enumerate(sorted(independent_I), start=7)}
independent_II_id_map = { participant: f"P{i:02d}" for i, participant in enumerate(sorted(independent_II), start=14)}

combined_rows = []

append_cohort(independent_I, independent_I_id_map, "Independent", "I")
append_cohort(paired_I, matched_id_map, "Matched", "I")
append_cohort(independent_II, independent_II_id_map, "Independent", "II")
append_cohort(paired_II, matched_id_map, "Matched", "II")

combined_df = pd.DataFrame(combined_rows)

expected_rows = 2 * 2 * (N_INDEPENDENT + N_MATCHED)
if len(combined_df) != expected_rows:
    raise ValueError(f"Expected {expected_rows} long-format rows; found {len(combined_df)}.")
if combined_df.isna().any().any():
    raise ValueError("The combined dataset contains missing values.")
if combined_df.duplicated(["Participant", "Protocol", "Method"]).any():
    raise ValueError("Duplicate participant/protocol/method rows detected.")
matched_check = combined_df[combined_df["Cohort"] == "Matched"]
matched_protocol_counts = matched_check.groupby("Participant")["Protocol"].nunique()
if len(matched_protocol_counts) != N_MATCHED or not (matched_protocol_counts == 2).all():
    raise ValueError("Matched participant IDs are not paired across both protocols.")
combined_file = (OUTPUT_FOLDER / "Classification_metrics_per_participant_and_group.csv")
combined_df.to_csv(combined_file, index=False)
print("\nCombined anonymized dataset saved:", combined_file)
print("Combined dataset shape:", combined_df.shape)
print(combined_df.groupby(["Cohort", "Protocol"])["Participant"].nunique())

##########################################################################################################################################################################
# FACTORIAL LINEAR MIXED-EFFECTS MODEL
# Fixed effects: Protocol * Method * Cohort
# Random effect: participant-specific intercept

combined_df["Protocol"] = pd.Categorical(combined_df["Protocol"], categories=["I", "II"])
combined_df["Method"] = pd.Categorical(combined_df["Method"], categories=["RG", "FBCSP"])
combined_df["Cohort"] = pd.Categorical(combined_df["Cohort"], categories=["Independent", "Matched"])
for metric in METRICS:
    print(f"\nFitting mixed-effects model for {metric}...")
    formula = f"{metric} ~ C(Protocol) * C(Method) * C(Cohort)"
    model = smf.mixedlm(formula,data=combined_df, groups=combined_df["Participant"], re_formula="1",)
    fitted = None
    used_optimizer = None
    for optimizer in ["powell", "lbfgs", "bfgs"]:
        try:
            candidate = model.fit( reml=False,  method=optimizer, maxiter=5000, disp=False)
            if candidate.converged:
                fitted = candidate
                used_optimizer = optimizer
                break
        except Exception as exc:
            print(f"  {optimizer} did not fit: {exc}")

    if fitted is None:
        raise RuntimeError(f"Model did not converge for {metric}. " "Statistical analysis stopped.")
    fitted_models[metric] = fitted
    print(f"  Converged using {used_optimizer}.")
    print(fitted.summary())
# plot
conditions = [("I", "RG"), ("I", "FBCSP"), ("II", "RG"), ("II", "FBCSP")]
condition_labels = ["Protocol I\nRG", "Protocol I\nFBCSP", "Protocol II\nRG", "Protocol II\nFBCSP",]
cohort_markers = {"Independent": "o", "Matched": "^"}
color_markers = {"Independent": "tab:green", "Matched": "tab:brown"}
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 20,              
    "axes.labelsize": 20,
    "axes.titlesize": 18,        
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.titlesize": 20,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
rng = np.random.default_rng(42)
for metric_index, metric in enumerate(METRICS):
    ax = axes[metric_index]
    for condition_index, (protocol, method) in enumerate(conditions):
        condition_data = combined_df[(combined_df["Protocol"].astype(str) == protocol) & (combined_df["Method"].astype(str) == method)]
        for cohort, marker in cohort_markers.items():
            cohort_data = condition_data[condition_data["Cohort"].astype(str) == cohort].dropna(subset=[metric])
            if cohort_data.empty:
                continue
            x = condition_index + rng.uniform( -0.07, 0.07, size=len(cohort_data))
            ax.scatter(x, cohort_data[metric], marker=marker, s=65, alpha=0.8, color = color_markers[cohort], edgecolors="black", linewidths=0.7, label=cohort if condition_index == 0 else None, zorder=3,)
    ax.set_xticks(range(4))
    ax.set_xticklabels(condition_labels, fontsize=14)
    ax.set_ylabel(metric, fontsize=20)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_xlim(-0.5, 3.5)
    metric_values = combined_df[metric].dropna()
    if not metric_values.empty:
        padding = max((metric_values.max() - metric_values.min()) * 0.12, 0.02)
        lower_limit = min(0, metric_values.min() - padding)
        upper_limit = max(1, metric_values.max() + padding)
        ax.set_ylim(lower_limit, upper_limit)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.delaxes(axes[5])

legend_handles = [
    plt.Line2D(
        [0], [0],
        marker="o",
        color="black",
        linestyle="None",
        markersize=9,
        markerfacecolor="tab:green",
        markeredgecolor="black",
        label="Independent participants (circles)"
    ),

    plt.Line2D(
        [0], [0],
        marker="^",
        color="black",
        linestyle="None",
        markersize=9,
        markerfacecolor="tab:brown",
        markeredgecolor="black",
        label="Matched participants (triangles)"
    )
]

fig.legend(handles=legend_handles,loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=2, frameon=False, fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUTPUT_FOLDER / "Figure_1_Overall_Classification_Performance.png", dpi=600, bbox_inches="tight")
plt.close(fig)

#############################################################################################################################################################
# Contrasts
w_independent = N_INDEPENDENT / (N_INDEPENDENT + N_MATCHED)
w_matched = N_MATCHED / (N_INDEPENDENT + N_MATCHED)

planned_contrasts = {
    "Protocol II vs I, RG": {
        "C(Protocol)[T.II]": 1,
        "C(Protocol)[T.II]:C(Cohort)[T.Matched]": w_matched,
    },

    "Protocol II vs I, FBCSP": {
        "C(Protocol)[T.II]": 1,
        "C(Protocol)[T.II]:C(Method)[T.FBCSP]": 1,
        "C(Protocol)[T.II]:C(Cohort)[T.Matched]": w_matched,
        "C(Protocol)[T.II]:C(Method)[T.FBCSP]:C(Cohort)[T.Matched]": w_matched,
    },

    "FBCSP vs RG, Protocol I": {
        "C(Method)[T.FBCSP]": 1,
        "C(Method)[T.FBCSP]:C(Cohort)[T.Matched]": w_matched,
    },

    "FBCSP vs RG, Protocol II": {
        "C(Method)[T.FBCSP]": 1,
        "C(Protocol)[T.II]:C(Method)[T.FBCSP]": 1,
        "C(Method)[T.FBCSP]:C(Cohort)[T.Matched]": w_matched,
        "C(Protocol)[T.II]:C(Method)[T.FBCSP]:C(Cohort)[T.Matched]": w_matched,
    },
}

contrast_results = []
for metric in METRICS:
    if metric not in fitted_models:
        print(f"Skipping {metric}: no converged model available.")
        continue
    fitted = fitted_models[metric]
    fixed_effects = fitted.fe_params
    covariance = fitted.cov_params().loc[fixed_effects.index, fixed_effects.index]
    for contrast_name, contrast_terms in planned_contrasts.items():
        contrast_vector = pd.Series( 0.0, index=fixed_effects.index)
        for term, weight in contrast_terms.items():
            if term not in contrast_vector.index:
                raise KeyError(f"Coefficient '{term}' not found in the model for {metric}.")
            contrast_vector[term] = weight
        estimate = float(contrast_vector @ fixed_effects)
        variance = float(contrast_vector @ covariance @ contrast_vector)
        standard_error = np.sqrt(max(variance, 0))
        # Wald test using the normal approximation.
        z_value = estimate / standard_error if standard_error > 0 else np.nan
        p_value = (float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan)
        ci_lower = estimate - 1.96 * standard_error
        ci_upper = estimate + 1.96 * standard_error
        contrast_results.append({
            "Metric": metric,
            "Comparison": contrast_name,
            "Estimate": estimate,
            "SE": standard_error,
            "Z_value": z_value,
            "P_value": p_value,
            "CI_lower": ci_lower,
            "CI_upper": ci_upper,
        })
planned_contrasts_df = pd.DataFrame(contrast_results)
planned_contrasts_df["Holm_P_value"] = np.nan
for comparison, indices in planned_contrasts_df.groupby("Comparison").groups.items():
    adjusted = holm_correction(planned_contrasts_df.loc[indices, "P_value"].to_numpy())
    planned_contrasts_df.loc[indices, "Holm_P_value"] = adjusted
planned_contrasts_file = (OUTPUT_FOLDER / "Planned_Mixed_Model_Contrasts.csv")
planned_contrasts_df.to_csv(planned_contrasts_file, index=False)
print("\nPLANNED MIXED-MODEL CONTRASTS")
print(planned_contrasts_df.to_string(index=False))
#plot
comparison_order = list(planned_contrasts.keys())
planned_contrasts_plot = OUTPUT_FOLDER / "Mixed_Model_Contrasts.png"
plot_planned_contrasts(data=planned_contrasts_df, metrics=METRICS, comparison_order=comparison_order, output_name=planned_contrasts_plot, xlabel="Estimated difference (95% confidence interval)")
print("\nPlanned contrast results saved to:", planned_contrasts_file)

###########################################################################################################################################################################
# PLANNED CONTRASTS: PROTOCOL II VS I WITHIN EACH COHORT AND METHOD
planned_contrasts = {
    "Protocol II vs I (RG, Matched)": {
        "C(Protocol)[T.II]": 1,
        "C(Protocol)[T.II]:C(Cohort)[T.Matched]": 1,
    },

    "Protocol II vs I (FBCSP, Matched)": {
        "C(Protocol)[T.II]": 1,
        "C(Protocol)[T.II]:C(Method)[T.FBCSP]": 1,
        "C(Protocol)[T.II]:C(Cohort)[T.Matched]": 1,
        "C(Protocol)[T.II]:C(Method)[T.FBCSP]:C(Cohort)[T.Matched]": 1,
    },

    "Protocol II vs I (RG, Independent)": {
        "C(Protocol)[T.II]": 1,
    },

    "Protocol II vs I (FBCSP, Independent)": {
        "C(Protocol)[T.II]": 1,
        "C(Protocol)[T.II]:C(Method)[T.FBCSP]": 1,
    },
}

contrast_results = []
for metric in METRICS:
    if metric not in fitted_models:
        print(f"Skipping {metric}: no converged model available.")
        continue
    fitted = fitted_models[metric]
    fixed_effects = fitted.fe_params
    covariance = fitted.cov_params().loc[fixed_effects.index, fixed_effects.index]
    for contrast_name, contrast_terms in planned_contrasts.items():
        # Construct contrast vector.
        contrast_vector = pd.Series(0.0, index=fixed_effects.index)
        for term, weight in contrast_terms.items():
            if term not in contrast_vector.index:
                raise KeyError(f"Coefficient '{term}' not found for {metric}.")
            contrast_vector[term] = weight
        # Estimate the difference: Protocol II - Protocol I.
        estimate = float(contrast_vector @ fixed_effects)
        # Standard error.
        variance = float(contrast_vector @ covariance @ contrast_vector)
        standard_error = np.sqrt(max(variance, 0))
        # Wald test.
        z_value = (estimate / standard_error if standard_error > 0 else np.nan)
        p_value = (float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan)
        # 95% confidence interval.
        ci_lower = estimate - 1.96 * standard_error
        ci_upper = estimate + 1.96 * standard_error
        contrast_results.append({
            "Metric": metric,
            "Comparison": contrast_name,
            "Estimate": estimate,
            "SE": standard_error,
            "Z_value": z_value,
            "P_value": p_value,
            "CI_lower": ci_lower,
            "CI_upper": ci_upper,
        })
planned_contrasts_df = pd.DataFrame(contrast_results)
planned_contrasts_df["Holm_P_value"] = np.nan
for metric, indices in planned_contrasts_df.groupby("Metric").groups.items():
    adjusted = holm_correction(planned_contrasts_df.loc[indices, "P_value"].to_numpy())
    planned_contrasts_df.loc[indices, "Holm_P_value"] = adjusted
planned_contrasts_df["Result_Summary"] = (
    planned_contrasts_df.apply(
        lambda row:
        f"{row['Estimate']:.4f} "
        f"(95% CI {row['CI_lower']:.4f} to "
        f"{row['CI_upper']:.4f}), "
        f"raw p={row['P_value']:.4f}, "
        f"Holm p={row['Holm_P_value']:.4f}",
        axis=1
    )
)
planned_contrasts_file = (OUTPUT_FOLDER / "Protocol_Comparisons_By_Cohort.csv")
planned_contrasts_df.to_csv(planned_contrasts_file, index=False)
print("\nPROTOCOL COMPARISONS BY COHORT AND METHOD")
print(planned_contrasts_df.to_string(index=False))
#plot
comparison_order = list(planned_contrasts.keys())
planned_contrasts_plot = OUTPUT_FOLDER / "Protocol_Comparisons_By_Cohort.png"
plot_planned_contrasts(data=planned_contrasts_df, metrics=METRICS, comparison_order=comparison_order, output_name=planned_contrasts_plot, xlabel="Estimated difference (95% confidence interval)")
print("\nProtocol comparison results saved to:", planned_contrasts_file)