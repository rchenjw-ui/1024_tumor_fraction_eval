from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from .eval_given_saved import IDAT_PATHS, load_patient_idat, reconstruct_model
except ImportError:
    from eval_given_saved import IDAT_PATHS, load_patient_idat, reconstruct_model

ROOT = Path(__file__).resolve().parent
SAMPLE_TO_IDAT = {
    "CHLA_RR_023_OS_DX": "IDAT_23",
    "CHLA_RR_024_OD_DX": "IDAT_24",
    "CHLA_RR_025_OS_DX": "IDAT_25",
}


def plot_method_comparison(comparison, output_path):
    methylation_columns = sorted(
        (column for column in comparison.columns if column.startswith("tumor_fraction_methylation_")),
        key=lambda column: int(column.split("_")[3]),
    )
    columns = ["cnv_estimate_bounded_fraction", *methylation_columns]
    matrix = comparison.set_index("sample")[columns].to_numpy(dtype=float)
    labels = [
        "CNA\nestimate",
        *[f"RB TFX\n{column.split('_')[3]} probes" for column in methylation_columns],
    ]
    samples = [sample.removeprefix("CHLA_RR_") for sample in comparison["sample"]]

    fig, ax = plt.subplots(figsize=(13, 4))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(samples)), labels=samples)
    ax.set_title("Experimental TFX metrics: CNA and RB methylation TFX")
    ax.set_xlabel("Methylation fractions")

    midpoint = len(labels) - len(methylation_columns) - 0.5
    ax.axvline(midpoint, color="black", linewidth=1.5)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text_color = "white" if value >= 0.65 else "black"
            ax.text(column, row, f"{value:.3f}", ha="center", va="center", color=text_color)

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Metric value (0–1)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def TFX_estimator(df, sample, threshold=0.05):
    ratios = pd.to_numeric(df[sample], errors="coerce").dropna().to_numpy()
    if ratios.size == 0:
        raise ValueError(f"No CNA ratios available for {sample}")

    baseline = float(np.median(ratios))
    deletions = ratios[ratios < baseline - threshold]
    gains = ratios[ratios > baseline + threshold]
    tf_estimates = []

    if deletions.size:
        tf_estimates.append(2 * (baseline - float(np.mean(deletions))))
    if gains.size:
        tf_estimates.append(2 * (float(np.mean(gains)) - baseline))

    estimated_tf = float(np.median(tf_estimates)) if tf_estimates else 0.0
    return baseline, estimated_tf


def find_altered_regions(df, sample, baseline, threshold=0.05):
    """List contiguous altered rows; coordinates are the first/last observed positions."""
    regions = []
    autosomal = df.loc[df["chrom"].between(1, 22)].reset_index(drop=True)
    ratios = pd.to_numeric(autosomal[sample], errors="coerce").to_numpy()
    altered = np.isfinite(ratios) & (np.abs(ratios - baseline) > threshold)

    start = None
    for i in range(len(autosomal) + 1):
        if i < len(autosomal) and altered[i]:
            same_run = (
                start is not None
                and autosomal.loc[i, "chrom"] == autosomal.loc[i - 1, "chrom"]
                and np.isclose(ratios[i], ratios[i - 1], rtol=0, atol=1e-8)
            )
            if same_run:
                continue

        if start is not None:
            end = i - 1
            start_row = autosomal.loc[start]
            end_row = autosomal.loc[end]
            ratio = float(np.mean(ratios[start:end + 1]))
            regions.append({
                "sample": sample,
                "chrom": int(start_row["chrom"]),
                "start_position_observed": int(start_row["chrompos"]),
                "end_position_observed": int(end_row["chrompos"]),
                "row_count": end - start + 1,
                "mean_ratio": ratio,
                "baseline": baseline,
                "alteration": "gain" if ratio > baseline else "loss",
            })
            start = None

        if i < len(autosomal) and altered[i]:
            start = i

    return regions


def methylation_rb_scores():
    scores = {}
    models_dir = ROOT / "models"
    model_paths = sorted(
        models_dir.glob("model_and_scaler_*.json"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    for model_path in model_paths:
        n_probes = int(model_path.stem.rsplit("_", 1)[1])
        probes, model, scaler = reconstruct_model(model_path)
        for sample_name, idat_path in IDAT_PATHS.items():
            beta_values = load_patient_idat(idat_path, probes)
            score = model.predict_proba(scaler.transform([beta_values]))[0, 1]
            scores[(sample_name, n_probes)] = float(score)
    return scores


def main():
    data = pd.read_excel(ROOT / "Segmented_CNA_RyanTFX.xlsx")
    data = data.loc[data["chrom"].between(1, 22)].reset_index(drop=True)
    results = {}
    alterations = []

    for sample in SAMPLE_TO_IDAT:
        baseline, raw_estimate = TFX_estimator(data, sample)
        results[sample] = {
            "baseline": baseline,
            "cnv_estimate_raw": raw_estimate,
            "cnv_estimate_bounded": float(np.clip(raw_estimate, 0.0, 1.0)),
        }
        alterations.extend(find_altered_regions(data, sample, baseline))

    methylation_scores = methylation_rb_scores()
    comparison_rows = []
    for cnv_sample, idat_sample in SAMPLE_TO_IDAT.items():
        row = {
            "sample": cnv_sample,
            "cnv_baseline_ratio": results[cnv_sample]["baseline"],
            "cnv_estimate_raw_fraction": results[cnv_sample]["cnv_estimate_raw"],
            "cnv_estimate_bounded_fraction": results[cnv_sample]["cnv_estimate_bounded"],
        }
        for (sample_name, n_probes), score in methylation_scores.items():
            if sample_name == idat_sample:
                row[f"tumor_fraction_methylation_{n_probes}_probes"] = score
        comparison_rows.append(row)

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    comparison = pd.DataFrame(comparison_rows)
    regions = pd.DataFrame(alterations, columns=[
        "sample",
        "chrom",
        "start_position_observed",
        "end_position_observed",
        "row_count",
        "mean_ratio",
        "baseline",
        "alteration",
    ])
    comparison.to_csv(results_dir / "cnv_methylation_comparison.csv", index=False)
    plot_method_comparison(comparison, results_dir / "method_comparison_matrix.png")
    regions.to_csv(results_dir / "cnv_altered_regions.csv", index=False)

    print("CNA estimates vs methylation tumor fractions:")
    print(comparison.to_string(index=False))
    print(f"\nMethod comparison matrix written to {results_dir / 'method_comparison_matrix.png'}")
    print(f"\nAltered genomic regions written to {results_dir / 'cnv_altered_regions.csv'}")
    print(regions.to_string(index=False))


if __name__ == "__main__":
    main()
