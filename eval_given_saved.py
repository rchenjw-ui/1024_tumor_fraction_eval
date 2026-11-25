import json
from pathlib import Path
from functools import lru_cache
import sklearn
import numpy as np 
import pandas as pd
import mepylome
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
try:
    from .data_extraction import load_pilot_samples, sample_color
except ImportError:
    from data_extraction import load_pilot_samples, sample_color

ROOT = Path(__file__).resolve().parent
IDAT_PATHS = {
    "IDAT_23": ROOT / "methyl_data/sample_23/205648300012_R01C01_Red.idat",
    "IDAT_24": ROOT / "methyl_data/sample_24/205648300012_R03C01_Red.idat",
    "IDAT_25": ROOT / "methyl_data/sample_25/205648300012_R04C01_Red.idat",
}
SAMPLE_GROUPS = {
    "RB_022_OS_DX": "RB reference",
    "RB_028_OD_DX": "RB reference",
    "Blood_022": "Normal blood",
    "Blood_048": "Normal blood",
    "WERI1": "RB cell line",
    "Y79": "RB cell line",
    "UM_005": "Uveal melanoma",
    "UM_007": "Uveal melanoma",
}

"""
- Uses the saved model
- Generates synthetic data for eval
    - Using in silico dilution mixtures (linear known combo), we create simulated data
    - The idea is to see how the model handles mixing of data
- We graph the output of the model; as it's a logistic regression, it handles noise well but at the cost of some nonlinearity
- Results: model underpredicts at low, over-predicts at high (sigmoid function fit)

Conclusion: 
- it is known model avoids "impossible" results and handles noise well (given UM and cell line eval)
- However, model's nonlinearity may be a problem that limits effectiveness
- We will need to run an evaluation using CNV data, on synthetic data 

Additional:
- Proposed upgrades: 
    - Isotonic regression calibration (if enough data)
    - specialized Beta Regression 
- Proposed test methods: 
    - Test direction vs chemotherapy time
    - check correlation with other possible metrics 

    links to see 
    https://www.biorxiv.org/content/10.1101/2025.11.27.688590v1
    https://www.nature.com/articles/s41467-022-33248-2
"""


def display_preds(samples, preds, name):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(samples.keys(), preds, color=[sample_color(sample) for sample in samples])
    ax.set_ylabel("Methylation RB TFX")
    ax.set_title(f"Estimated TFX by RB methylation similarity - {name}")
    ax.tick_params(axis="x", labelrotation=35)
    ax.legend(handles=[
        Patch(facecolor="#00bfff", label="RB references / cell lines"),
        Patch(facecolor="#ffe600", label="Blood / uveal melanoma controls"),
        Patch(facecolor="#c96eac", label="Unlabeled patient IDATs"),
    ])
    fig.tight_layout()
    fig.savefig(ROOT / f"results/patient_data_{name}_tumor_fraction_barplot.png")
    plt.close()

#lin's CCC: used to see how closely matched to real across synthetic
def lins_ccc(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    r = np.corrcoef(y_true, y_pred)[0][1]
    if np.isnan(r):
        r = 0.0
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true, ddof=1), np.var(y_pred, ddof=1)
    sd_true, sd_pred = np.sqrt(var_true), np.sqrt(var_pred)
    
    numerator = 2 * r * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    return numerator / denominator if denominator != 0 else 0.0

def evaluate_stratified_ccc(y_true, y_pred):
    """Computes Lin's CCC across three clinical tumor fraction zones."""
    zones = {
        "Low Zone (<10% TF)": (y_true < 0.10),
        "Med Zone (10%-70% TF)": (y_true >= 0.10) & (y_true <= 0.70),
        "High Zone (>70% TF)": (y_true > 0.70),
        "Global Performance": np.ones_like(y_true, dtype=bool)
    }
    metrics = {}
    for name, mask in zones.items():
        if np.sum(mask) >= 2:
            metrics[name] = float(lins_ccc(y_true[mask], y_pred[mask]))
        else:
            metrics[name] = "Insufficient data points"
    return metrics


def reconstruct_model(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    probes = data["probes"]
    params = []
    for i in probes:
        params.append(data["model_coef"][i])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.mean_ = np.array(data["scaler_mean"])
    scaler.var_ = np.array(data["scaler_var"])
    scaler.scale_ = np.sqrt(scaler.var_)
    scaler.n_samples_seen_ = 1

    model = sklearn.linear_model.LogisticRegression()
    model.coef_ = np.array(params).reshape(1, -1)
    model.intercept_ = data["model_int"]
    model.classes_ = np.array([0, 1]) 
    return probes, model, scaler



def create_synthetic_mixtures(rb_vals, n_probes):
    probes_df = pd.read_csv(ROOT / f'probe_sets/top_{n_probes}_probes_probe_beta_pairs.csv')
    probes = probes_df["probe"].tolist()
    wt_vals = probes_df["beta_wt"].tolist()
    if len(rb_vals) != len(probes):
        raise ValueError(f"Expected {len(probes)} tumor beta values; received {len(rb_vals)}")
    
    # 101 points provides fine increments (0.00, 0.01, ..., 1.00)
    true_tfs = np.linspace(0.0, 1.0, 101)
    synthetic_matrix = []
    
    for tf in true_tfs:
        tfx = [(tf * rb_vals[x]) + (wt_vals[x] * (1.0 - tf)) for x in range(len(probes))]
        synthetic_matrix.append(tfx)
        
    return true_tfs, np.array(synthetic_matrix)


@lru_cache(maxsize=3)
def _read_idat(path):
    return mepylome.MethylData(file=path)


def load_patient_idat(path, probes):
    path = Path(path).resolve()
    methyl_data = _read_idat(str(path))
    methylated_signals   = methyl_data.methylated_df.loc[probes]
    unmethylated_signals = methyl_data.unmethylated_df.loc[probes]
    suffix = path.stem.removesuffix("_Red")
    if suffix not in methylated_signals.columns:
        raise KeyError(f"IDAT sample {suffix!r} is not present in {path}")
    probes_betas = {
        x:float(methylated_signals.loc[x][suffix] / (methylated_signals.loc[x][suffix] + unmethylated_signals.loc[x][suffix] + 100))
        for x in probes
        }
    beta_vals = list(probes_betas.values())
    return beta_vals
def load_sample(probes, path=None):
    pilot_path = Path(path) if path is not None else ROOT / "pilot_8samples_merged.csv"
    samples = load_pilot_samples(pilot_path, probes)
    for sample_name, idat_path in IDAT_PATHS.items():
        samples[sample_name] = load_patient_idat(idat_path, probes)
    return samples


def _sample_group(sample):
    if sample in SAMPLE_GROUPS:
        return SAMPLE_GROUPS[sample]
    if sample.startswith("IDAT_"):
        return "Unlabeled patient IDAT"
    return "Unclassified"


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator != 0 else np.nan


def evaluate_saved_models():
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    score_rows = []
    metric_rows = []

    positive_samples = ["RB_022_OS_DX", "RB_028_OD_DX", "WERI1", "Y79"]
    normal_samples = ["Blood_022", "Blood_048"]
    um_samples = ["UM_005", "UM_007"]
    non_rb_samples = normal_samples + um_samples

    model_paths = sorted(
        (ROOT / "models").glob("model_and_scaler_*.json"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    for model_path in model_paths:
        n_probes = int(model_path.stem.rsplit("_", 1)[1])
        probes, model, scaler = reconstruct_model(model_path)
        samples = load_sample(probes)
        sample_names = list(samples)
        x = scaler.transform([samples[sample] for sample in sample_names])
        predictions = model.predict_proba(x)[:, 1]
        scores = dict(zip(sample_names, predictions))

        for sample, score in scores.items():
            score_rows.append({
                "model_probes": n_probes,
                "sample": sample,
                "group": _sample_group(sample),
                "tumor_fraction": float(score),
            })

        positive_scores = [scores[sample] for sample in positive_samples]
        normal_scores = [scores[sample] for sample in normal_samples]
        um_scores = [scores[sample] for sample in um_samples]
        non_rb_scores = [scores[sample] for sample in non_rb_samples]
        mean_rb = float(np.mean([scores[sample] for sample in positive_samples[:2]]))
        metrics = {
            "model_probes": n_probes,
            "min_RB_or_cellline_minus_max_non_RB": float(min(positive_scores) - max(non_rb_scores)),
            "min_RB_or_cellline_minus_max_blood": float(min(positive_scores) - max(normal_scores)),
            "mean_RB_or_cellline_tfx": float(np.mean(positive_scores)),
            "mean_normal_blood_tfx": float(np.mean(normal_scores)),
            "mean_RB_or_cellline_minus_normal_blood": float(np.mean(positive_scores) - np.mean(normal_scores)),
            "mean_UM_to_mean_RB_tfx_ratio": _safe_ratio(float(np.mean(um_scores)), mean_rb),
        }

        for sample in IDAT_PATHS:
            true_tfs, synthetic = create_synthetic_mixtures(samples[sample], n_probes)
            synthetic_predictions = model.predict_proba(scaler.transform(synthetic))[:, 1]
            metrics[f"{sample}_synthetic_Lin_CCC"] = float(lins_ccc(true_tfs, synthetic_predictions))

        metric_rows.append(metrics)
        display_preds(samples, predictions, f"{n_probes}_probes")

    score_df = pd.DataFrame(score_rows)
    score_matrix = score_df.pivot(
        index="model_probes",
        columns="sample",
        values="tumor_fraction",
    ).reset_index()
    metrics_df = pd.DataFrame(metric_rows)
    score_df.to_csv(results_dir / "sample_scores_long.csv", index=False)
    score_matrix.to_csv(results_dir / "sample_scores_matrix.csv", index=False)
    metrics_df.to_csv(results_dir / "evaluation_metrics_matrix.csv", index=False)
    print("Tumor fractions:")
    print(score_matrix.to_string(index=False))
    print("Evaluation metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    evaluate_saved_models()
