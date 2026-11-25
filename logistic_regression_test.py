import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path
try:
    from .data_extraction import load_pilot_samples, load_signals, sample_color
except ImportError:
    from data_extraction import load_pilot_samples, load_signals, sample_color

ROOT = Path(__file__).resolve().parent



def graph_metrics(results,name):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot vertical bars
    bars = ax.bar(results.keys(), results.values(), color='skyblue', edgecolor='black', width=0.6)
    ax.bar_label(bars, padding=3)
    # Customizations
    ax.set_title(f'{name} - metrics')
    ax.set_xlabel("Metrics")
    ax.set_ylabel('Count')
    plt.savefig(f"results/metrics_graph_{name}_probes.jpg")
    plt.close()

def display_preds(samples, preds, name):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(samples.keys(), preds, color=[sample_color(sample) for sample in samples])
    ax.set_ylabel("Methylation RB TFX")
    ax.set_title(f"Estimated TFX by RB methylation similarity - {name}")
    ax.tick_params(axis="x", labelrotation=35)
    ax.legend(handles=[
        Patch(facecolor="#00bfff", label="RB references / cell lines"),
        Patch(facecolor="#ffe600", label="Blood / uveal melanoma controls")
    ])
    fig.tight_layout()
    fig.savefig(ROOT / f"results/predicted_tumor_fraction_barplot_{name}.png")
    plt.close()





#______________________________________________________________________________________________________________________________________________________________________________


#data loading
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.pandas2ri as pandas2ri
import pandas as pd
from tqdm import tqdm

from tqdm.contrib import tzip


def rdata_to_dataframes_with_names(rdata_file):
    """
    Load RData file and convert R matrices/data frames to pandas DataFrames with proper index and columns.
    Returns a dict of variable name -> pandas DataFrame or original R object.
    """
    robjects.r['load'](rdata_file)
    var_names = list(robjects.r('ls()'))
    result = {}
    with localconverter(robjects.default_converter + pandas2ri.converter):
        for varname in var_names:
            r_obj = robjects.r[varname]
            try:
                # Try to get row and column names
                row_names = list(robjects.r(f'rownames({varname})'))
                col_names = list(robjects.r(f'colnames({varname})'))
                # Convert to DataFrame with assigned labels
                df = pd.DataFrame(robjects.conversion.rpy2py(r_obj), index=row_names, columns=col_names)
                result[varname] = df
            except Exception:
                # Fallback - return raw object if conversion fails
                result[varname] = r_obj
    return result


def extract_beta_columns(df, ref_probes):
    """
    Extract columns ending with '_AVG_Beta' from df, filter rows to ref_probes, 
    and organize as probes x samples DataFrame.
    
    Args:
        df (pd.DataFrame): Full dataframe loaded from RData.
        ref_probes (list or set): Probe IDs to keep.
        
    Returns:
        pd.DataFrame: probes x samples beta values.
    """
    # Select beta columns
    beta_columns = [c for c in df.columns if c.endswith('_AVG_Beta')]
    beta_df = df[beta_columns].copy()
    
    # Rename columns: remove suffix '_AVG_Beta'
    beta_df.columns = [c.replace('_AVG_Beta', '') for c in beta_df.columns]
    
    # Filter rows by probe IDs (assuming probe IDs are in index)
    filtered_beta_df = beta_df.loc[beta_df.index.intersection(ref_probes)]
    
    return filtered_beta_df


def beta_df_to_sample_probe_dict(beta_df):
    """
    Convert a DataFrame of beta values (probes x samples) into nested dictionary format.
    Args:
        beta_df (pd.DataFrame): DataFrame indexed by probes, columns are sample IDs.
    Returns:
        dict: {sample: {probe: beta_value}}
    """
    sample_dict = {}
    for sample in beta_df.columns:
        sample_dict[sample] = beta_df[sample].dropna().to_dict()
    return sample_dict




#gets beta values for embryo

def beta_values_embryonic(ref_probes, paths = ('embryo_data/GSE57362_Embryo_retinas_meth_signal.txt', 'embryo_data/GSE57362_Embryo_retinas_unmeth_signal.txt')):
    with open(paths[0], 'r') as f:
        meth_data = f.read().strip().split('\n')
    with open(paths[1], 'r') as f:
        unmeth_data = f.read().strip().split('\n')
    sample_ids = meth_data[0] 
    beta_values_embryo = {sample_id: {} for sample_id in sample_ids.split('\t')}  
    for line, line1 in tzip(meth_data[1:], unmeth_data[1:]):
        probe = line.split('\t')[0]
        if probe in ref_probes:
            data1 = line.split('\t')[1:]
            data2 = line1.split('\t')[1:]
            for sample_id, value1, value2 in zip(beta_values_embryo.keys(), data1, data2):
                beta = float(value1) / (float(value1) + float(value2)) if (float(value1) + float(value2)) != 0 else float('nan')
                beta_values_embryo[sample_id][probe] = beta
    return beta_values_embryo

#gets data for tfx

def beta_values_rb(ref_probes, path = 'tumor_data/GSE58783_Methylation_processed_data.RData'):
    rdata_dict = rdata_to_dataframes_with_names(path)
    beta_df = rdata_dict.get('processed_data')
    filtered_beta_df = extract_beta_columns(beta_df, ref_probes)
    beta_values_rb = beta_df_to_sample_probe_dict(filtered_beta_df)
    return beta_values_rb

#convert both into np 

def generate_matrices(beta_values_embryo, beta_values_rb, ref_probes):
    feature_matrix = []
    output_matrix = []
    for i in tqdm(beta_values_embryo.keys()):
        sample = beta_values_embryo[i]
        features = [sample[probe] for probe in ref_probes]
        if np.isnan(features).any() != True:
            feature_matrix.append(features)
            output_matrix.append(0)
        else:
            print("nan")

    for i in tqdm(beta_values_rb.keys()):
        sample = beta_values_rb[i]
        features = [sample[probe] for probe in ref_probes]
        if np.isnan(features).any() != True:
            feature_matrix.append(features)
            output_matrix.append(1)
        else:
            print("nan")

    feature_matrix = np.array(feature_matrix)
    output_matrix = np.array(output_matrix)
    return feature_matrix, output_matrix

def generate_ref_probes(name):

    if name == "probes":
        return ['cg10507988', 'cg08447324', 'cg23748923', 'cg15384383', 'cg02408775', 'cg15320905', 'cg12621203', 'cg15258447', 'cg06038470', 'cg26678920', 'cg27298164', 'cg13118906', 'cg21042627', 'cg02809746', 'cg26188698', 'cg26966245', 'cg02132702', 'cg14769786', 'cg00766482', 'cg19509829', 'cg13276570', 'cg16520288', 'cg27493151', 'cg27041381', 'cg08502239', 'cg05542681', 'cg16493531', 'cg11802666', 'cg26228266', 'cg08384314', 'cg17029019', 'cg27201775', 'cg20814095', 'cg16094026', 'cg21615583']
    path = ROOT / f"{name}_probe_beta_pairs.csv"
    df = pd.read_csv(path, usecols=['probe'])
    column_list = df["probe"].tolist()
    df = pd.read_csv(ROOT / "pilot_8samples_merged.csv", usecols = ["ProbeID"])
    probes = df["ProbeID"].tolist()
    column_list = [x for x in column_list if x in probes]
    return column_list


def coefficient_feature_importance(model, probes):
    """Return standardized logistic-regression coefficients ranked by absolute size."""
    coefficients = np.asarray(model.coef_)
    if coefficients.ndim != 2 or coefficients.shape[0] != 1:
        raise ValueError("Feature importance expects a binary logistic-regression model")
    coefficients = coefficients[0]
    if len(coefficients) != len(probes):
        raise ValueError("The number of coefficients must match the number of probes")

    return (
        pd.DataFrame({
            "probe": list(probes),
            "coefficient": coefficients,
            "importance": np.abs(coefficients),
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def main():
    results_dir = ROOT / "results"
    models_dir = ROOT / "models"
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    embryo_paths = (
        ROOT / "embryo_data/GSE57362_Embryo_retinas_meth_signal.txt",
        ROOT / "embryo_data/GSE57362_Embryo_retinas_Unmeth_signal.txt",
    )
    tumor_data = rdata_to_dataframes_with_names(str(ROOT / "tumor_data/GSE58783_Methylation_processed_data.RData"))

    for n_probes in range(10, 111, 20):
        name = f"probe_sets/top_{n_probes}_probes"
        probes = generate_ref_probes(name)
        beta_embryo = beta_values_embryonic(probes, paths=tuple(map(str, embryo_paths)))
        rb_beta_df = extract_beta_columns(tumor_data["processed_data"], probes)
        beta_rb = beta_df_to_sample_probe_dict(rb_beta_df)
        X, y = generate_matrices(beta_embryo, beta_rb, probes)

        model = sklearn.linear_model.LogisticRegression()
        scaler = StandardScaler()
        scaled_x = scaler.fit_transform(X)
        model.fit(scaled_x, y)

        probe_file = ROOT / f"{name}_probe_beta_pairs.csv"
        ref_probes = pd.read_csv(probe_file)["probe"].tolist()
        samples = load_pilot_samples(ROOT / "pilot_8samples_merged.csv", ref_probes)
        predictions = model.predict_proba(
            scaler.transform([samples[sample] for sample in samples])
        )[:, 1]
        display_preds(samples, predictions, f"{n_probes}_probes")

        importance = coefficient_feature_importance(model, ref_probes)
        importance.to_csv(results_dir / f"feature_importance_{n_probes}_probes.csv", index=False)

        model_data = {
            "probes": ref_probes,
            "scaler_var": scaler.var_.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "model_coef": dict(zip(ref_probes, model.coef_[0].tolist())),
            "model_int": model.intercept_.tolist(),
        }
        with (models_dir / f"model_and_scaler_{n_probes}.json").open("w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=4)


if __name__ == "__main__":
    main()
