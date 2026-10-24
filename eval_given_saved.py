import json
from pathlib import Path
import sklearn
import numpy as np 
import pandas as pd
import mepylome
import matplotlib.pyplot as plt

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

def display_preds(preds, name):
    plt.figure(figsize=(12, 7))
    plt.bar(samples.keys(), preds, color=["#00bfff", "#00bfff", "#ffe600", "#ffe600","#00bfff", "#00bfff", "#ffe600", "#ffe600", "#00bfff", "#00bfff"])
    plt.ylabel(f"Predicted Tumor Fraction - {name}")
    plt.title(f"Predicted Tumor Fraction for Each Sample - {name}")  
    plt.show()
    plt.savefig(f"predicted_tumor_fraction_barplot_{name}.png")
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
    model.coef_ = np.array(params)
    model.intercept_ = data["model_int"]
    model.classes_ = np.array([0, 1]) 
    return probes, model, scaler



def create_synthetic_mixtures(rb_vals):
    probes_df = pd.read_csv('top_50_probes_probe_beta_pairs.csv')
    probes = probes_df["probe"].tolist()
    wt_vals = probes_df["beta_wt"].tolist()
    
    # 101 points provides fine increments (0.00, 0.01, ..., 1.00)
    true_tfs = np.linspace(0.0, 1.0, 101)
    synthetic_matrix = []
    
    for tf in true_tfs:
        tfx = [(tf * rb_vals[x]) + (wt_vals[x] * (1.0 - tf)) for x in range(len(probes))]
        synthetic_matrix.append(tfx)
        
    return true_tfs, np.array(synthetic_matrix)



"""
DATA LOADING SECTION
Load betas of a sample
"""
#Load IDAT and then get beta values
methyl_data = mepylome.MethylData(file='methyl_data/205648300012_R03C01_Red.idat')

probes, model, scaler = reconstruct_model("model_and_scaler_50.json")

methylated_signals   = methyl_data.methylated_df.loc[probes]
unmethylated_signals = methyl_data.unmethylated_df.loc[probes]

probes_betas = {
    x:float(methylated_signals.loc[x]["205648300012_R03C01"]/(methylated_signals.loc[x]["205648300012_R03C01"]+unmethylated_signals.loc[x]["205648300012_R03C01"]+100))
    for x in probes
    }

beta_vals = list(probes_betas.values())
rb_vals = list(pd.read_csv('top_50_probes_probe_beta_pairs.csv')["beta_rb"]) #synthetic retinoblastoma case

samples = {}
with open("pilot_8samples_merged.csv", 'r') as file:
    #create dictionary, sample_wise probe addition
    file = file.read().splitlines()
    labels = file[0].split(",")
    for i in file[1:]:
        line = i.split(',')
        if line[0] in probes:
            for j in range(1,len(line)):
                if labels[j] in samples:
                    samples[labels[j]].append(float(line[j]))
                else:
                    samples[labels[j]] = [float(line[j])]
samples["IDAT_23"] = beta_vals
samples["RB_mean"] = rb_vals

for i in samples:
    print(i)
    print(len(samples[i]))

"""
below is evaluation code across the samples, plus gap between cancer and wt
"""

#predict betas
print(model.predict_proba(scaler.transform([beta_vals]))[0][1]) 
print(model.predict_proba(scaler.transform([rb_vals]))[0][1])

#begin evaluating betas

preds = [x[1] for x in model.predict_proba(scaler.transform([samples[i] for i in samples]))]
display_preds(preds, f"logistic regression results")

"""
below is synthetic eval code for both
"""

print("First eval on average RB, then beta of target ")

for sample, name in zip((rb_vals, beta_vals), ["IDAT_23", "RB_mean"]):
    
    print(f"Synthetic Evals - {name}")

    true_tfs, synthetic = create_synthetic_mixtures(sample)

    preds = model.predict_proba(scaler.transform(synthetic))[:, 1]
    #we have tumor fractions now

    ccc = evaluate_stratified_ccc(true_tfs, preds)
    print(f"Concordance Corelation Coefficients")
    for i in ccc:
        print(f"{i} {ccc[i]}")

    #plot

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot vertical bars
    bars = ax.bar(true_tfs, list(preds), color='skyblue', edgecolor='black', width=0.01)
    # Customizations
    ax.set_title(f'synthetic tfx vs predicted - {name}')
    plt.plot(true_tfs, true_tfs, color='green', label=f'Reference line')

    ax.set_xlabel("synthetic tfx")
    ax.set_ylabel('predicted tfx')

    fig.show()
    plt.show()
    print("\n")
