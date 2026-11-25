from data_extraction import rdata_to_dataframe, extract_beta_columns, load_signals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import t
import scipy.stats
from tqdm import tqdm
import warnings
import csv 

# This script will load the methylation and unmethylation signal files, compute beta values for each probe and sample, and print out example beta values for the first few probes.

 
def compute_beta_dataframe(meth_signals, unmeth_signals, sample_names):
   """
   Return beta DataFrame (probes x samples) for all probes and samples.
   Pads out singleton or missing signal arrays with np.nan.
   """
   probes = list(meth_signals.keys())
   num_samples = len(sample_names)
   beta_matrix = np.full((len(probes), num_samples), np.nan)
   for i, probe in tqdm(enumerate(probes), desc="computing betas", unit="probes"):
       meth = np.asarray(meth_signals[probe])
       unmeth = np.asarray(unmeth_signals.get(probe, [np.nan]*num_samples))
       # Pad or truncate to num_samples length
       if len(meth) < num_samples:
           meth = np.concatenate([meth, np.full(num_samples - len(meth), np.nan)])
       elif len(meth) > num_samples:
           meth = meth[:num_samples]
       if len(unmeth) < num_samples:
           unmeth = np.concatenate([unmeth, np.full(num_samples - len(unmeth), np.nan)])
       elif len(unmeth) > num_samples:
           unmeth = unmeth[:num_samples]
       with np.errstate(divide='ignore', invalid='ignore'):
           beta = np.where((meth + unmeth) != 0, meth / (meth + unmeth), np.nan)
       beta_matrix[i, :] = beta
   return pd.DataFrame(beta_matrix, index=probes, columns=sample_names)




#test: deltas, variance in rb + variance in retina, n in rb > 0.4, n in retina_beta


#will also 



def get_probes_deltas(rb_beta_df, retina_beta_df, sample_probes):
    """Compute delta beta (RB - Retina) for probes present in both DataFrames."""
    common_probes = set(rb_beta_df.index) & set(retina_beta_df.index) & set(sample_probes)
    deltas = {}
    for probe in tqdm(common_probes, desc = "analyzing probe deltas", unit= "probe"):
        rb_values = rb_beta_df.loc[probe].values
        retina_values = retina_beta_df.loc[probe].values
        # Compute mean beta across samples for each probe
        rb_mean = np.nanmean(rb_values)
        retina_mean = np.nanmean(retina_values)
        if not (np.isnan(rb_mean) or np.isnan(retina_mean)):
            deltas[probe] = rb_mean - retina_mean
        else:
            deltas[probe] = np.nan
    return deltas




def get_probe_data(rb_beta_df, retina_beta_df, ref_probes):
    """Compute delta beta (RB - Retina) for probes present in both DataFrames."""
    common_probes = set(rb_beta_df.index) & set(retina_beta_df.index) 
    deltas = {}
    for probe in tqdm(ref_probes, desc = "analyzing probes", unit= "probe"):
        

        rb_values = np.array(rb_beta_df.loc[probe].values)
        rb_values = rb_values[~np.isnan(rb_values)]
        retina_values = np.array(retina_beta_df.loc[probe].values)
        retina_values = retina_values[~np.isnan(retina_values)]
        deltas[probe] = {}

        rb_mean = np.nanmean(rb_values)
        retina_mean = np.nanmean(retina_values)
        if not (np.isnan(rb_mean) or np.isnan(retina_mean)):
            deltas[probe]["delta"] = rb_mean - retina_mean
        else:
            deltas[probe]["delta"] = np.nan

        #compute variance total
        rb_variance = np.nanvar(rb_values)
        retina_variance = np.nanvar(retina_values)
        #use formula for coefficient of variance
        if not (np.isnan(rb_variance) or np.isnan(retina_variance)):
            grand_mean = (len(rb_values)*rb_mean + len(retina_values)*retina_mean)/(len(rb_values)+len(retina_values))
            total_variance = ( (len(rb_values) - 1)*rb_variance + (len(retina_values)-1)*retina_variance)/ (len(rb_values)+len(retina_values)-2) 
            deltas[probe]["variance"] = total_variance/(grand_mean)
        else:
            deltas[probe]["variance"] = np.nan
    low_var = [x for x in deltas if deltas[x]["variance"] < 0.03]
    low_var = sorted(low_var, key = lambda x: deltas[x]["delta"], reverse = True)
    return low_var


def write_csvs(probes, names):

    for items, title in zip(probes, names):
        data = []
        for i in tqdm(items):
            rb_values = rb_beta_df.loc[i].values
            retina_values = retina_beta_df.loc[i].values
            data.append({"probe":i, "beta_rb":np.nanmean(rb_values), "beta_wt":np.nanmean(retina_values)})
        with open(f"{title}_probe_beta_pairs.csv", mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=["probe", "beta_rb", "beta_wt"])
            writer.writeheader()
            writer.writerows(data)

if __name__ == "__main__":
    #load retina beta DataFrame
    CUTOFF = 0.4 


    meth_signals, sample_names = load_signals("embryo_data/GSE57362_Embryo_retinas_meth_signal.txt")
    unmeth_signals, _ = load_signals("embryo_data/GSE57362_Embryo_retinas_Unmeth_signal.txt")
    retina_beta_df = compute_beta_dataframe(meth_signals, unmeth_signals, sample_names) 

    # Load tumor beta DataFrame (existing code - output is probe x tumor-sample)
    data = rdata_to_dataframe("tumor_data/GSE58783_Methylation_processed_data.RData")['processed_data']
    rb_beta_df = extract_beta_columns(data)

    print("Found betas in RB/WT!")
    df = pd.read_csv("pilot_8samples_merged.csv", usecols = ["ProbeID"])
    sample_probes = df["ProbeID"].tolist()

    deltas = get_probes_deltas(rb_beta_df, retina_beta_df, sample_probes)
    probes_over_cutoff = [x for x in deltas if abs(deltas[x]) > CUTOFF]

    print(len(probes_over_cutoff))

    print("getting probe data...")
    probe_info = get_probe_data(rb_beta_df, retina_beta_df, probes_over_cutoff)
    print("Probe data retrieved!")

    print(len(probe_info))


    #matching_keys = [k for k, v in my_dict.items() if v < constraint]

    #need t100
    probes = []
    names = []
    for i in range (1,15, 2):
        probes_upper = probe_info[0:5*i]
        probes_lower = probe_info[-5*i:]
        probes.append(probes_upper+probes_lower)
        names.append(f"probe_sets/top_{i*10}_probes")
    write_csvs(probes,names)