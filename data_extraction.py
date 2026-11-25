

"""
function defs
"""
import os
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri 
import pandas as pd
import numpy as np

SAMPLE_COLORS = {
   "RB reference": "#00bfff",
   "Normal blood": "#ffe600",
   "RB cell line": "#00bfff",
   "Uveal melanoma": "#ffe600",
   "Unlabeled patient IDAT": "#c96eac",
}


def sample_group(sample):
   if sample in ("RB_022_OS_DX", "RB_028_OD_DX"):
       return "RB reference"
   if sample in ("Blood_022", "Blood_048"):
       return "Normal blood"
   if sample in ("WERI1", "Y79"):
       return "RB cell line"
   if sample in ("UM_005", "UM_007"):
       return "Uveal melanoma"
   if sample.startswith("IDAT_"):
       return "Unlabeled patient IDAT"
   return "Unclassified"


def sample_color(sample):
   return SAMPLE_COLORS.get(sample_group(sample), "#808080")


def rdata_to_dataframe(rdata_path):
   """Load an RData file and return a dict of variable names to pandas DataFrames with probe IDs and sample IDs as index/columns."""
   result = {}
   try:
       robjects.r(f'load("{rdata_path}")')
       var_names = list(robjects.r('ls()'))
       for varname in var_names:
           r_obj = robjects.r[varname]
           try:
               df = pandas2ri.rpy2py(r_obj)
               # Try to extract row names (probe IDs) from the R object
               try:
                   row_names = list(robjects.r(f'rownames({varname})'))
                   if row_names and df is not None:
                       if isinstance(df, pd.DataFrame):
                           df.index = row_names
                       elif hasattr(df, 'shape') and len(df.shape) == 2:
                           df = pd.DataFrame(df, index=row_names)
               except Exception:
                   pass  # Row names not available, continue without them
              
               # Try to extract column names (sample IDs) from the R object
               try:
                   col_names = list(robjects.r(f'colnames({varname})'))
                   if col_names and df is not None and isinstance(df, pd.DataFrame):
                       df.columns = col_names
               except Exception:
                   pass  # Column names not available, continue without them
              
               result[varname] = df
           except Exception as e:
               result[varname] = None
               print(f"Could not convert {varname} to DataFrame: {e}")
   except Exception as e:
       print(f"Error loading RData file: {e}")
   return result


def extract_beta_columns(df):
   """Extract only the AVG_Beta columns from a processed_data DataFrame."""
   beta_cols = [col for col in df.columns if 'AVG_Beta' in col]
   return df[beta_cols]


def load_signals(file_path):
   """Load signal file into dict: {ID: [values]}, returns (signals, sample_names)"""
   signals = {}
   with open(file_path, 'r') as f:
       lines = f.read().strip().split('\n')
       header = lines[0].strip().split('\t')  # sample names
       for line in lines[1:]:
           if not line.strip():
               continue
           parts = line.strip().split('\t')
           probe_id = parts[0]
           values = [float(x) if x not in ('', 'NA', 'NaN') else float('nan') for x in parts[1:]]
           signals[probe_id] = values
   return signals, header


def load_pilot_samples(file_path, probes):
   """Load pilot samples with values ordered exactly as the requested probes."""
   df = pd.read_csv(file_path)
   if 'ProbeID' not in df.columns:
       raise ValueError(f"{file_path} must contain a ProbeID column")
   if df['ProbeID'].duplicated().any():
       raise ValueError(f"{file_path} contains duplicate ProbeID values")

   df = df.set_index('ProbeID')
   missing_probes = [probe for probe in probes if probe not in df.index]
   if missing_probes:
       raise ValueError(f"{file_path} is missing {len(missing_probes)} requested probes")

   return {
       sample: df.loc[list(probes), sample].astype(float).tolist()
       for sample in df.columns
   }
