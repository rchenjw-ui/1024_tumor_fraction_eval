

"""
function defs
"""
import os
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri 
import pandas as pd
import numpy as np




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
       header = lines[0].strip().split('\t')[1:]  # sample names
       for line in lines[1:]:
           if not line.strip():
               continue
           parts = line.strip().split('\t')
           probe_id = parts[0]
           values = [float(x) if x not in ('', 'NA', 'NaN') else float('nan') for x in parts[1:]]
           signals[probe_id] = values
   return signals, header


