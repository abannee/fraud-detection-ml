import os
import gc
import numpy as np
import pandas as pd

def reduce_mem_usage(df):
    """
    Iterate through all columns of a dataframe and modify the data type
    to reduce memory usage safely by checking numerical bounds.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of dataframe is {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # FIX: Use pandas native checker to safely isolate numerical columns
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Check if it's an integer type
            if pd.api.types.is_integer_dtype(col_type):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)  
            
            # Handle Float columns safely
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32) # Using float32 for stability
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            # Safely bypasses <StringDtype(na_value=nan)>, objects, and categories
            continue
            
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df



def load_and_merge_data(transaction_path, identity_path):
    """
    Loads the IEEE-CIS transaction and identity datasets and performs
    a defensive left merge on TransactionID before optimizing memory.
    """
    print("Loading transaction dataset...")
    trans = pd.read_csv(transaction_path)
    
    print("Loading identity dataset...")
    ids = pd.read_csv(identity_path)
    
    print("Executing defensive Left-Merge across TransactionID fields...")
    df = pd.merge(trans, ids, on='TransactionID', how='left')
    
    # Clean up unmerged individual dataframes from memory immediately
    del trans, ids
    gc.collect()
    
    # Apply memory reduction
    df = reduce_mem_usage(df)
    
    return df


def engineer_time_features(df):
    """Extracts structural calendar signals from the anonymous TransactionDT field."""
    # Reference anchor approximation (seconds from artificial baseline point)
    df['Hour'] = (df['TransactionDT'] // 3600) % 24
    df['DayOfWeek'] = (df['TransactionDT'] // (3600 * 24)) % 7
    return df