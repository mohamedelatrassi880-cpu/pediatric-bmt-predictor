import pandas as pd
import numpy as np
import sys
import os

# Ensure the test file can find the 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import optimize_memory

def test_optimize_memory():
    """
    Tests if the optimize_memory function successfully reduces memory footprint
    by downcasting a float64 column to float32.
    """
    # 1. Create fake data using large 64-bit numbers
    df = pd.DataFrame({'test_col': np.array([1.5, 2.5, 3.5], dtype=np.float64)})
    start_mem = df.memory_usage().sum()
    
    # 2. Run your optimization function
    df_optimized = optimize_memory(df)
    end_mem = df_optimized.memory_usage().sum()
    
    # 3. Assert (Verify) that it worked
    assert end_mem < start_mem, "Memory was not reduced!"
    assert df_optimized['test_col'].dtype == np.float32, "Column was not downcasted to float32!"

def test_optimize_memory_multiple_types():
    """Satisfies: Run multiple tests"""
    # Test that the function correctly handles a mix of text and numbers without crashing
    df = pd.DataFrame({
        'text_col': ['A', 'B', 'C'],
        'int_col': [1, 2, 3]
    })
    df['int_col'] = df['int_col'].astype(np.int64)
    
    optimized_df = optimize_memory(df)
    
    # Text should remain untouched, int64 should be compressed to int8 or int32
    assert optimized_df['text_col'].dtype == object
    assert optimized_df['int_col'].dtype != np.int64

def test_optimize_memory_extremes():
    """Satisfies: Add new extremes test"""
    # Test how the system handles extreme outliers (massive numbers)
    # 1e100 is so large it cannot be safely downcasted to a smaller float type
    df = pd.DataFrame({'extreme_col': [1e100, -1e100, 0.0]})
    df['extreme_col'] = df['extreme_col'].astype(np.float64)
    
    optimized_df = optimize_memory(df)
    
    # The function should recognize the extreme values and safely leave it as float64
    assert optimized_df['extreme_col'].dtype == np.float64