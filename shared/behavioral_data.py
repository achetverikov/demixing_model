"""Behavioral-data cleaning helpers for fitting and rescoring."""
import numpy as np

def filter_data_for_fitting(data, feat_diff_col=None, bias_col=None, verbose=True,
                            min_diss=4.0, max_diss=180.0):
    """
    Filter and clean data for GMM fitting by removing invalid values and applying range constraints.

    Handles both numpy arrays and pandas DataFrames:
    - For 2-column arrays: assumes first column is feat_diff, second is bias
    - For DataFrames: uses specified column names

    Args:
        data: Either a 2-column numpy array or pandas DataFrame
        feat_diff_col: Column name for feature differences (required for DataFrame)
        bias_col: Column name for bias values (required for DataFrame)
        verbose: Whether to print filtering statistics
        min_diss: Dissimilarity floor in the INPUT (raw, pre-scaling) space; values
            below it are clamped up. max_diss: dissimilarity ceiling; values above it
            are dropped. Defaults (4, 180) preserve the legacy raw-space clamp. The
            demixing fitter passes period/scale-aware bounds (feat_diff_range / scale)
            so the effective clamp is the model-space grid range for every dataset —
            see HISTORY.md (2026-06 to 2026-09 audit closure).

    Returns:
        Cleaned data in the same format as input
    """
    import pandas as pd
    
    is_dataframe = isinstance(data, pd.DataFrame)
    
    if is_dataframe:
        if feat_diff_col is None or bias_col is None:
            raise ValueError("feat_diff_col and bias_col must be specified for DataFrames")
        original_len = len(data)
        clean_data = data.copy()
        feat_diff_vals = clean_data[feat_diff_col]
        bias_vals = clean_data[bias_col]
    else:
        # Assume 2-column array: first column feat_diff, second bias
        if data.shape[1] != 2:
            raise ValueError("Array must have exactly 2 columns")
        original_len = len(data)
        clean_data = data.copy()
        feat_diff_vals = clean_data[:, 0]
        bias_vals = clean_data[:, 1]
    
    # Check for NaN values
    nan_mask = np.isnan(feat_diff_vals) | np.isnan(bias_vals)
    nan_count = nan_mask.sum()
    
    if nan_count > 0 and verbose:
        print(f"Found {nan_count} NaN values")
        
    # Remove NaN values
    if is_dataframe:
        clean_data = clean_data[~nan_mask].copy().reset_index(drop=True)
        feat_diff_vals = clean_data[feat_diff_col]
        bias_vals = clean_data[bias_col]
    else:
        clean_data = clean_data[~nan_mask]
        feat_diff_vals = clean_data[:, 0]
        bias_vals = clean_data[:, 1]
    
    # Check for infinite values
    inf_mask = ~(np.isfinite(feat_diff_vals) & np.isfinite(bias_vals))
    inf_count = inf_mask.sum()
    
    if inf_count > 0 and verbose:
        print(f"Found {inf_count} infinite values")
        
    # Remove infinite values
    if is_dataframe:
        clean_data = clean_data[~inf_mask].copy().reset_index(drop=True)
        feat_diff_vals = clean_data[feat_diff_col]
        bias_vals = clean_data[bias_col]
    else:
        clean_data = clean_data[~inf_mask]
        feat_diff_vals = clean_data[:, 0]
        bias_vals = clean_data[:, 1]
    
    # Save original values before any modifications (after all row filtering)
    if is_dataframe:
        original_feat_diff = clean_data[feat_diff_col].copy()
    else:
        original_feat_diff = clean_data[:, 0].copy()
    
    # Check feat_diff range constraints
    feat_below_min = feat_diff_vals < min_diss
    feat_above_max = feat_diff_vals > max_diss
    feat_below_count = feat_below_min.sum()
    feat_above_count = feat_above_max.sum()

    if feat_below_count > 0 and verbose:
        print(f"Found {feat_below_count} feat_diff values < {min_diss:g} - will clamp to {min_diss:g}")
        below_vals = feat_diff_vals[feat_below_min]
        print(f"  Values < {min_diss:g}: min={below_vals.min():.2f}, examples={list(below_vals.head(3) if hasattr(below_vals, 'head') else below_vals[:3].round(2))}")

    if feat_above_count > 0 and verbose:
        print(f"Found {feat_above_count} feat_diff values > {max_diss:g} - will remove")
        above_vals = feat_diff_vals[feat_above_max]
        print(f"  Values > {max_diss:g}: max={above_vals.max():.2f}, examples={list(above_vals.head(3) if hasattr(above_vals, 'head') else above_vals[:3].round(2))}")
        
    # Save clamping examples before filtering removes them
    clamp_examples_orig = None
    clamp_examples_new = None
    if feat_below_count > 0 and verbose:
        below_vals = feat_diff_vals[feat_below_min]
        orig_below_vals = original_feat_diff[feat_below_min]
        
        if hasattr(below_vals, 'head'):
            clamp_examples_new = below_vals.head(3).round(2)
            clamp_examples_orig = orig_below_vals.head(3).round(2)
        else:
            clamp_examples_new = below_vals[:3].round(2)
            clamp_examples_orig = orig_below_vals[:3].round(2)
    
    # Apply feat_diff constraints
    if is_dataframe:
        clean_data[feat_diff_col] = np.clip(clean_data[feat_diff_col], min_diss, None)  # Clamp minimum
        clean_data = clean_data[clean_data[feat_diff_col] <= max_diss].copy().reset_index(drop=True)  # Remove above max
        feat_diff_vals = clean_data[feat_diff_col]
        bias_vals = clean_data[bias_col]
    else:
        clean_data[:, 0] = np.clip(clean_data[:, 0], min_diss, None)  # Clamp minimum
        clean_data = clean_data[clean_data[:, 0] <= max_diss]  # Remove above max
        feat_diff_vals = clean_data[:, 0]
        bias_vals = clean_data[:, 1]
    
    # Show clamping examples
    if feat_below_count > 0 and verbose and clamp_examples_orig is not None:
        print(f"  Clamp examples: {list(zip(clamp_examples_orig, clamp_examples_new))}")
    
    # Check bias range and apply circular wrapping
    bias_out_of_range = (bias_vals < -180) | (bias_vals > 180)
    bias_out_count = bias_out_of_range.sum()
    
    if bias_out_count > 0 and verbose:
        print(f"Found {bias_out_count} bias values outside [-180, 180] - will wrap")
        bias_out_rows = bias_vals[bias_out_of_range]
        print(f"  Out-of-range bias: min={bias_out_rows.min():.1f}, max={bias_out_rows.max():.1f}")
    
    # Apply circular wrapping for bias
    if is_dataframe:
        original_bias = clean_data[bias_col].copy()
        clean_data[bias_col] = ((clean_data[bias_col] + 180) % 360) - 180
        bias_vals = clean_data[bias_col]
    else:
        original_bias = clean_data[:, 1].copy()
        clean_data[:, 1] = ((clean_data[:, 1] + 180) % 360) - 180
        bias_vals = clean_data[:, 1]
    
    # Show wrapping examples
    if bias_out_count > 0 and verbose:
        # Find positions of out-of-range values and take first 5
        if hasattr(bias_out_of_range, 'to_numpy'):
            out_of_range_positions = np.where(bias_out_of_range.to_numpy())[0][:5]
        else:
            out_of_range_positions = np.where(bias_out_of_range)[0][:5]
            
        if len(out_of_range_positions) > 0:
            if hasattr(original_bias, 'iloc'):
                orig_vals = original_bias.iloc[out_of_range_positions].round(1)
                new_vals = bias_vals.iloc[out_of_range_positions].round(1)
            else:
                orig_vals = original_bias[out_of_range_positions].round(1)
                new_vals = bias_vals[out_of_range_positions].round(1)
            print(f"  Wrap examples: {list(zip(orig_vals, new_vals))}")
    
    filtered_count = original_len - len(clean_data)
    if filtered_count > 0 and verbose:
        print(f"Filtered out {filtered_count}/{original_len} invalid data points")
    
    # Final statistics
    if verbose:
        if is_dataframe:
            feat_range = f"[{clean_data[feat_diff_col].min():.1f}, {clean_data[feat_diff_col].max():.1f}]"
            bias_range = f"[{clean_data[bias_col].min():.1f}, {clean_data[bias_col].max():.1f}]"
        else:
            feat_range = f"[{clean_data[:, 0].min():.1f}, {clean_data[:, 0].max():.1f}]"
            bias_range = f"[{clean_data[:, 1].min():.1f}, {clean_data[:, 1].max():.1f}]"
        
        print(f"Final: {len(clean_data)} trials, feat_diff range {feat_range}, bias range {bias_range}")
    
    return clean_data
