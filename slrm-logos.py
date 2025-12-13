# slrm-logos.py
# Author: Logos
# Version: V2.2 (Data Purification by Y-Averaging)
#
# Segmented Linear Regression Model (SLRM) Implementation.
# Core Logic: Deterministic Sequential Simplification Algorithm.
# This file contains the COMPLETE simulation flow (Instant Training, Lossless and Lossy Compression).

import numpy as np
import math

# --- CONFIGURATION ---
# Numerical tolerance for comparing floating-point numbers (virtual zero).
TOLERANCE = 1e-9 

# Default Epsilon (tolerance) for Lossy Compression.
# Represents the maximum accepted absolute error.
EPSILON = 0.03 

# --- UTILITY FUNCTIONS ---

def calculate_segment_params(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    """Calculates the Slope (P) and Intercept (O) of the line between two points."""
    # Check for vertical lines using tolerance.
    # NOTE: After purification in train_slrm, this check primarily serves as a safeguard 
    # against points pathologically close in X (not identical inputs).
    if np.isclose(x2, x1, atol=TOLERANCE):
        raise ZeroDivisionError("Points with identical X coordinates (vertical line) detected.")
    
    P = (y2 - y1) / (x2 - x1)
    O = y1 - P * x1
    return P, O

def format_key_for_display(key):
    """Ensures keys are displayed with two decimals and the '+' sign if positive."""
    if key >= 0:
        return f"+{key:.2f}"
    return f"{key:.2f}"

def print_mrls_dictionary(dictionary: dict, title: str):
    """Prints any SLRM/MRLS dictionary with enhanced and sorted formatting."""
    
    sorted_keys = sorted(dictionary.keys())
    
    print(f"\n--- {title} ---")
    print("// Key: X_start | Value: [P (Slope), O (Intercept)]")
    print("{")
    
    for key in sorted_keys:
        P, O = dictionary[key]
        
        key_str = format_key_for_display(key)
        
        # Use math.isnan to check if the value is undefined (NaN)
        p_str = f"{P:7.4f}" if not math.isnan(P) else "   NaN"
        # Include the sign for the intercept for clarity in printing
        o_str = f"{O:+7.4f}" if not math.isnan(O) else "   NaN"
        
        print(f"  {key_str}: [ {p_str}, {o_str} ]")
        
    print("}")
    # The dictionary always has one extra key (the NaN marker)
    num_segments = len(sorted_keys) - 1
    print(f"Total Segments: {num_segments if num_segments >= 0 else 0}")
    print("-----------------------------------")


# --- CORE ALGORITHM: SEQUENTIAL SEGMENT SIMPLIFICATION (The Logos Core) ---

def _sequential_segment_simplification(sorted_data: np.ndarray, tolerance: float) -> dict:
    """
    Implements the core deterministic sequential simplification algorithm.
    Extends the segment from a base point as far as possible while satisfying
    the provided tolerance (geometric invariance if tolerance=TOLERANCE, or epsilon if tolerance=EPSILON).
    
    Args:
        sorted_data: The input data, PURIFIED and sorted by X.
        tolerance: The maximum allowed absolute error (or TOLERANCE for lossless mode).
        
    Returns:
        dict: The SLRM dictionary of segments {X_start: [P, O]}.
    """
    N = len(sorted_data)
    final_dict = {}
    base_index = 0

    while base_index < N - 1:
        x_base, y_base = sorted_data[base_index]
        segment_end_index = base_index + 1
        
        # 'last_valid_index' tracks the furthest point that successfully defined the segment.
        last_valid_index = base_index + 1

        while segment_end_index < N:
            x_candidate, y_candidate = sorted_data[segment_end_index]
            
            # Since data is purified, x_candidate > x_base is guaranteed.
            # 1. Calculate the segment (P, O) from base to candidate
            try:
                P_cand, O_cand = calculate_segment_params(x_base, y_base, x_candidate, y_candidate)
            except ZeroDivisionError:
                # Should not be reached with purified data, but kept as final defensive layer.
                break 

            # 2. Check if ALL points *between* the base and the candidate satisfy the tolerance
            is_valid_segment = True
            
            # Iterate through points from base+1 up to (but NOT including) the candidate
            for i in range(base_index + 1, segment_end_index):
                x_inter, y_true = sorted_data[i]
                
                # Predict Y on the line (P_cand, O_cand)
                y_pred = x_inter * P_cand + O_cand
                
                # Check condition: 
                if tolerance == TOLERANCE: # Lossless Check (Geometric Invariance)
                    if not np.isclose(y_pred, y_true, atol=TOLERANCE):
                        is_valid_segment = False
                        break
                elif np.abs(y_true - y_pred) > tolerance: # Lossy Check (Epsilon Tolerance)
                    is_valid_segment = False
                    break
            
            # 3. Check if the candidate point itself satisfies the tolerance
            if is_valid_segment:
                y_pred_candidate = x_candidate * P_cand + O_cand
                if tolerance == TOLERANCE:
                     if not np.isclose(y_pred_candidate, y_candidate, atol=TOLERANCE):
                        is_valid_segment = False
                elif np.abs(y_candidate - y_pred_candidate) > tolerance:
                    is_valid_segment = False
            
            
            if is_valid_segment:
                # Segment can be extended: The candidate defines the valid segment.
                last_valid_index = segment_end_index
                segment_end_index += 1
            else:
                # An intermediate point or the candidate itself broke the tolerance/invariance. Stop search.
                break
        
        # 4. Register the final segment (from base_index to last_valid_index)
        
        x_end, y_end = sorted_data[last_valid_index]
        # This call is safe because the data is purified (x_end > x_base is guaranteed)
        P_final, O_final = calculate_segment_params(x_base, y_base, x_end, y_end)
        
        final_dict[x_base] = [P_final, O_final]
        
        # 5. Set the new base index
        base_index = last_valid_index

    # 6. Handle the very last point (marks the end of the dictionary)
    if base_index == N - 1:
        x_last, _ = sorted_data[base_index]
        final_dict[x_last] = [float('nan'), float('nan')] # Mark the end
    
    return final_dict

# --- SLRM STEPS ---

def compress_lossless(sorted_data: np.ndarray) -> dict:
    """
    Step 2: Lossless Compression (Geometric Invariance).
    Finds the minimum number of segments that perfectly represent the data.
    """
    print(f"\n--- 2. Lossless Compression (Geometric Invariance) ---")
    lossless_dict = _sequential_segment_simplification(sorted_data, TOLERANCE)
    print_mrls_dictionary(lossless_dict, "Result of Lossless Compression") # Print for validation
    
    return lossless_dict

def compress_lossy(sorted_data: np.ndarray, epsilon: float) -> dict:
    """
    Step 3: Lossy Compression (Epsilon Criterion).
    Finds the minimum number of segments that represent the data within max error 'epsilon'.
    """
    print(f"\n--- 3. Lossy Compression (Max Tolerance: {epsilon:.4f}) ---")
    lossy_dict = _sequential_segment_simplification(sorted_data, epsilon)
    return lossy_dict

def train_slrm(data: list, epsilon: float) -> dict:
    """
    Main function to run the SLRM training process.
    Includes an initial Data Purification step (Paso 0) to handle duplicate X values
    by using the average Y value, ensuring robustness against user input errors.
    
    Args:
        data: List of [X, Y] pairs.
        epsilon: Maximum absolute error allowed for lossy compression.

    Returns:
        dict: The Final SLRM Dictionary {X_start: [P (Slope), O (Intercept)]}.
    """
    if len(data) < 2:
        print("Error: At least 2 points are required for SLRM training (after purification).")
        return {}
    
    # --- 1. DATA PURIFICATION (Paso 0 for Robustness) ---
    # Handles X duplicates (identical X, different Y) by calculating the mean Y.
    # Structure: {X: [Y_sum, count]}
    aggregated_data = {}
    for x, y in data:
        # We ensure X is float for consistent key handling
        x_float = float(x)
        y_float = float(y)
        if x_float in aggregated_data:
            aggregated_data[x_float][0] += y_float
            aggregated_data[x_float][1] += 1
        else:
            aggregated_data[x_float] = [y_float, 1]

    # Convert aggregated data back to [X, Y_average] list
    purified_data = []
    for x, (y_sum, count) in aggregated_data.items():
        y_avg = y_sum / count
        purified_data.append([x, y_avg])
        
    if len(purified_data) < 2:
        print("Error: Less than 2 unique X points remain after purification.")
        return {}
        
    # --- 1. INSTANT TRAINING (Conversion and Sorting) ---
    
    input_array = np.array(purified_data, dtype=float)
    # Sorting by X (column 0)
    sorted_data = input_array[input_array[:, 0].argsort()]
    print(f"--- 1. Instant Training (Purified & Sorted Data, N={len(sorted_data)}) ---")

    # Step 2: Run and display Lossless Compression
    compress_lossless(sorted_data)
    
    # Step 3: Run Lossy Compression (Final Model Generation)
    final_model = compress_lossy(sorted_data, epsilon)
    
    # Display the final, most compressed model
    print_mrls_dictionary(final_model, "4. FINAL SLRM Dictionary (Lossy Compression)")

    return final_model

def predict_slrm(x_input: float, slrm_dict: dict) -> float:
    """
    Performs a prediction using the Final SLRM Dictionary (The Master Equation).
    Uses np.searchsorted for efficient segment lookup.
    """
    if not slrm_dict:
        return np.nan
        
    # Convert keys to a NumPy array for fast searchsorted lookup
    keys = np.array(list(slrm_dict.keys()))
    
    # np.searchsorted finds the index of the segment start (largest key <= x_input)
    idx = np.searchsorted(keys, x_input, side='right') - 1
    
    # 1. Handle Lower Extrapolation (x_input < keys[0]):
    # If x_input is less than the first key, idx will be -1. We default to the first segment (index 0).
    idx = max(0, idx) 
    
    active_key = keys[idx]
    P, O = slrm_dict.get(active_key, [np.nan, np.nan])
    
    # 2. Handle Upper Extrapolation (if the selected segment is the NaN marker):
    # If the active segment is the final NaN marker, we must use the parameters of the preceding segment.
    if math.isnan(P) and idx > 0:
        active_key = keys[idx - 1] # Use the second-to-last segment
        P, O = slrm_dict[active_key]
    elif math.isnan(P):
        # Handle case of a single-point model (no segments possible, only NaN marker)
        return np.nan 

    # The Master Equation: Y = X * P + O
    y_predicted = x_input * P + O
    
    return y_predicted

# --- DEMONSTRATION ---

if __name__ == '__main__':
    
    # Example Dataset (X, Y)
    # Includes:
    # 1. X Duplicates, Y different (4, 5 and 4, 6) -> Should become (4, 5.5)
    # 2. Exact Duplicates (2, 3 and 2, 3) -> Should become a single (2, 3)
    # 3. Normal points
    INPUT_SET = [
        [-6.00, -6.00], [2.00, 3.00], [-8.00, -4.00], [0.00, 0.00], [4.00, 5.0],
        [-4.00, -6.00], [6.00, 18.0], [4.00, 6.0], [2.00, 3.00], [-2.00, -4.00]
    ]

    print(f"--- SLRM (Logos V2.2) Training Demonstration (Includes Purification) ---")
    print(f"Input Data Points: {len(INPUT_SET)}")

    # TRAINING (This automatically prints steps 1, 2, 3, and 4)
    final_model = train_slrm(INPUT_SET, EPSILON)

    # PREDICTION TEST
    print("\n--- 5. PREDICTION TESTS ---")
    
    # Define test points for Interpolation and Extrapolation
    x_min_data = -8.0 # Min X in purified set
    x_max_data = 6.0  # Max X in purified set
    
    # Test points include lower extrapolation (-9.0), interpolation (4.0 - the averaged point), and upper extrapolation (8.0).
    test_points = [-9.0, -7.0, -5.5, 1.0, 4.0, 8.0]

    for x_test in test_points:
        y_pred = predict_slrm(x_test, final_model)
        
        is_extrapolation = x_test < x_min_data or x_test > x_max_data
        status = "EXTRAPOLATION" if is_extrapolation else "Interpolation"
        
        print(f"  X_in: {x_test:6.2f} | Y_pred: {y_pred:8.4f} | Type: {status}")
