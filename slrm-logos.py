# slrm-logos.py
# Author: Logos
# Version: V2.0
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
EPSILON = 0.03 

# --- UTILITY FUNCTIONS ---

def calculate_segment_params(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    """Calculates the Slope (P) and Intercept (O) of the line between two points."""
    # Check for vertical lines using tolerance. This function assumes data is purified.
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
    """Prints any MRLS dictionary with enhanced and sorted formatting."""
    
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
    print(f"Total Segments: {len(sorted_keys) - 1}")
    print("-----------------------------------")


# --- CORE ALGORITHM: SEQUENTIAL SEGMENT SIMPLIFICATION (The Logos Core) ---

def _sequential_segment_simplification(sorted_data: np.ndarray, tolerance: float) -> dict:
    """
    Implements the core deterministic sequential simplification algorithm.
    It extends the segment from a base point as far as possible while satisfying
    the provided tolerance (geometric invariance if tolerance=TOLERANCE, or epsilon if tolerance=EPSILON).
    
    Args:
        sorted_data: The input data, sorted by X.
        tolerance: The maximum allowed absolute error (or TOLERANCE for lossless mode).
        
    Returns:
        dict: The MRLS dictionary of segments {X_start: [P, O]}.
    """
    N = len(sorted_data)
    final_dict = {}
    base_index = 0

    while base_index < N - 1:
        x_base, y_base = sorted_data[base_index]
        segment_end_index = base_index + 1
        
        # 'last_valid_index' tracks the furthest point that successfully defines the segment
        last_valid_index = base_index 

        while segment_end_index < N:
            x_candidate, y_candidate = sorted_data[segment_end_index]
            
            # 1. Calculate the segment (P, O) from base to candidate
            try:
                P_cand, O_cand = calculate_segment_params(x_base, y_base, x_candidate, y_candidate)
            except ZeroDivisionError:
                # If vertical line is encountered, stop segment search here.
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
            
            # 3. Check if the candidate point itself satisfies the tolerance (Only needed for Lossy, 
            # but applying it here simplifies the loop break logic)
            if is_valid_segment:
                y_pred_candidate = x_candidate * P_cand + O_cand
                if tolerance == TOLERANCE:
                     if not np.isclose(y_pred_candidate, y_candidate, atol=TOLERANCE):
                        is_valid_segment = False
                elif np.abs(y_candidate - y_pred_candidate) > tolerance:
                    is_valid_segment = False
            
            
            if is_valid_segment:
                # Segment can be extended: The candidate itself defines the valid segment.
                last_valid_index = segment_end_index
                segment_end_index += 1
            else:
                # An intermediate point or the candidate itself broke the tolerance/invariance. Stop search.
                break
        
        # 4. Register the final segment (from base_index to last_valid_index)
        
        # If the index did not advance (N=2 case, or immediate failure), last_valid_index must be at least base_index + 1
        if last_valid_index == base_index:
             # This should only happen if the segment search failed immediately, meaning the segment
             # must minimally connect to the next point (N is at least 2).
             last_valid_index = base_index + 1

        x_end, y_end = sorted_data[last_valid_index]
        P_final, O_final = calculate_segment_params(x_base, y_base, x_end, y_end)
        
        final_dict[x_base] = [P_final, O_final]
        
        # 5. Set the new base index
        base_index = last_valid_index

    # 6. Handle the very last point (marks the end of the dictionary)
    if base_index == N - 1:
        x_last, _ = sorted_data[base_index]
        final_dict[x_last] = [float('nan'), float('nan')] # Mark the end
    
    return final_dict

# --- MRLS STEPS ---

def compress_lossless(sorted_data: np.ndarray) -> dict:
    """
    Step 2: Lossless Compression (Geometric Invariance).
    Finds the minimum number of segments that perfectly represent the data.
    """
    print(f"\n--- 2. Lossless Compression (Geometric Invariance) ---")
    lossless_dict = _sequential_segment_simplification(sorted_data, TOLERANCE)
    
    # NOTE: In V2.0, we proceed directly to Lossy Compression on the original data,
    # as the Lossless result is implicitly captured by the lossy check at epsilon=TOLERANCE.
    # We run it here primarily for logging/debugging the intermediate step.
    return lossless_dict

def compress_lossy(sorted_data: np.ndarray, epsilon: float) -> dict:
    """
    Step 3: Lossy Compression (Epsilon Criterion).
    Finds the minimum number of segments that represent the data within max error 'epsilon'.
    """
    print(f"\n--- 3. Lossy Compression (Max Tolerance: {epsilon:.4f}) ---")
    lossy_dict = _sequential_segment_simplification(sorted_data, epsilon)
    return lossy_dict

def train_mrls(data: list, epsilon: float) -> dict:
    """
    Main function to run the MRLS training process.
    
    Args:
        data: List of [X, Y] pairs.
        epsilon: Maximum absolute error allowed for lossy compression.

    Returns:
        dict: The Final MRLS Dictionary {X_start: [P (Slope), O (Intercept)]}.
    """
    # 1. PREPARATION: Convert to NumPy array and sort (Instant Training)
    
    if len(data) < 2:
        print("Error: At least 2 points are required for MRLS training.")
        return {}
    
    # NOTE ON STEP 0: Data Purification (Handling duplicate X) is assumed to be
    # completed prior to calling this function to maintain core logic purity.

    input_array = np.array(data, dtype=float)
    # Sorting by X (column 0)
    sorted_data = input_array[input_array[:, 0].argsort()]
    print(f"--- 1. Instant Training (Sorted Data, N={len(sorted_data)}) ---")

    # The lossless step is conceptually necessary but in V2.0,
    # we rely on the single simplification core logic for the final result.
    compress_lossless(sorted_data)
    
    # 3. Lossy Compression (Step 3 - Final Model Generation)
    final_model = compress_lossy(sorted_data, epsilon)

    return final_model

def predict_mrls(x_input: float, mrls_dict: dict) -> float:
    """
    Performs a prediction using the Final MRLS Dictionary (The Master Equation).
    """
    if not mrls_dict:
        return np.nan

    keys = sorted(list(mrls_dict.keys()))
    
    # Search for the active segment (the largest X_start that is <= x_input)
    active_key = None
    for key in reversed(keys):
        # Use TOLERANCE for floating-point comparison
        if x_input >= key - TOLERANCE: 
            active_key = key
            break
            
    # Handle Lower Extrapolation: If x_input is below the first X_start, use the first segment.
    if active_key is None:
        active_key = keys[0]

    P, O = mrls_dict.get(active_key, [np.nan, np.nan])
    
    # If the active key is the final (NaN) point, use the segment starting at the second-to-last key
    if math.isnan(P):
        active_key = keys[-2]
        P, O = mrls_dict[active_key]

    # The Master Equation: Y = X * P + O
    y_predicted = x_input * P + O
    
    return y_predicted

# --- DEMONSTRATION ---

if __name__ == '__main__':
    
    # Example Dataset (X, Y)
    INPUT_SET = [
        [-6.00, -6.00], [2.00, 4.00], [-8.00, -4.00], [0.00, 0.00], [4.00, 10.0],
        [-4.00, -6.00], [6.00, 18.0], [-5.00, -6.01], [3.00, 7.00], [-2.00, -4.00]
    ]

    print(f"--- MRLS (Logos V2.0) Training Demonstration ---")
    print(f"Input Data Points: {len(INPUT_SET)}")

    # TRAINING
    final_model = train_mrls(INPUT_SET, EPSILON)

    # DISPLAY RESULTS
    print_mrls_dictionary(final_model, "4. FINAL MRLS Dictionary (Lossy Compression)")

    # PREDICTION TEST
    print("\n--- 5. PREDICTION TESTS ---")
    
    # Define test points for Interpolation and Extrapolation
    x_min_data = min(x[0] for x in INPUT_SET)
    x_max_data = max(x[0] for x in INPUT_SET)
    
    test_points = [-9.0, -7.0, -5.5, 1.0, 5.0, 8.0]

    for x_test in test_points:
        y_pred = predict_mrls(x_test, final_model)
        
        is_extrapolation = x_test < x_min_data or x_test > x_max_data
        status = "EXTRAPOLATION" if is_extrapolation else "Interpolation"
        
        print(f"  X_in: {x_test:6.2f} | Y_pred: {y_pred:8.4f} | Type: {status}")
        
