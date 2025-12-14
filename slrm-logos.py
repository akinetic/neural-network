# slrm-logos.py
# Author: Logos
# Version: V4.0 (Core Vectorization for Training Speed)
#
# Segmented Linear Regression Model (SLRM) Implementation.
# Core Logic: Deterministic Sequential Simplification Algorithm.
# This file contains the COMPLETE simulation flow (Instant Training, Lossless and Lossy Compression).

import numpy as np
import math
import collections # Used for the prediction cache (OrderedDict)

# --- CONFIGURATION ---
# Numerical tolerance for comparing floating-point numbers (virtual zero).
TOLERANCE = 1e-9 

# Default Epsilon (tolerance) for Lossy Compression.
# Represents the maximum accepted absolute error.
EPSILON = 0.03 

# Cache size for predictions (Least Recently Used strategy).
PREDICTION_CACHE_SIZE = 100 

# Global Prediction Cache (Initialized as an OrderedDict for LRU behavior)
# Stores {x_input: result_dictionary}
prediction_cache = collections.OrderedDict()


# --- UTILITY FUNCTIONS ---

def calculate_segment_params(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    """Calculates the Slope (P) and Intercept (O) of the line between two points."""
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
    V4.0: The segment check is now fully vectorized using NumPy for massive speedup.
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
            
            # 1. Calculate the segment (P, O) from base to candidate
            try:
                P_cand, O_cand = calculate_segment_params(x_base, y_base, x_candidate, y_candidate)
            except ZeroDivisionError:
                break 

            # --- V4.0: VECTORIZED CHECK ---
            
            # Slice the data to check (from base_index up to and including segment_end_index)
            intermediate_data = sorted_data[base_index : segment_end_index + 1]
            X_segment = intermediate_data[:, 0]
            Y_true_segment = intermediate_data[:, 1]
            
            # Calculate predicted Y values for the entire segment at once: Y = X * P + O
            Y_pred_segment = X_segment * P_cand + O_cand
            
            is_valid_segment = False

            if tolerance == TOLERANCE: # Lossless Check (Geometric Invariance)
                # np.isclose checks if all points lie exactly on the line (within TOLERANCE)
                is_valid_segment = np.all(np.isclose(Y_pred_segment, Y_true_segment, atol=TOLERANCE))
            else: # Lossy Check (Epsilon Tolerance)
                # Calculate absolute errors and check if the maximum error is within epsilon
                errors = np.abs(Y_true_segment - Y_pred_segment)
                max_error = np.max(errors)
                is_valid_segment = max_error <= tolerance
            
            # --- END V4.0: VECTORIZED CHECK ---

            if is_valid_segment:
                # Segment can be extended: The candidate defines the valid segment.
                last_valid_index = segment_end_index
                segment_end_index += 1
            else:
                # An intermediate point or the candidate itself broke the tolerance/invariance. Stop search.
                break
        
        # 3. Register the final segment (from base_index to last_valid_index)
        x_end, y_end = sorted_data[last_valid_index]
        P_final, O_final = calculate_segment_params(x_base, y_base, x_end, y_end)
        
        final_dict[x_base] = [P_final, O_final]
        
        # 4. Set the new base index
        base_index = last_valid_index

    # 5. Handle the very last point (marks the end of the dictionary)
    if base_index == N - 1:
        x_last, _ = sorted_data[base_index]
        final_dict[x_last] = [float('nan'), float('nan')] # Mark the end
    
    return final_dict

# --- VALIDATION FUNCTION (L_inf Norm Check) ---
# NOTE: This function remains loop-based as it relies on segment lookup (prediction) for each point.

def validate_max_error(original_data: np.ndarray, slrm_dict: dict, max_allowed_epsilon: float):
    """
    Calculates the true maximum absolute error (L_inf norm) of the compressed model
    against the original, purified data points.
    """
    max_error = 0.0
    
    for x_true, y_true in original_data:
        # We use the internal prediction logic to find the P and O used for this X
        P, O = predict_slrm_internal(x_true, slrm_dict)[1:] # [1:] extracts P and O
            
        if math.isnan(P) or math.isnan(O):
            continue
            
        y_pred = x_true * P + O
        error = np.abs(y_true - y_pred)
        
        if error > max_error:
            max_error = error
    
    print("\n--- 5. Validation of Max Absolute Error (L_inf Norm) ---")
    print(f"Epsilon used for compression: {max_allowed_epsilon:.4f}")
    print(f"Real Max Absolute Error found: {max_error:.6f}")
    
    if max_error <= max_allowed_epsilon + TOLERANCE:
        print("Model Status: ✅ PASSED (Max Error within Epsilon tolerance).")
    else:
        print("Model Status: ❌ FAILED (Max Error exceeded Epsilon).")


# --- SLRM STEPS ---

def compress_lossless(sorted_data: np.ndarray) -> dict:
    """
    Step 2: Lossless Compression (Geometric Invariance).
    """
    print(f"\n--- 2. Lossless Compression (Geometric Invariance) ---")
    lossless_dict = _sequential_segment_simplification(sorted_data, TOLERANCE)
    print_mrls_dictionary(lossless_dict, "Result of Lossless Compression") 
    
    return lossless_dict

def compress_lossy(sorted_data: np.ndarray, epsilon: float) -> dict:
    """
    Step 3: Lossy Compression (Epsilon Criterion).
    """
    print(f"\n--- 3. Lossy Compression (Max Tolerance: {epsilon:.4f}) ---")
    lossy_dict = _sequential_segment_simplification(sorted_data, epsilon)
    
    # Validation step added in V2.3
    validate_max_error(sorted_data, lossy_dict, epsilon)
    
    return lossy_dict

def train_slrm(data: list, epsilon: float) -> dict:
    """
    Main function to run the SLRM training process.
    """
    if len(data) < 2:
        print("Error: At least 2 points are required for SLRM training (after purification).")
        return {}
    
    # --- 1. DATA PURIFICATION (Paso 0 for Robustness) ---
    aggregated_data = {}
    for x, y in data:
        x_float = float(x)
        y_float = float(y)
        if x_float in aggregated_data:
            aggregated_data[x_float][0] += y_float
            aggregated_data[x_float][1] += 1
        else:
            aggregated_data[x_float] = [y_float, 1]

    purified_data = []
    for x, (y_sum, count) in aggregated_data.items():
        y_avg = y_sum / count
        purified_data.append([x, y_avg])
        
    if len(purified_data) < 2:
        print("Error: Less than 2 unique X points remain after purification.")
        return {}
        
    # --- 1. INSTANT TRAINING (Conversion and Sorting) ---
    input_array = np.array(purified_data, dtype=float)
    sorted_data = input_array[input_array[:, 0].argsort()]
    print(f"--- 1. Instant Training (Purified & Sorted Data, N={len(sorted_data)}) ---")

    # Step 2: Run and display Lossless Compression
    compress_lossless(sorted_data)
    
    # Step 3: Run Lossy Compression (Final Model Generation)
    final_model = compress_lossy(sorted_data, epsilon)
    
    # Display the final, most compressed model
    print_mrls_dictionary(final_model, "4. FINAL SLRM Dictionary (Lossy Compression)")

    return final_model

def predict_slrm_internal(x_input: float, slrm_dict: dict) -> tuple[float, float, float]:
    """
    Internal function to find the segment parameters (P, O) for a given X.
    Returns: (Active_X, P, O) or (NaN, NaN, NaN)
    """
    if not slrm_dict:
        return np.nan, np.nan, np.nan
        
    keys = np.array(list(slrm_dict.keys()))
    
    # Find the index of the segment start (largest key <= x_input)
    idx = np.searchsorted(keys, x_input, side='right') - 1
    
    # 1. Handle Lower Extrapolation (x_input < keys[0]): default to first segment (index 0).
    idx = max(0, idx) 
    
    active_key = keys[idx]
    P, O = slrm_dict.get(active_key, [np.nan, np.nan])
    
    # 2. Handle Upper Extrapolation (if the selected segment is the NaN marker):
    if math.isnan(P) and idx > 0:
        active_key = keys[idx - 1] # Use the second-to-last segment
        P, O = slrm_dict[active_key]
    elif math.isnan(P):
        return np.nan, np.nan, np.nan

    return active_key, P, O


def predict_slrm(x_input: float, slrm_dict: dict) -> dict:
    """
    Performs a prediction using the Final SLRM Dictionary (The Master Equation).
    V3.0 Feature: Uses prediction cache for speed optimization.
    
    Returns:
        dict: {
            'x_in': x_input, 
            'y_pred': float, 
            'segment_x_start': float, 
            'slope_P': float, 
            'intercept_O': float
        }
    """
    # --- V3.0: CACHE CHECK ---
    if x_input in prediction_cache:
        # Move the key to the end to mark it as most recently used (LRU logic)
        prediction_cache.move_to_end(x_input)
        return prediction_cache[x_input]

    # --- NO CACHE HIT: CALCULATE ---
    active_key, P, O = predict_slrm_internal(x_input, slrm_dict)
    
    if math.isnan(P):
        result = {
            'x_in': x_input, 
            'y_pred': np.nan, 
            'segment_x_start': np.nan, 
            'slope_P': np.nan, 
            'intercept_O': np.nan
        }
    else:
        # The Master Equation: Y = X * P + O
        y_predicted = x_input * P + O
        
        result = {
            'x_in': x_input, 
            'y_pred': y_predicted, 
            'segment_x_start': active_key, 
            'slope_P': P, 
            'intercept_O': O
        }

    # --- V3.0: CACHE UPDATE ---
    prediction_cache[x_input] = result
    
    # Ensure cache size limit is maintained (LRU)
    if len(prediction_cache) > PREDICTION_CACHE_SIZE:
        prediction_cache.popitem(last=False) # Removes the first (least recently used) item
        
    return result

# --- DEMONSTRATION ---

if __name__ == '__main__':
    
    # Example Dataset (X, Y)
    INPUT_SET = [
        [-6.00, -6.00], [2.00, 3.00], [-8.00, -4.00], [0.00, 0.00], [4.00, 5.0],
        [-4.00, -6.00], [6.00, 18.0], [4.00, 6.0], [2.00, 3.00], [-2.00, -4.00],
        [8.00, 26.00], [10.00, 27.00]
    ]

    print(f"--- SLRM (Logos V4.0) Training Demonstration (Core Vectorization Enabled) ---")
    print(f"Input Data Points: {len(INPUT_SET)}")
    print(f"Prediction Cache Size: {PREDICTION_CACHE_SIZE}")

    # TRAINING (Steps 1-5 printed)
    final_model = train_slrm(INPUT_SET, EPSILON)

    # PREDICTION TEST
    print("\n--- 6. PREDICTION TESTS (Cache Demonstration) ---")
    
    x_min_data = -8.0 
    x_max_data = 10.0  
    
    # Test points (4.0 is repeated to hit the cache)
    test_points = [-10.0, -7.0, 4.0, 1.0, 4.0, 12.0]

    for x_test in test_points:
        # Check cache status before prediction
        cache_hit = x_test in prediction_cache
        
        result = predict_slrm(x_test, final_model)
        
        is_extrapolation = result['x_in'] < x_min_data or result['x_in'] > x_max_data
        status = "EXTRAPOLATION" if is_extrapolation else "INTERPOLATION"
        
        cache_msg = "HIT" if cache_hit else "MISS"
        
        # Enhanced output display
        print(f"  X_in: {result['x_in']:6.2f} | Y_pred: {result['y_pred']:8.4f} | Status: {status:15} | Cache: {cache_msg:4} | Segment P/O: {result['slope_P']:6.4f} / {result['intercept_O']:+6.4f}")
        
    print(f"\nCache final size: {len(prediction_cache)}")
