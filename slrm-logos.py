# -*- coding: utf-8 -*-
# slrm-logos.py: Segmented Linear Regression Model (SLRM) Implementation
# with a lossy neural compression algorithm.
#
# This file contains the COMPLETE simulation flow for generating the
# compressed dictionary used in the prediction function.

import math

# --- UTILITY AND DISPLAY FUNCTIONS ---

def format_key_for_display(key):
    """Ensures keys are displayed with two decimals and the '+' sign if positive."""
    if key >= 0:
        return f"+{key:.2f}"
    return f"{key:.2f}"

def print_slrm_dictionary(dictionary, title="SLRM Dictionary"):
    """Prints any SLRM dictionary with enhanced and sorted formatting."""
    
    # Sort keys numerically
    sorted_keys = sorted(dictionary.keys())
    
    print(f"\n--- {title} ---")
    print("// Key: X_start | Value: [Slope (P), Intercept (O)]")
    print("{")
    
    for key in sorted_keys:
        P, O = dictionary[key]
        
        key_str = format_key_for_display(key)
        
        # Use math.isnan to check if the value is undefined (NaN)
        p_str = f"{P:7.2f}" if not math.isnan(P) else "   NaN"
        # Include the sign for the intercept for clarity in printing
        o_str = f"{O:+7.2f}" if not math.isnan(O) else "   NaN"
        
        print(f"  {key_str}: [ {p_str}, {o_str} ]")
        
    print("}")
    print("-----------------------------------")


# --- SLRM COMPRESSION SECTIONS ---

def create_base_dictionary(dataframe):
    """
    Section I & II: Creates the Base Dictionary (Input: X, Output: Y) by 
    sorting the data from smallest to largest by the Input (X).
    """
    print("--- 1. Creating Base Dictionary (Instant Training) ---")
    # Ensure keys and values are float
    base_dict = {float(x): float(y) for x, y in dataframe} 
    sorted_dict = dict(sorted(base_dict.items()))
    
    print(f"Base Dictionary Created (Total Points: {len(sorted_dict)})")
    return sorted_dict

def optimize_dictionary(base_dict):
    """
    Section III: Optimizes the Base Dictionary by calculating the Slope (Weight) 
    and the Intercept (Bias) for each segment between pairs of points.
    """
    print("\n--- 2. Optimizing Dictionary (Calculating Weights and Biases) ---")
    keys = list(base_dict.keys())
    optimized_dict = {}

    for i in range(len(keys) - 1):
        x1, y1 = keys[i], base_dict[keys[i]]
        x2, y2 = keys[i+1], base_dict[keys[i+1]]
        
        if x2 - x1 == 0:
            slope = float('nan')
            intercept = float('nan')
        else:
            # Slope (Weight) Formula
            slope = (y2 - y1) / (x2 - x1)
            # Intercept (Bias) Formula
            intercept = y1 - slope * x1
        
        # Storing as (Slope, Intercept)
        optimized_dict[x1] = (slope, intercept)

    # The last point: Marks the upper limit, its segment is undefined.
    optimized_dict[keys[-1]] = (float('nan'), float('nan'))
    
    print(f"Initial number of segments: {len(optimized_dict) - 1}")
    return optimized_dict

def compress_lossless(optimized_dict, epsilon=1e-6):
    """
    Section IV: Lossless Compression (Geometric Invariance).
    Removes intermediate points where the Slope of the adjacent segment is identical.
    """
    print("\n--- 3. Lossless Compression (Applying Geometric Invariance) ---")
    keys = list(optimized_dict.keys())
    compressed_dict = {}
    
    if not keys:
        return {}

    # The first point is always kept as the starting point
    compressed_dict[keys[0]] = optimized_dict[keys[0]]

    for i in range(1, len(keys) - 1):
        x_current = keys[i]
        
        slope_current, _ = optimized_dict[x_current]
        slope_prev, _ = optimized_dict[keys[i-1]]
        
        # Invariance check: If the difference in slopes is greater than epsilon, we keep the point.
        if abs(slope_current - slope_prev) > epsilon:
            compressed_dict[x_current] = optimized_dict[x_current]
    
    # Add the upper extreme point
    compressed_dict[keys[-1]] = optimized_dict[keys[-1]]
    
    print(f"Redundant Segments Removed (Remaining Points): {len(compressed_dict) - 1}")
    return compressed_dict


def compress_lossy(compressed_dict, base_dict, tolerance=0.03):
    """
    Section V: Lossy Compression (Human Criterion).
    Removes Neurons/points whose removal does not produce an error greater than the 
    tolerance at the original Base Dictionary points.
    """
    print(f"\n--- 4. Lossy Compression (Maximum Tolerance: {tolerance:.3f}) ---")
    
    # Transform the dictionary into a list of points representing active segments
    compressed_list = [(x, p, o) for x, (p, o) in compressed_dict.items() if not math.isnan(p)]
    
    final_compressed = []
    
    if not compressed_list:
        # Returns the dictionary with only the end point (NaN, NaN) if it was empty
        if compressed_dict:
             return {list(compressed_dict.keys())[-1]: (float('nan'), float('nan'))}
        return {}

    # The first point is always kept as the starting point
    final_compressed.append(compressed_list[0])

    for i in range(1, len(compressed_list)):
        x_current, p_current, o_current = compressed_list[i]
        
        # Try to 'skip' the current point (x_current) and use the slope of the previous point (x_prev)
        x_prev, p_prev, o_prev = final_compressed[-1] 
        
        # 1. Y_true: The original output value at X_current (or its lossless prediction if already removed)
        y_true = base_dict.get(x_current)
        if y_true is None:
             # If the point was already removed in Sec IV, use its lossless prediction as Y_true
             y_true = x_current * p_current + o_current 
        
        # 2. Y_hat: The prediction if the extended previous segment (p_prev, o_prev) is used
        y_hat = x_current * p_prev + o_prev
        
        error = abs(y_true - y_hat)
        
        # Criterion: If the error is greater than the tolerance, the point is Relevant and is kept.
        if error > tolerance:
            final_compressed.append(compressed_list[i])

    # Reconstruct the dictionary with lossy compression.
    lossy_dict = {x: (p, o) for x, p, o in final_compressed}
    
    # Add the upper extreme point
    if compressed_dict:
        lossy_dict[list(compressed_dict.keys())[-1]] = (float('nan'), float('nan')) 
    
    print(f"Non-Relevant Neurons Removed. Final segments: {len(lossy_dict) - 1}")
    return lossy_dict


# --- PREDICTION FUNCTION (Sections III and VII - The Clean Operational Core) ---

def predict(x: float, dictionary: dict) -> tuple:
    """
    Performs a prediction Y for a given X value using the compressed SLRM model.
    Applies generalization (extrapolation) if X is outside the training range.
    
    Args:
        x: The input value for which the prediction is required.
        dictionary: The SLRM dictionary containing the linear segments.
        
    Returns:
        A tuple (y_predicted, segment_info, description)
    """
    
    # 1. Prepare Keys and Limits
    keys = sorted(dictionary.keys())
    if len(keys) < 2:
        return None, "Error: The SLRM dictionary must have at least two points.", "ERROR"

    min_x = keys[0]             # The lower training limit
    final_x_limit = keys[-1]    # The absolute upper training limit
    
    # The second to last point is the key of the last valid segment (P and O defined)
    max_segment_key = keys[-2] 

    target_x = None
    description = "INTERPOLATION (Within Range)"

    # 2. Handling Generalization (Extrapolation)
    
    # Lower Extreme: If X is less than the lower limit, the first segment is used.
    if x < min_x:
        target_x = min_x
        description = "LOWER EXTRAPOLATION (< X_min)"
    
    # Search for the active segment (for interpolation or upper extrapolation)
    else:
        # Find the closest key X_n less than or equal to X
        # Search backwards to find the active segment (Xn <= X)
        for key in reversed(keys):
            # Only search keys that mark the start of a valid segment (not NaN)
            if x >= key and not math.isnan(dictionary.get(key, (None, None))[0]): 
                target_x = key
                break
        
        # Upper Extreme: If the segment found is the last valid one (max_segment_key) 
        # and X exceeds the final limit (final_x_limit), that segment is maintained.
        if target_x == max_segment_key and x >= final_x_limit:
            description = "UPPER EXTRAPOLATION (> X_max)"
        # If x is exactly the endpoint (keys[-1]), target_x will be max_segment_key 
        # (since keys[-1] has NaN) and is treated as the last valid interpolation.

    # 3. Prediction Calculation
    
    if target_x is None:
        # This occurs if the lowest segment is invalid or if there is a logical error
        return None, "Error: Could not find an active segment for X.", "ERROR"
        
    # Get the active segment parameters
    P, O = dictionary[target_x]
    
    # Handle the very unlikely case where target_x is the endpoint (NaN) but was not detected above.
    if math.isnan(P) or math.isnan(O):
        P, O = dictionary[max_segment_key]
        target_x = max_segment_key

    # Calculate Y
    y_predicted = x * P + O
    
    # 4. Segment Information
    segment_info = f"Segment: [X={target_x:.2f}] with P={P:.2f} and O={O:+.2f}"
    
    return y_predicted, segment_info, description


# --- TOY MODEL EXECUTION ---
# This block demonstrates the generation of the final dictionary and prediction tests.

if __name__ == "__main__":
    
    print("--- COMPLETE Segmented Linear Regression Model (SLRM) Demonstration ---")
    
    # I. Purified DataFrame (Training Data)
    # Data will be sorted internally: [-8, -4], [-6, -6], [-5, -6.01], [-4, -6], [-2, -4], [0, 0], [2, 4], [3, 7], [4, 10], [6, 18]
    dataframe = [
        [-6.00, -6.00], [2.00, 4.00], [-8.00, -4.00], [0.00, 0.00],
        [4.00, 10.0], [-4.00, -6.00], [6.00, 18.0], [-5.00, -6.01],
        [3.00, 7.00], [-2.00, -4.00]
    ]

    # 1. Base dictionary creation
    base_dict = create_base_dictionary(dataframe)

    # 2. Optimization (Weights/Biases)
    optimized_dict = optimize_dictionary(base_dict)

    # 3. Lossless Compression (Geometric Invariance)
    compressed_lossless_dict = compress_lossless(optimized_dict)

    # 4. Lossy Compression (Human Criterion)
    COMPRESSION_TOLERANCE = 0.03
    compressed_lossy_dict = compress_lossy(compressed_lossless_dict, base_dict, tolerance=COMPRESSION_TOLERANCE)
    
    # Display the final compression result (Dictionary used by the web)
    # Expected points to be removed: -5.00, -4.00, 0.00, 3.00
    print_slrm_dictionary(compressed_lossy_dict, "4. FINAL Compressed Dictionary (SLRM)")


    print("\n--- 5. PREDICTION TESTS WITH THE FINAL MODEL ---")
    
    test_values = [
        -10.0, # Lower Extrapolation
        -7.0,  # Interpolation Segment -8.0
        -5.0,  # Interpolation Segment -6.0 (point removed in lossy compression)
        -3.0,  # Safety Test
        0.0,   # Interpolation Segment -2.0 (point removed in lossy compression)
        3.0,   # Interpolation Segment +2.0 (point removed in lossy compression)
        5.0,   # Interpolation Segment +4.0
        6.0,   # Maximum Limit
        8.0    # Upper Extrapolation
    ]

    for x_val in test_values:
        y, info, desc = predict(x_val, compressed_lossy_dict)
        
        if y is not None:
            # Print the prediction result
            print(f"X = {x_val:6.2f} | Y pred = {y:7.3f} | {desc:25} | {info}")
        else:
            print(f"X = {x_val:6.2f} | Prediction error: {info}")
            
    print("\n-------------------------------------------")
