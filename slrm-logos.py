# slrm-logos.py
# Segmented Linear Regression Model (SLRM) - Logos Core
# Version: V5.5 (Diffusion Phase)
# Authors: Alex Kinetic and Logos
#
# Complete implementation of the training process (compression) and the optimized
# prediction function for the SLRM model. Uses NumPy for vectorization and speed optimization.

import numpy as np
import math
from collections import OrderedDict
import time

# --- GLOBAL CONSTANTS ---
# Error Tolerance (Epsilon) used for Lossy compression (Human Criterion).
# Default value used in the Technical Manual.
EPSILON = 0.03
# LRU Cache Size for the prediction function.
CACHE_SIZE = 100

# --- 1. PREDICTION CACHE (LRU Cache) ---

class LRUCache:
    """
    Simple Least Recently Used (LRU) cache optimized for SLRM prediction.
    Stores the last predictions to avoid repeated segment lookups.
    """
    def __init__(self, capacity):
        # OrderedDict maintains insertion order, useful for the LRU policy.
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        """Retrieves a value and moves it to the end (most recent)."""
        if key not in self.cache:
            return None
        # Move the key to the end to mark it as recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        """Inserts or updates a value. If capacity is exceeded, removes the least recently used item."""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # Remove the first element (the least recent)
                self.cache.popitem(last=False)
            self.cache[key] = value

# Initialize the global cache.
_prediction_cache = LRUCache(CACHE_SIZE)

# --- 2. TRAINING UTILITY FUNCTIONS ---

def _calculate_segment_parameters(X1, Y1, X2, Y2):
    """
    Calculates the Slope (P) and Intercept (O) of the line connecting (X1, Y1) and (X2, Y2).
    
    Master Equation: Y = X * P + O
    P (Slope/Weight): (Y2 - Y1) / (X2 - X1)
    O (Y-Intercept/Bias): Y1 - P * X1
    """
    if X2 == X1:
        # This should be handled in _clean_and_sort_data, but serves as a safeguard.
        return np.nan, np.nan 
        
    P = (Y2 - Y1) / (X2 - X1)
    O = Y1 - P * X1
    return P, O

def _clean_and_sort_data(input_data):
    """
    Section I & II: Data preparation. 
    1. Converts to a NumPy array.
    2. Sorts by X.
    3. Purifies: Handles X duplicates by averaging their Y values.
    """
    if not input_data:
        return np.array([]), np.array([])
        
    data_array = np.array(input_data, dtype=float)
    
    # 1. Sort by the X column (index 0)
    data_array = data_array[data_array[:, 0].argsort()]
    
    # 2. Purification: Handle X duplicates
    X_unique, indices, counts = np.unique(data_array[:, 0], return_index=True, return_counts=True)
    
    if np.all(counts == 1):
        # No duplicates, use the original sorted array.
        X_clean = data_array[:, 0]
        Y_clean = data_array[:, 1]
    else:
        # Duplicates exist: average the Y values for each unique X
        Y_sum = np.zeros_like(X_unique, dtype=float)
        
        # Group and sum the Y values for each unique X
        current_index = 0
        for i, count in enumerate(counts):
            Y_sum[i] = np.sum(data_array[current_index:current_index + count, 1])
            current_index += count
            
        Y_clean = Y_sum / counts
        X_clean = X_unique
        
    return X_clean, Y_clean

def _calculate_optimized_dict(X, Y):
    """
    Section III: Converts the (X, Y) points into segment parameters (P, O).
    """
    if len(X) < 2:
        return {}

    optimized_dict = {}
    
    # Vectorization of P and O calculations
    X1, Y1 = X[:-1], Y[:-1]
    X2, Y2 = X[1:], Y[1:]
    
    # Calculate Slope (P)
    P_values = (Y2 - Y1) / (X2 - X1)
    # Calculate Intercept (O)
    O_values = Y1 - P_values * X1
    
    # Create the dictionary: {X_start: [P, O]}
    for i in range(len(X1)):
        # Only store if the segment is valid (not vertical)
        if not np.isnan(P_values[i]):
            optimized_dict[X1[i]] = [P_values[i], O_values[i]]
            
    # Add a final NaN point as a segment end marker
    if len(X) > 0:
        optimized_dict[X[-1]] = [np.nan, np.nan] 

    return optimized_dict


# --- 3. COMPRESSION FUNCTIONS (Logos Core) ---

def _lossless_compression(optimized_dict, tolerance=1e-6):
    """
    Section IV: Lossless Compression (Geometric Invariance).
    Removes intermediate points that fall on the same straight line (same slope).
    """
    keys = sorted(optimized_dict.keys())
    if len(keys) < 3:
        return optimized_dict

    new_dict = OrderedDict()
    
    # Always keep the first point
    new_dict[keys[0]] = optimized_dict[keys[0]]
    
    i = 0
    while i < len(keys) - 2:
        P_current, _ = optimized_dict[keys[i]]
        P_next, _ = optimized_dict[keys[i+1]]
        
        # Criterion: If Slope(i) is approximately equal to Slope(i+1), 
        # the point keys[i+1] is redundant and is skipped (removed).
        if abs(P_current - P_next) < tolerance:
            # Geometric redundancy found. Skip the intermediate point (keys[i+1]).
            # Segment i now conceptually extends to keys[i+2].
            i += 1
        else:
            # Change of direction, the point keys[i+1] is relevant (breakpoint).
            new_dict[keys[i+1]] = optimized_dict[keys[i+1]]
            i += 1
            
    # Always keep the second to last relevant point and the final NaN marker
    if i < len(keys) - 1:
        new_dict[keys[i]] = optimized_dict[keys[i]]

    new_dict[keys[-1]] = optimized_dict[keys[-1]] # NaN marker

    return new_dict


def _lossy_compression(optimized_dict, epsilon):
    """
    Section V: Lossy Compression (Human Criterion/MRLS).
    Removes points whose interpolation between neighbors does not exceed the epsilon error.
    """
    keys = sorted(optimized_dict.keys())
    if len(keys) < 3:
        return optimized_dict

    # Convert the dictionary to a list of points (X, Y) temporarily
    X_all = np.array([k for k in keys[:-1]])
    
    # Calculate the original Y values at the X breakpoints. 
    Y_all = np.zeros_like(X_all)

    # Determine the Y value for each breakpoint X_start
    for i, x_start in enumerate(X_all):
        if i == 0:
            # For the first point, the Y is defined by the start of the first segment.
            P, O = optimized_dict[x_start]
            Y_all[i] = x_start * P + O # The original Y of the X_start point
        else:
            # For subsequent points, the Y is defined by the end of the previous segment.
            X_prev = X_all[i-1]
            P_prev, O_prev = optimized_dict[X_prev]
            Y_all[i] = x_start * P_prev + O_prev
            
    
    # MRLS (Minimum Required Line Segments) Algorithm
    i = 0
    while i < len(keys) - 2:
        # P0: Start point index
        # P1: Middle point index (candidate for removal)
        # P2: End point index
        
        X_start = keys[i]
        X_end = keys[i+2]
        
        # 1. Define the test line: from X_start to X_end
        P_test, O_test = _calculate_segment_parameters(X_start, Y_all[i], X_end, Y_all[i+2])
        
        # 2. Calculate the error of the intermediate point (keys[i+1])
        X_mid = keys[i+1]
        Y_true_mid = Y_all[i+1]
        
        # Check for vertical segment protection
        if math.isnan(P_test):
            # Cannot test, keep the intermediate point and move to the next segment
            i += 1
            continue

        Y_hat_mid = X_mid * P_test + O_test
        
        error = abs(Y_true_mid - Y_hat_mid)
        
        if error <= epsilon:
            # Error is within tolerance. The intermediate point is redundant (Lossy Compression).
            # Remove keys[i+1]. The segment X_start now extends to X_end.
            
            # Remove from keys and Y_all arrays
            keys.pop(i+1)
            Y_all = np.delete(Y_all, i+1)
            
            # Recalculate P and O for the newly extended segment (X_start to new X_next)
            X_new_next = keys[i+1]
            Y_new_next = Y_all[i+1]
            
            P_new, O_new = _calculate_segment_parameters(X_start, Y_all[i], X_new_next, Y_new_next)
            
            # Update the parameters for the segment starting at X_start
            optimized_dict[X_start] = [P_new, O_new]
            
            # Do NOT advance 'i' here, as we test the newly extended segment with the next point
        else:
            # Error > Epsilon. The point keys[i+1] is essential (breakpoint).
            # Advance to the next segment.
            i += 1
            
    # Reconstruct the final dictionary with the corrected parameters
    final_model = {}
    for key in keys:
        if key in optimized_dict:
            final_model[key] = optimized_dict[key]
        
    return final_model


# --- 4. MAIN TRAINING FUNCTION ---

def train_slrm(input_data, epsilon=EPSILON):
    """
    Trains the Segmented Linear Regression Model (SLRM) from (X, Y) data.
    The training is deterministic and non-iterative.

    Process:
    1. Base Dictionary (Cleaning and Sorting).
    2. Optimization (P and O Calculation).
    3. Lossless Compression (Geometric redundancy).
    4. Lossy Compression (Human Criterion/MRLS).

    Args:
        input_data (list of list/tuple): List of [X, Y] pairs.
        epsilon (float): Maximum error tolerance for Lossy compression.

    Returns:
        dict: The final SLRM model: {X_start: [P, O]}.
    """
    global _prediction_cache # Necessary to clear the cache

    if not input_data:
        print("Warning: Input data is empty. Returning empty model.")
        return {}

    # 1. Cleaning and Sorting (Base Dictionary)
    X_clean, Y_clean = _clean_and_sort_data(input_data)
    
    if len(X_clean) < 2:
        print("Warning: Less than 2 valid points after cleaning. Returning empty model.")
        return {}
        
    # 2. Optimization (Optimized Dictionary)
    # Generates the initial {X_start: [P, O]} dictionary
    optimized_dict = _calculate_optimized_dict(X_clean, Y_clean)

    # 3. Lossless Compression (Geometric Invariance)
    lossless_model = _lossless_compression(optimized_dict)

    # 4. Lossy Compression (Human Criterion/MRLS)
    final_model = _lossy_compression(lossless_model, epsilon)
    
    # Clear the prediction cache when training a new model
    _prediction_cache = LRUCache(CACHE_SIZE)
    
    return final_model


# --- 5. MAIN PREDICTION FUNCTION ---

def predict_slrm(x_in, slrm_model):
    """
    Predicts the Y value for an input X using the compressed SLRM model.

    1. Segment search using the closest key <= x_in.
    2. Master Equation: Y = X * P + O.
    3. Uses LRU cache for fast predictions.

    Args:
        x_in (float): The input X value to predict.
        slrm_model (dict): The final SLRM dictionary {X_start: [P, O]}.

    Returns:
        dict: Result with predicted Y and active segment parameters.
    """
    if not slrm_model:
        return {'x_in': x_in, 'y_pred': np.nan, 'slope_P': np.nan, 'intercept_O': np.nan, 'cache_hit': False}

    # Attempt to get the prediction from the cache
    cached_result = _prediction_cache.get(x_in)
    if cached_result is not None:
        cached_result['cache_hit'] = True
        return cached_result
        
    keys = sorted(slrm_model.keys())
    
    # 1. Find the active segment
    
    active_segment_x_start = np.nan
    
    # Find the largest X_start key that is <= x_in
    # This defines the active segment [X_start, X_next)
    for key in reversed(keys):
        if key <= x_in:
            active_segment_x_start = key
            break
            
    if np.isnan(active_segment_x_start):
        # Extrapolation (Left Extreme Segment)
        active_segment_x_start = keys[0]
        
    # Extract P and O parameters of the active segment
    P, O = slrm_model.get(active_segment_x_start, [np.nan, np.nan])
    
    if math.isnan(P) or math.isnan(O):
        # Handle the final NaN marker case or a search error
        if active_segment_x_start == keys[-1] and math.isnan(P):
            # It's the final (NaN) marker point, use the parameters of the previous segment
            prev_key = keys[-2]
            P, O = slrm_model.get(prev_key, [np.nan, np.nan])
        else:
            return {'x_in': x_in, 'y_pred': np.nan, 'slope_P': np.nan, 'intercept_O': np.nan, 'cache_hit': False}
            
    # 2. Master Equation
    y_pred = x_in * P + O
    
    result = {
        'x_in': x_in, 
        'y_pred': y_pred, 
        'slope_P': P, 
        'intercept_O': O, 
        'cache_hit': False
    }

    # Save to cache
    _prediction_cache.put(x_in, result)
    
    return result

# --- QUICK USAGE EXAMPLE TEST ---
if __name__ == '__main__':
    
    # 1. Define the example data (same as in README.md)
    INPUT_DATA_EXAMPLE = [
        [-6.00,-6.00], [+2.00,+4.00], [-8.00,-4.00], [+0.00,+0.00],
        [+4.00,+10.0], [-4.00,-6.00], [+6.00,+18.0], [-5.00,-6.01],
        [+3.00,+7.00], [-2.00,-4.00], [+1.00, +2.00] # Added a collinear point for lossless test
    ]
    
    print("--- SLRM Training and Prediction Example ---")
    print(f"Epsilon used: {EPSILON}")
    
    start_time = time.time()
    # 2. Train the model
    final_slrm_model = train_slrm(INPUT_DATA_EXAMPLE, EPSILON)
    training_duration = time.time() - start_time
    
    print(f"\n[INFO] Training completed in {training_duration:.4f} seconds.")
    print(f"[INFO] Original points: {len(INPUT_DATA_EXAMPLE)}")
    print(f"[INFO] Final SLRM segments (Breakpoints): {len(final_slrm_model) - 1}")
    
    print("\n--- Optimized Dictionary (Final Model) ---")
    for x_start, params in final_slrm_model.items():
        P, O = params
        if not math.isnan(P):
            print(f"X_Start: {x_start:+.2f} -> P: {P:+.4f}, O: {O:+.4f}")
        else:
            print(f"X_Start: {x_start:+.2f} (End Marker)")
            
    # 3. Prediction Test
    print("\n--- Prediction Test ---")
    X_TEST_VALUES = [-7.0, -5.5, -4.0, 1.5, 5.0, 7.0]
    
    for x in X_TEST_VALUES:
        result = predict_slrm(x, final_slrm_model)
        
        # Cache test for the last point
        hit = False
        if x == 7.0:
            result_cached = predict_slrm(x, final_slrm_model)
            hit = result_cached['cache_hit']
        
        print(f"Predict X={result['x_in']:+.2f} | Y={result['y_pred']:+.4f} | Active Segment P={result['slope_P']:+.4f}, O={result['intercept_O']:+.4f} | Cache Hit: {hit}")

    print(f"\n[INFO] LRU Cache Status (Size: {len(_prediction_cache.cache)}/{CACHE_SIZE})")
