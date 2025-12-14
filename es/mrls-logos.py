# slrm-logos.py
# Segmented Linear Regression Model (SLRM) - Logos Core
# Version: V5.8 (Geometric Invariance & MRLS Optimization)
# Authors: Alex Kinetic and Logos
#
# Complete implementation of the SLRM training (compression) and optimized
# prediction process. Uses the robust two-phase compression: Lossless (Geometric
# Invariance) followed by Lossy (MRLS, Human Criterion). Features an LRU cache
# for prediction speed.

import numpy as np
import math
from collections import OrderedDict
from typing import List, Tuple, Dict, Any

# --- GLOBAL CONSTANTS ---
# Default Error Tolerance (Epsilon) for Lossy compression.
EPSILON = 0.50
# LRU Cache Size for the prediction function.
CACHE_SIZE = 100
# Tolerance used for float comparisons (Geometric Invariance or Epsilon=0)
FLOAT_TOLERANCE = 1e-9

# --- 1. PREDICTION CACHE (LRU Cache) ---

class LRUCache:
    """
    Simple Least Recently Used (LRU) cache optimized for SLRM prediction.
    Stores the last predictions to avoid repeated segment lookups.
    """
    def __init__(self, capacity: int):
        # OrderedDict maintains insertion order, useful for the LRU policy.
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: float) -> Optional[Dict[str, Any]]:
        """Retrieves a value and moves it to the end (most recent)."""
        if key not in self.cache:
            return None
        # Move the key to the end to mark it as recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: float, value: Dict[str, Any]):
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

# ==============================================================================
# 2. TRAINING UTILITY FUNCTIONS & PREPROCESSING
# ==============================================================================

def _clean_and_sort_data(data_string: str) -> List[Tuple[float, float]]:
    """
    Parses and cleans the input data string.
    1. Handles formatting (commas, spaces).
    2. Sorts by X value.
    3. Purifies: Handles X duplicates by averaging their Y values.
    Returns a clean, sorted list of (X, Y) tuples.
    """
    points_map = {}
    
    for line in data_string.strip().split('\n'):
        # Split by comma or space
        parts = line.strip().replace(',', ' ').split()
        if len(parts) >= 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                
                if x in points_map:
                    current_y, count = points_map[x]
                    new_y = (current_y * count + y) / (count + 1)
                    points_map[x] = (new_y, count + 1)
                else:
                    points_map[x] = (y, 1)
            except ValueError:
                continue

    cleaned_data = [(x, y_count[0]) for x, y_count in points_map.items()]
    cleaned_data.sort(key=lambda p: p[0])
    
    return cleaned_data

# ==============================================================================
# 3. COMPRESSION FUNCTIONS (LOGOS CORE V5.8)
# ==============================================================================

def _lossless_compression(data: List[Tuple[float, float]]) -> List[float]:
    """
    Section IV: Lossless Compression (Geometric Invariance).
    Removes intermediate points colineal with their neighbors.
    Returns a list of critical X-breakpoints.
    """
    if len(data) < 3:
        return [p[0] for p in data]

    critical_x = [data[0][0]]
    
    for i in range(1, len(data) - 1):
        p0, p1, p2 = data[i - 1], data[i], data[i + 1]

        dx_a = p1[0] - p0[0]
        dx_b = p2[0] - p1[0]

        # Use absolute check for collinearity if segments are not vertical
        if dx_a != 0 and dx_b != 0:
            P_a = (p1[1] - p0[1]) / dx_a
            P_b = (p2[1] - p1[1]) / dx_b

            # Criterion: If slopes are NOT equal (using tolerance), it is a breakpoint.
            if abs(P_a - P_b) > FLOAT_TOLERANCE:
                critical_x.append(p1[0])
        else:
             # Case of vertical segments or coincident points (already handled by cleaning)
             critical_x.append(p1[0])
    
    # Always keep the last point
    if len(data) > 1:
        critical_x.append(data[-1][0])

    return sorted(list(set(critical_x)))


def _lossy_compression(initial_keys: List[float], epsilon: float, data: List[Tuple[float, float]]) -> Tuple[Dict[float, List[float]], float]:
    """
    Section V: Lossy Compression (MRLS - Minimum Required Line Segments).
    Finds the longest possible segment from each breakpoint that respects epsilon.
    Returns: (SLRM_Model: {X_start: [P, O, X_end]}, Max_Error_Achieved)
    """
    if len(initial_keys) < 2:
        return {}, 0.0

    data_map = {x: y for x, y in data}
    data_x_list = [x for x, y in data]
    
    # Critical Epsilon Logic: If user sets epsilon=0, we enforce strict checking (1e-12).
    # Otherwise, we use the user's epsilon.
    epsilon_threshold = max(epsilon, 1e-12) if epsilon == 0 else epsilon

    final_model: Dict[float, List[float]] = {}
    i = 0  # Index of the starting breakpoint in initial_keys
    max_overall_error = 0.0

    while i < len(initial_keys) - 1:
        
        x_start = initial_keys[i]
        y_start = data_map[x_start]
        
        j = i + 1  # Index of the candidate ending breakpoint (x_end_candidate)

        while j < len(initial_keys):
            x_end_candidate = initial_keys[j]
            y_end_candidate = data_map[x_end_candidate]

            dx = x_end_candidate - x_start
            
            # 1. Calculate the test line (P_test, O_test)
            if dx == 0:
                P_test, O_test = np.nan, np.nan
            else:
                P_test = (y_end_candidate - y_start) / dx
                O_test = y_start - P_test * x_start
            
            error_exceeded = False
            current_max_error = 0.0
            
            # Find point indices for bounds check
            start_index = data_x_list.index(x_start)
            end_index = data_x_list.index(x_end_candidate)

            # 2. Check all intermediate points against the test line
            for k in range(start_index + 1, end_index):
                x_mid, y_true_mid = data[k]
                
                y_hat_mid = P_test * x_mid + O_test
                error = abs(y_true_mid - y_hat_mid)

                current_max_error = max(current_max_error, error)

                if error > epsilon_threshold:
                    error_exceeded = True
                    break
            
            if error_exceeded:
                # Segment failed at index j. Commit the previous valid segment (i -> j-1).
                x_end_committed = initial_keys[j - 1]
                y_end_committed = data_map[x_end_committed]
                
                # Recalculate P and O for the COMMITTED segment
                dx_committed = x_end_committed - x_start
                if dx_committed == 0:
                    P, O = np.nan, np.nan
                else:
                    P = (y_end_committed - y_start) / dx_committed
                    O = y_start - P * x_start
                
                final_model[x_start] = [P, O, x_end_committed]
                
                # Update max overall error (using the committed segment's max error)
                max_overall_error = max(max_overall_error, current_max_error)

                i = j - 1 # Next segment starts at j-1
                break 
                
            elif j == len(initial_keys) - 1:
                # Reached the very last point. Commit the segment (i -> j).
                x_end = initial_keys[j]
                y_end = data_map[x_end]
                
                dx = x_end - x_start
                if dx == 0:
                    P, O = np.nan, np.nan
                else:
                    P = (y_end - y_start) / dx
                    O = y_start - P * x_start
                    
                final_model[x_start] = [P, O, x_end]
                max_overall_error = max(max_overall_error, current_max_error)
                
                i = j # Loop terminates
                break
            
            j += 1 # Try to extend the segment further

    # Add the final NaN marker if the loop didn't explicitly add it
    last_key = initial_keys[-1]
    if last_key not in final_model:
        final_model[last_key] = [np.nan, np.nan, np.nan]

    return final_model, max_overall_error

# ==============================================================================
# 4. MAIN TRAINING AND PREDICTION FUNCTIONS
# ==============================================================================

def train_slrm(input_data_string: str, epsilon: float = EPSILON) -> Tuple[Dict[float, List[float]], List[Tuple[float, float]], float]:
    """
    Trains the Segmented Linear Regression Model (SLRM) from data.

    Args:
        input_data_string (str): Data points (X, Y) separated by lines.
        epsilon (float): Maximum error tolerance for Lossy compression.

    Returns:
        Tuple: (SLRM Model, Cleaned Original Points, Max Error Achieved)
    """
    global _prediction_cache 
    
    # 1. Cleaning and Sorting
    original_points = _clean_and_sort_data(input_data_string)
    
    if len(original_points) < 2:
        return {}, original_points, 0.0
        
    # 2. Lossless Compression (Geometric Invariance)
    initial_breakpoints_x = _lossless_compression(original_points)
    
    # 3. Lossy Compression (MRLS)
    final_model, max_error = _lossy_compression(initial_breakpoints_x, epsilon, original_points)
    
    # Clear the prediction cache when training a new model
    _prediction_cache = LRUCache(CACHE_SIZE)
    
    return final_model, original_points, max_error


def predict_slrm(x_in: float, slrm_model: Dict[float, List[float]], original_points: List[Tuple[float, float]]) -> Dict[str, Any]:
    """
    Predicts the Y value for an input X using the compressed SLRM model.
    """
    if not slrm_model or not original_points:
        return {'x_in': x_in, 'y_pred': np.nan, 'slope_P': np.nan, 'intercept_O': np.nan, 'cache_hit': False}

    # Attempt to get the prediction from the cache
    cached_result = _prediction_cache.get(x_in)
    if cached_result is not None:
        cached_result['cache_hit'] = True
        return cached_result
        
    # Get keys for valid segments (those with calculated P)
    segment_starts = sorted([x for x, segment in slrm_model.items() if not math.isnan(segment[0])])
    
    if not segment_starts:
        return {'x_in': x_in, 'y_pred': np.nan, 'slope_P': np.nan, 'intercept_O': np.nan, 'cache_hit': False}

    min_x = original_points[0][0]
    max_x = original_points[-1][0]
    
    active_key = None

    if x_in < min_x:
        # Extrapolation (Left): use the first segment
        active_key = segment_starts[0]
    elif x_in >= max_x:
        # Extrapolation (Right) or exact end point: use the last segment
        active_key = segment_starts[-1]
    else:
        # Interpolation: find the segment where x_start <= x_in < x_end
        for x_start in segment_starts:
            x_end = slrm_model[x_start][2] # X_end of the segment
            
            if x_in >= x_start and x_in < x_end:
                active_key = x_start
                break

    if active_key is None:
        # Should only happen if the data is a single point or unusual edge case not covered
        P, O = np.nan, np.nan
    else:
        P, O, _ = slrm_model[active_key]
        
    y_pred = x_in * P + O if not math.isnan(P) else np.nan
    
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

# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    import time
    
    # Data used in the interactive visualizer V5.8
    SAMPLE_DATA = """
1, 1
2, 1.5
3, 1.7
4, 3.5
5, 5
6, 4.8
7, 4.5
8, 4.3
9, 4.1
10, 4.2
11, 4.3
12, 4.6
13, 5.5
14, 7
15, 8.5
"""
    
    print("--- SLRM Training and Prediction Example (V5.8) ---")
    
    # --------------------------------------------------------------------------
    # TEST 1: Lossy Compression (epsilon=0.5)
    # --------------------------------------------------------------------------
    epsilon_test = 0.5
    print(f"\n[TEST 1] Training with Epsilon = {epsilon_test:.6f}")
    
    start_time = time.time()
    model, points, max_error = train_slrm(SAMPLE_DATA, epsilon_test)
    training_duration = time.time() - start_time
    
    segment_count = sum(1 for P, O, X_end in model.values() if not math.isnan(P))
    breakpoint_count = segment_count + 1 if segment_count > 0 else 0
    
    print(f"Time Taken: {training_duration:.4f} seconds.")
    print(f"Original Points: {len(points)}")
    print(f"Final Breakpoints: {breakpoint_count}")
    print(f"Segments Generated: {segment_count}")
    print(f"Max Error Achieved: {max_error:.7f}")
    
    print("\nModel Result (X_start: [P, O, X_end]):")
    for x_start, segment in model.items():
        if not math.isnan(segment[0]):
            print(f"  {x_start:+.2f}: P={segment[0]:+.4f}, O={segment[1]:+.4f}, X_end={segment[2]:+.2f}")
    
    # Prediction Test
    X_TEST_VALUES = [0.0, 5.5, 9.5, 15.0, 16.0]
    print("\nPrediction Test:")
    for x in X_TEST_VALUES:
        result = predict_slrm(x, model, points)
        print(f"Predict X={result['x_in']:+.2f} | Y={result['y_pred']:+.6f} | Active Segment P={result['slope_P']:+.4f}")
    
    # --------------------------------------------------------------------------
    # TEST 2: Lossless Compression (epsilon=0)
    # --------------------------------------------------------------------------
    epsilon_zero = 0.0
    print(f"\n[TEST 2] Training with Epsilon = {epsilon_zero:.6f} (Enforcing Geometric Invariance)")
    
    model_zero, _, max_error_zero = train_slrm(SAMPLE_DATA, epsilon_zero)
    
    segment_count_zero = sum(1 for P, O, X_end in model_zero.values() if not math.isnan(P))
    
    print(f"Segments Generated: {segment_count_zero}")
    print(f"Max Error Achieved: {max_error_zero:.7f} (Should be near zero)")

    # Cache Test
    x_cache_test = 5.5
    predict_slrm(x_cache_test, model, points) # First call (miss)
    cache_result = predict_slrm(x_cache_test, model, points) # Second call (hit)
    print(f"\n[INFO] Cache Test (X={x_cache_test}): Hit={cache_result['cache_hit']}, Y={cache_result['y_pred']:+.6f}")
    print(f"[INFO] LRU Cache Status (Size: {len(_prediction_cache.cache)}/{CACHE_SIZE})")
