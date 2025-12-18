# SLRM-LOGOS Technical Manual (V5.12)

## Segmented Linear Regression Model (SLRM) Core Architecture

The Segmented Linear Regression Model (SLRM), driven by the deterministic Logos Core algorithm, is a fast and efficient solution for the compression and modeling of one-dimensional data (Time Series, calibration curves, etc.).

It utilizes the **Minimum Required Line Segments (MRLS)** technique to represent datasets with the smallest possible number of linear segments, meeting a user-defined error tolerance ($\epsilon$).

---

### Key Features (SLRM V5.12)

* **Verification and Robustness:** Version V5.12 verified and tested with critical corrections in the handling of maximum error in Lossy compression.
* **Speed:** The core compression process is non-iterative and deterministic, relying on NumPy for optimal training speed, making it an **Instant Training** model.
* **Compression (Lossless/Lossy):** Implements both exact geometric compression (`_lossless_compression`) and tolerance-based compression (`_lossy_compression`).
* **Prediction Optimized:** Uses an **LRU (Least Recently Used)** prediction cache for dramatically accelerating real-time inference of repeated or nearby points.
* **Data Integrity:** Includes an initial data purification stage (`_clean_and_sort_data`) to handle duplicates (averaging Y values for the same X) and sort the data by X.
* **Full Interpretability:** The model is a "transparent box," storing explicit knowledge (Slope $P$ and Intercept $O$) for every segment.

### Training Process Stages (V5.12)

The `train_slrm` function executes a three-stage compression pipeline:

1.  **Purification & Sorting (`_clean_and_sort_data`):**
    * Converts input data into a clean, sorted sequence of points.
    * Handles X duplicates by averaging their corresponding Y values.
2.  **Lossless Compression (`_lossless_compression`):**
    * **Geometric Invariance:** Identifies and removes all intermediate points that are geometrically collinear (critical breakpoint points).
    * The result is the **Breakpoints Base** that preserves the original precision.
3.  **Lossy Compression (MRLS, `_lossy_compression`):**
    * Applies the human error criterion $\epsilon$ over the Breakpoints Base.
    * Utilizes the **Minimum Required Line Segments (MRLS)** technique to extend segments until the interpolation error exceeds $\epsilon$.
    * **The final model** ($P$, $O$, $X_{end}$) is generated and the actual maximum error ($Error_{max}$) is returned.

### Usage Quick Guide

The model is trained and used via the `train_slrm` and `predict_slrm` functions in the `slrm-logos.py` file.

#### Requirements:

You need to have Python and the NumPy library installed.
```bash
pip install numpy
```

#### 1. Training (V5.12)

Define your input data and the tolerance ($\epsilon$).
```python
from slrm_logos import train_slrm

INPUT_DATA = [
    [-8.00, -4.00], [4.00, 5.0], [6.00, 18.0], [10.00, 27.00]
    # ... more points
]

# Train the model. It returns the final model dictionary, the original points, and the max error achieved.
final_model, original_points, max_error_achieved = train_slrm(INPUT_DATA, epsilon=0.03) 

# The result 'final_model' is a dictionary {X_start: [P, O, X_end]} # that explicitly defines each segment.
```

#### 2. Prediction (V5.12)

Use the model dictionary (`final_model`) to get predictions for any X point. The LRU cache is managed internally in the `predict_slrm` function.
```python
from slrm_logos import predict_slrm

x_test = 4.5
result = predict_slrm(x_test, final_model, original_points)

print(f"Prediction for X={result['x_in']}: Y={result['y_pred']:.4f}")
print(f"Active Segment: P={result['slope_P']:.4f}, O={result['intercept_O']:.4f}")
print(f"Cache Hit: {result['cache_hit']}") # Shows if the prediction came from the cache
```

### Resources and Reference Scripts:

* [Main Project Repository](https://github.com/akinetic/neural-network/)
* [Production SLRM Script (slrm-logos.py)](slrm-logos.py)
* [Performance and Efficiency Report (slrm\_performance\_report.md)](slrm_performance_report.md)
