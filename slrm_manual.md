# SLRM-LOGOS Technical Manual (V5.10b)

## Segmented Linear Regression Model (SLRM) Core Architecture

The Segmented Linear Regression Model (SLRM), driven by the deterministic Logos Core algorithm, is a fast and efficient solution for the compression and modeling of one-dimensional data (Time Series, calibration curves, etc.).

It utilizes the **Minimum Required Line Segments (MRLS)** technique to represent datasets with the smallest possible number of linear segments, meeting a user-defined error tolerance ($\epsilon$).

---

### Key Features (SLRM V5.10b)

* **Verification and Robustness:** Versión V5.10b verificada y probada con correcciones críticas en el manejo del error máximo en compresión Lossy.
* **Speed:** The core compression process is non-iterative and deterministic, relying on NumPy for optimal training speed, making it an **Instant Training** model.
* **Compression (Lossless/Lossy):** Implements both exact geometric compression (`_lossless_compression`) and tolerance-based compression (`_lossy_compression`).
* **Prediction Optimized:** Uses an **LRU (Least Recently Used)** prediction cache for dramatically accelerating real-time inference of repeated or nearby points.
* **Data Integrity:** Includes an initial data purification stage (`_clean_and_sort_data`) to handle duplicates (averaging Y values for the same X) and sort the data by X.
* **Full Interpretability:** The model is a "transparent box," storing explicit knowledge (Slope $P$ and Intercept $O$) for every segment.

### Training Process Stages (V5.10b)

The `train_slrm` function executes a three-stage compression pipeline:

1.  **Purification & Sorting (`_clean_and_sort_data`):**
    * Converts input data into a clean, sorted sequence of points.
    * Handles X duplicates by averaging their corresponding Y values.
2.  **Lossless Compression (`_lossless_compression`):**
    * **Geometric Invariance:** Identifica y elimina todos los puntos intermedios que son geométricamente colineales (puntos críticos de quiebre).
    * El resultado es la **Base de Breakpoints** que preserva la precisión original.
3.  **Lossy Compression (MRLS, `_lossy_compression`):**
    * Aplica el criterio de error humano $\epsilon$ sobre la Base de Breakpoints.
    * Utiliza la técnica de **Segmentos de Línea Mínimos Requeridos (MRLS)** para extender segmentos hasta que el error de interpolación exceda $\epsilon$.
    * **El modelo final** ($P$, $O$, $X_{end}$) es generado y el máximo error real ($Error_{max}$) es retornado.

### Usage Quick Guide

The model is trained and used via the `train_slrm` and `predict_slrm` functions in the `slrm-logos.py` file.

#### Requirements:

You need to have Python and the NumPy library installed.
```bash
pip install numpy
```

#### 1. Training (V5.10b)

Define your input data and the tolerance ($\epsilon$).
```python
from slrm_logos import train_slrm

INPUT_DATA = [
    [-8.00, -4.00], [4.00, 5.0], [6.00, 18.0], [10.00, 27.00]
    # ... more points
]

# Train the model. It returns the final model dictionary, the original points, and the max error achieved.
final_model, original_points, max_error_achieved = train_slrm(INPUT_DATA, epsilon=0.03) 

# El resultado 'final_model' es un diccionario {X_start: [P, O, X_end]} 
# que define explícitamente cada segmento.
```

#### 2. Prediction (V5.10b)

Use el diccionario del modelo (`final_model`) para obtener predicciones para cualquier punto X. El caché LRU se gestiona internamente en la función `predict_slrm`.
```python
from slrm_logos import predict_slrm

x_test = 4.5
result = predict_slrm(x_test, final_model, original_points)

print(f"Prediction for X={result['x_in']}: Y={result['y_pred']:.4f}")
print(f"Active Segment: P={result['slope_P']:.4f}, O={result['intercept_O']:.4f}")
print(f"Cache Hit: {result['cache_hit']}") # Muestra si la predicción vino del caché
```

### Resources and Reference Scripts:

* [📧 Main Project Repository](https://github.com/akinetic/neural-network/)
* [💻 Production SLRM Script (slrm-logos.py)](https://github.com/akinetic/neural-network/blob/main/slrm-logos.py)
* [📄 Performance and Efficiency Report (slrm\_performance\_report.md)](https://github.com/akinetic/neural-network/blob/main/slrm_performance_report.md)
