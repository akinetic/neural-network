# Segmented Linear Regression Model (SLRM)

> This project implements the Segmented Linear Regression Model (SLRM), an alternative to traditional Artificial Neural Networks (ANNs). The SLRM models datasets with piecewise linear functions, using a **neural compression** process to reduce complexity without compromising precision beyond a user-defined tolerance.

The core of the solution is the compression algorithm, which transforms an unordered dataset (`DataFrame` / `X, Y`) into a final, highly optimized dictionary, ready for prediction.

## Project Structure

* **`slrm-logos.py`**: Contains the complete implementation of the training process (Creation, Optimization, Compression) and the prediction function (`predict`). This code generates the final SLRM dictionary consumed by the web application.
* **`index.html`**: Implementation of the visualization using D3.js and Vanilla JavaScript, which shows the dataset, the SLRM prediction curve (the piecewise linear function), and allows real-time interaction with the prediction function.

---

## 🧠 SLRM Architecture: The Training Process (Compression)

SLRM training is achieved through four main sections, implemented sequentially in `slrm-logos.py`:

### 1. Base Dictionary Creation (Section I and II)

The SLRM is a **non-iterative** model (Instant Training). The "training" begins by sorting the input dataset (`X, Y`) by the lowest to the highest value of `X`. This sorting transforms the initial `DataFrame` into the fundamental structure of the SLRM: a dictionary where each point `(X, Y)` is indexed by its `X` value.

**Input Set Example:**

To demonstrate the process, we use the following unordered dataset (Input $X$, Output $Y$):

```
[-6.00,-6.00]
[+2.00,+4.00]
[-8.00,-4.00]
[+0.00,+0.00]
[+4.00,+10.0]
[-4.00,-6.00]
[+6.00,+18.0]
[-5.00,-6.01]
[+3.00,+7.00]
[-2.00,-4.00]
```
Once sorted by $X$, this becomes the **Base Dictionary**:

```
// Base Dictionary (Sorted by X)
[-8.00,-4.00]
[-6.00,-6.00]
[-5.00,-6.01]
[-4.00,-6.00]
[-2.00,-4.00]
[+0.00,+0.00]
[+2.00,+4.00]
[+3.00,+7.00]
[+4.00,+10.0]
[+6.00,+18.0]
```

### 2. Optimization (Section III)

Based on the sorted base dictionary, the linear function connecting each pair of adjacent points $(x_1, y_1)$ and $(x_2, y_2)$ is calculated. This step transforms the data $(X, Y)$ into the segment parameters:

* **Slope (P)**: Represents the **Weight** (`W`) of the segment.
    $$P = \frac{y_2 - y_1}{x_2 - x_1}$$
* **Y-Intercept (O)**: Represents the **Bias** (`B`) of the segment.
    $$O = y_1 - P \cdot x_1$$

The result is an **Optimized Dictionary** where each key $X_n$ (the start of the segment) stores the tuple $(P, O)$. This is the explicit knowledge of the model.

**Optimized Dictionary Example (Weights and Biases):**

```
// Optimized Dictionary (Weights and Biases)
[-8.00] (-1.00,-12.0)
[-6.00] (-0.01,-6.06)
[-5.00] (+0.01,-5.96)
[-4.00] (+1.00,-2.00)
[-2.00] (+2.00,+0.00)
[+0.00] (+2.00,+0.00)
[+2.00] (+3.00,-2.00)
[+3.00] (+3.00,-2.00)
[+4.00] (+4.00,-6.00)
```

### 3. Lossless Compression (Geometric Invariance - Section IV)

This step eliminates the geometric redundancy of the model. If three consecutive points $(X_{n-1}, X_n, X_{n+1})$ lie on the same straight line, the intermediate point $X_n$ is considered redundant.

* **Criterion:** If $\text{Slope}(X_{n-1}) \approx \text{Slope}(X_n)$, the point $X_n$ is removed from the dictionary.
* **Result:** Intermediate "neurons" that do not contribute to a change in the curve's direction are eliminated, achieving **lossless** compression of the dictionary's geometric information.

**Lossless Compression Example:**

`[+0.00]` and `[+3.00]` are removed due to Slope redundancy, resulting in:
```
// Optimized Dictionary (Lossless Compression)
[-8.00] (-1.00,-12.0)
[-6.00] (-0.01,-6.06)
[-5.00] (+0.01,-5.96)
[-4.00] (+1.00,-2.00)
[-2.00] (+2.00,+0.00)
[+2.00] (+3.00,-2.00)
[+4.00] (+4.00,-6.00)
```

### 4. Lossy Compression (Human Criterion - Section V)

This is the step for maximum compression, where a **human criterion** (the tolerance $\epsilon$) is applied to eliminate points whose contribution to the global error is below a predefined threshold.

* **Tolerance ($\epsilon$):** An acceptable maximum error value (e.g., $0.03$).
* **Permanence Criterion:** The point $X_{\text{current}}$ is considered **Relevant** and is kept if the absolute error when interpolating between its neighbors is greater than $\epsilon$.

$$\text{Error} = | Y_{\text{true}} - Y_{\text{hat}} |$$

If $\text{Error} > \epsilon$, the point is kept. If $\text{Error} \le \epsilon$, it is removed (lossy compression).

**Final Lossy Compression Example ($\epsilon=0.03$):**

`[-5.00]` is removed as its error is $0.01 \le 0.03$ when interpolated between `[-6.00]` and `[-4.00]`.

```
// Optimized Dictionary (Final Lossy Compression)
[-8.00] (-1.00,-12.0)
[-6.00] (+0.00,-6.00) // Adjusted parameters due to interpolation
[-4.00] (+1.00,-2.00)
[-2.00] (+2.00,+0.00)
[+2.00] (+3.00,-2.00)
[+4.00] (+4.00,-6.00)
```

---

## 5. SLRM Extensions and Operational Properties

The modular nature of the SLRM segments provides operational properties that distinguish it from iterative neural network models:

### 5.1 Modularity and Hot Swapping
Since each segment is autonomous and does not interact with the weights of other segments, the SLRM allows for **Hot Swapping**. This means that a specific sector of the dictionary can be updated, optimized, or new data added **in real-time**, without interrupting the inference operation of the rest of the network.

### 5.2 Non-Linear Activation and Multimodal Compression
The compression process can be extended to locally replace a set of multiple linear segments with a single higher-order function (e.g., quadratic or exponential), provided the substitution error remains within tolerance ($\epsilon$). This generates **Multimodal Compression** and further compacts the architecture.

### 5.3 Transparent Box (Full Interpretability)
The SLRM is a "transparent box" model. It stores knowledge explicitly (Slope $P$ and Y-Intercept $O$ for each segment). This allows for full traceability of every prediction and is ideal for environments requiring high interpretability and auditing.

---

## 🎯 Prediction and Generalization (Section VII)

The `predict(X)` function uses the final, compressed SLRM dictionary.

1. **Active Segment Search:** For a new input $X$, the model finds the key $X_n$ that is closest to and less than or equal to $X$ ($X_n \le X$). This $X_n$ defines the active linear segment $(P, O)$.
2. **Master Equation:** The linear formula is applied to obtain the predicted output $Y_{\text{predicted}}$.

$$Y_{\text{predicted}} = X \cdot P + O$$

### Generalization (Extrapolation)

The SLRM handles extrapolation outside the training limits in the following way:

* **Segmental Extrapolation (Short Distance):** The boundary linear segment (the first or the last) is extended to infinity, using the parameters $(P, O)$ of the segment closest to the limit.
* **Zonal Projection (Advanced Metaprogression):** In advanced models, the SLRM can analyze the progression of the Weights ($P$) and Biases ($O$) near the limits to detect higher-order patterns. This allows for projecting the next segment based on the **global pattern of the network**, offering potentially more accurate long-distance extrapolation.

---

## IX. Conceptual Bibliography

The following conceptual references inspire or contrast with the fundamental principles of the Segmented Linear Regression Model (SLRM):

1.  Segmented Regression and Curve Fitting: Works on approximating complex functions using piecewise defined regression models.
2.  Quantization and Model Compression: Techniques aimed at reducing the size of neural models for implementation on memory-constrained hardware.
3.  White Box Models (Interpretability): Studies on the traceability and understanding of a prediction model's decisions.
4.  Modularity and Decoupled Architectures: Software design principles that allow for local modification without collateral effects.
5.  Training Processes and Optimization Algorithms: Concepts related to training efficiency and non-iterative training.
6.  Handling Sparse Data and Outliers: Techniques to ensure model robustness against isolated points or inconsistencies in input data.
7.  Associative Memory Systems: The Optimized Dictionary as a form of efficient data structure for fast pattern storage and retrieval.
8.  Fault-Tolerant System Design: Principles that allow components to be modified or updated (Hot Swapping) without interrupting global service.
9.  Time Series Theory: Works on detecting progression patterns (Metaprogression) to perform more accurate long-range extrapolations.

---

Project Files

[readme-demo](index.html) : The web application for visualization.

[slrm-logos.py](slrm-logos.py) : The main source code containing the training and prediction logic (V5.10b).

[slrm_manual.html](slrm_manual.html)  : (Content in this file): This technical manual (V5.10b).

[slrm_visualizer.html](slrm_visualizer.html) : The web application for visualization (Using previous logic - Not updated).

---

Authors

Alex Kinetic and Logos

Project under MIT License


