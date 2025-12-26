# Segmented Linear Regression Model (SLRM)

> This project implements the Segmented Linear Regression Model (SLRM), an alternative to traditional Artificial Neural Networks (ANNs). The SLRM models datasets with piecewise linear functions, using a **neural compression** process to reduce complexity without compromising precision beyond a user-defined tolerance.

The core of the solution is the compression algorithm, which transforms an unordered dataset (`DataFrame` / `X, Y`) into a final, highly optimized dictionary, ready for prediction.

## Project Structure

* **`slrm-logos.py`**: Contains the complete implementation of the training process (Creation, Optimization, Compression) and the prediction function (`predict`). This code generates the final SLRM dictionary consumed by the web application.
* **`index.html`**: Implementation of the visualization using D3.js and Vanilla JavaScript, which shows the dataset and the SLRM prediction curve (the piecewise linear function).
* **`slrm_to_relu.py`**: The bridge between SLRM and modern Deep Learning. This script converts the optimized SLRM dictionary into a single **Universal ReLU Equation**, demonstrating that SLRM is the geometric architect of the perfect Neural Network.

---

## SLRM Architecture: The Training Process (Compression)

SLRM training is achieved through four main sections, implemented sequentially in `slrm-logos.py`:

### 1. Base Dictionary Creation

The SLRM is a **non-iterative** model (Instant Training). The "training" begins by sorting the input dataset (`X, Y`) by the lowest to the highest value of `X`. This sorting transforms the initial `DataFrame` into the fundamental structure of the SLRM: a dictionary where each point `(X, Y)` is indexed by its `X` value.

**Input Set Example:**

To demonstrate the process, we use the following unordered dataset (Input $X$, Output $Y$):

```text
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

```text
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

---

### 2. Optimization

Based on the sorted base dictionary, the linear function connecting each pair of adjacent points $(x_1, y_1)$ and $(x_2, y_2)$ is calculated. This step transforms the data $(X, Y)$ into the segment parameters:

* **Slope (P)**: Represents the **Weight** (`W`) of the segment.
    $$P = \frac{y_2 - y_1}{x_2 - x_1}$$
* **Y-Intercept (O)**: Represents the **Bias** (`B`) of the segment.
    $$O = y_1 - P \cdot x_1$$

The result is an **Optimized Dictionary** where each key $X_n$ (the start of the segment) stores the tuple $(P, O)$. This is the explicit knowledge of the model.

**Optimized Dictionary Example (Weights and Biases):**

```text
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

---

### 3. Lossless Compression (Geometric Invariance)

This step eliminates the geometric redundancy of the model. If three consecutive points $(X_{n-1}, X_n, X_{n+1})$ lie on the same straight line, the intermediate point $X_n$ is considered redundant.

* **Criterion:** If $\text{Slope}(X_{n-1}) \approx \text{Slope}(X_n)$, the point $X_n$ is removed from the dictionary.
* **Result:** Intermediate "neurons" that do not contribute to a change in the curve's direction are eliminated, achieving **lossless** compression of the dictionary's geometric information.

**Lossless Compression Example:**

`[+0.00]` and `[+3.00]` are removed due to Slope redundancy, resulting in:

```text
// Optimized Dictionary (Lossless Compression)
[-8.00] (-1.00,-12.0)
[-6.00] (-0.01,-6.06)
[-5.00] (+0.01,-5.96)
[-4.00] (+1.00,-2.00)
[-2.00] (+2.00,+0.00)
[+2.00] (+3.00,-2.00)
[+4.00] (+4.00,-6.00)
```

---

### 4. Lossy Compression (Human Criterion)

This is the step for maximum compression, where a **human criterion** (the tolerance $\epsilon$) is applied to eliminate points whose contribution to the global error is below a predefined threshold.

* **Tolerance ($\epsilon$):** An acceptable maximum error value (e.g., $0.03$).
* **Permanence Criterion:** The point $X_{\text{current}}$ is considered **Relevant** and is kept if the absolute error when interpolating between its neighbors is greater than $\epsilon$.

$$\text{Error} = | Y_{\text{true}} - Y_{\text{hat}} |$$

If $\text{Error} > \epsilon$, the point is kept. If $\text{Error} \le \epsilon$, it is removed (lossy compression).

**Final Lossy Compression Example ($\epsilon=0.03$):**

`[-5.00]` is removed as its error is $0.01 \le 0.03$ when interpolated between `[-6.00]` and `[-4.00]`.

```text
// Optimized Dictionary (Final Lossy Compression)
[-8.00] (-1.00,-12.0)
[-6.00] (+0.00,-6.00) // Adjusted parameters due to interpolation
[-4.00] (+1.00,-2.00)
[-2.00] (+2.00,+0.00)
[+2.00] (+3.00,-2.00)
[+4.00] (+4.00,-6.00)
```

---

## 5. Prediction and Generalization

The `predict(X)` function uses the final, compressed SLRM dictionary to generate instant inferences through a "search and execute" architecture.

### 5.1 Inference Mechanism
1. **Search:** For a new input $X$, the model finds the key $X_n$ that is closest to and less than or equal to $X$ ($X_n \le X$) within the optimized dictionary.
2. **Execution:** The linear formula is applied using the Weight ($P$) and Bias ($O$) parameters stored at that key through the **Master Equation**:

$$Y_{\text{predicted}} = X \cdot P + O$$

### 5.2 Generalization (Extrapolation)
The SLRM handles data outside the training limits in two ways:
* **Segmental Extrapolation:** The boundary linear segment (the first or the last) is extended to infinity, maintaining its trajectory.
* **Zonal Projection:** (Optional) Analysis of the progression of Weights ($P$) and Biases ($O$) near the limits to project the next segment based on the global network pattern.

---

## 6. SLRM Superiority: Efficiency vs. Standard Models

While SLRM is fundamentally an architecture for **Knowledge Compression**, its performance in modeling complex non-linear data surpasses standard parametric models and demonstrates structural efficiency against complex hierarchical models like Decision Trees.

A comparative test was conducted against scikit-learn models using a challenging 15-point non-linear dataset ($\epsilon=0.5$).

### Performance and Complexity Metrics (Decision Sheet)

The results demonstrate that SLRM achieves near-perfect accuracy with the highest data compression, proving its structural superiority in terms of simplicity and interpretability.

| Model | $R^2$ (Coefficient of Determination) | Model Complexity | Compression Rate |
| :--- | :--- | :--- | :--- |
| **SLRM (Segmented)** | **0.9893** | **6 (Key Points/Segments)** | **60.00%** |
| Decision Tree (Depth 5) | **0.9964** | **9 (Leaf Nodes/Regions)** | 0% |
| Polynomial (Degree 3) | 0.9328 | 4 (Coefficients) | 0% |
| SLR (Simple Linear) | 0.7399 | 2 (Parameters) | 0% |

> **Conclusion:** SLRM achieves $R^2=0.9893$ with **60% data compression** using only **5 linear segments** (6 key points). The Decision Tree achieves similar accuracy ($R^2=0.9964$) but requires **9 regions** to do so, confirming SLRM's superior **geometric efficiency** and inherent simplicity. 

---

## 7. SLRM Extensions and Operational Properties

The modular nature of the SLRM segments provides operational properties that distinguish it from iterative neural network models:

### 7.1 Modularity and Hot Swapping
Since each segment is autonomous and does not interact with the weights of other segments, the SLRM allows for **Hot Swapping**. This means that a specific sector of the dictionary can be updated, optimized, or new data added **in real-time**, without interrupting the inference operation of the rest of the network.

### 7.2 Non-Linear Activation and Multimodal Compression
The compression process can be extended to locally replace a set of multiple linear segments with a single higher-order function (e.g., quadratic or exponential), provided the substitution error remains within tolerance ($\epsilon$). This generates **Multimodal Compression** and further compacts the architecture.

### 7.3 Transparent Box (Full Interpretability)
The SLRM is a "transparent box" model. It stores knowledge explicitly (Slope $P$ and Y-Intercept $O$ for each segment). This allows for full traceability of every prediction and is ideal for environments requiring high interpretability and auditing.

---

## 8. From SLRM to Universal ReLU Equation (The AI Bridge)

While traditional Artificial Neural Networks (ANNs) spend massive computational resources "learning" weights through iterative trial and error (Backpropagation), **SLRM deduces them geometrically**.

Using the `slrm_to_relu.py` module, the model translates its optimized linear segments into a single, continuous mathematical function using **ReLU** (Rectified Linear Units), the standard activation function of modern Deep Learning.

### 8.1 The "Magic" Equation

For any dataset, SLRM generates a modular equation in the following form:

$$y = (W_{base} \cdot x + B_{base}) + \sum W_i \cdot \max(0, x - P_i)$$

**Where:**

* **$W_{base}$ / $B_{base}$**: Initial slope and bias (the starting trajectory).
* **$P_i$**: The **Critical Point** (Breakpoint) where the data trend shifts.
* **$W_i$**: The **Slope Delta** (The exact weight adjustment required at that specific point).

### 8.2 Architectural Superiority

* **Zero-Shot Training:** We do not "train" neural weights; we calculate them with $100\%$ precision in milliseconds.
* **Semantic Neurons:** Unlike "Black-Box" models, every ReLU unit in this equation has a physical meaning: it represents a specific, traceable change in the data's behavior.
* **Energy Efficiency:** This approach replaces hours of GPU-intensive training with a single, elegant geometric calculation.

---

## 9. Installation and Usage

**SLRM-LOGOS** is designed to be extremely lightweight with zero external dependencies.

### Installation via NPM

```bash
npm install slrm-logos
```

### JavaScript Usage Example (Node.js)

```JavaScript
const { train_slrm, predict_slrm } = require('slrm-logos');

// 1. Training data (Format: "x, y")
const data = "1,2\n2,4\n3,8\n4,16";

// 2. Train the model with a tolerance (Epsilon) of 0.5
const { model, originalData, maxError } = train_slrm(data, 0.5);

// 3. Perform a prediction
const inputX = 2.5;
const prediction = predict_slrm(inputX, model, originalData);

console.log(`Prediction for X=${inputX}: Y=${prediction.y_pred}`);
console.log(`Model Max Error: ${maxError}`);
```

---

## Conceptual Bibliography

The following conceptual references inspire or contrast with the fundamental principles of the Segmented Linear Regression Model (SLRM):

1. **Segmented Regression and Curve Fitting:** Works on approximating complex functions using piecewise defined regression models.
2. **Quantization and Model Compression:** Techniques aimed at reducing the size of neural models for implementation on memory-constrained hardware.
3. **White Box Models (Interpretability):** Studies on the traceability and understanding of a prediction model's decisions.
4. **Modularity and Decoupled Architectures:** Software design principles that allow for local modification without collateral effects.
5. **Time Series Theory:** Works on detecting progression patterns (Metaprogression) to perform more accurate long-range extrapolations.

---

## Project Resources & Navigation

* **[Live Interactive Visualizer](https://akinetic.github.io/neural-network/)**
    *Use the web app to test the SLRM model and compression in real-time.*

* **[slrm-logos.py](slrm-logos.py)**
    *The core production engine containing the full training and compression logic.*

* **[slrm\_to\_relu.py](slrm_to_relu.py)**
    *Universal translator from Geometric Segments to Neural ReLU Equations.*

* **[Technical Manual](slrm_manual.md)**
    *Deep dive into the mathematical foundations and architecture.*

* **[Performance Report](slrm_performance_report.md)**
    *Detailed comparison vs. Scikit-Learn models (Accuracy & Efficiency).*

---

## Authors

* **Alex Kinetic**
* **Logos**

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

> *"Simplicity is the ultimate sophistication."* - Segmented Linear Regression Model (SLRM)
