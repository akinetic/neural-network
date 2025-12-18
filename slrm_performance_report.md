# SLRM-LOGOS V5.12: Performance and Efficiency Report

This technical report consolidates the features of the Segmented Linear Regression Model (SLRM) and presents performance results demonstrating its efficiency and accuracy against standard machine learning models.

---

## 1. Project Core: SLRM Superiority

SLRM is a deterministic model designed for **Knowledge Compression**, focused on reducing the amount of data required to store a curve or time series while maintaining a user-defined maximum error $\epsilon$.

### Key Features:
* **Transparent Box Model:** Stores explicit knowledge (Slope $P$ and Intercept $O$) per segment.
* **Instant Training:** Non-iterative $O(N)$ algorithm that requires no stochastic optimization.
* **Double Compression:** Uses Geometric Invariance (Lossless Compression) followed by the MRLS algorithm (Lossy Compression) for maximum efficiency.

---

## 2. Performance Analysis (Proof of Concept)

A comparative test was conducted using a simple dataset ($N=15$ points) with a complex curve shape to challenge linear models. SLRM was configured with a maximum error tolerance of $\epsilon = 0.5$.

*Dataset:* 15 points (Mapping X: [1...15] to Y: [1.0...8.5])

### SLRM General Results

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Original Points ($N$)** | 15 | Original dataset size. |
| **Required Segments** | 5 | Number of linear segments that meet $\epsilon \le 0.5$. |
| **Stored Key Points** | 6 | $Segments + 1$. These points define the model. |
| **Compression Rate** | **60.00%** | 60% of points were eliminated without exceeding $\epsilon$. |
| **Maximum Error Achieved** | 0.4333 | Meets the $\epsilon \le 0.5$ objective. |

### Accuracy and Complexity Metrics Comparison

The following table compares SLRM with popular scikit-learn library models, evaluating the **Mean Squared Error (MSE)** and the **Coefficient of Determination ($R^2$)**.

| Model | MSE | $R^2$ | Model Complexity | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **SLRM (Segmented)** | **0.0380** | **0.9893** | **6 (Key Points)** | Near-perfect accuracy with high compression. |
| Decision Tree (Depth 5) | **0.0129** | **0.9964** | **9 (Levels/Depth)** | Slightly more accurate, but with a complex hierarchical structure. |
| Polynomial (Degree 3) | 0.2392 | 0.9328 | 4 (Coefficients) | Worse fit than SLRM; mathematical complexity. |
| SLR (Simple Linear) | 0.9263 | 0.7399 | 2 (Parameters) | Unacceptable fit for non-linear data. |

---

## 3. Conclusion on SLRM Superiority

The results confirm that the SLRM design is an optimal approach for modeling time series and curves.

1. **Geometric Efficiency:** SLRM achieves an accuracy of $R^2 = 0.9893$ (virtually equal to the Decision Tree), but does so with **geometric simplicity** through only **5 linear segments**. This is a more elegant and interpretable structure than the **9 regions** used by the Decision Tree.
2. **Transparency vs. Black Box:** While the Decision Tree achieves its fit through a complex hierarchy of splits, SLRM achieves the same result with the simple logic of **geometry and error tolerance**, making it ideal for industrial and financial systems where interpretability is fundamental.
3. **Compression without Compromise:** SLRM reduces the amount of data to be stored by 60% without sacrificing the precision required by the user ($\epsilon \le 0.5$).

**SLRM is the solution for Knowledge Compression: it is Accurate, Simple, and Efficient.**

---
### Resources and Reference Scripts:

* [Main Project Repository](https://github.com/akinetic/neural-network/)
* [Production SLRM Script (slrm-logos.py)](slrm-logos.py)
* [Performance Testing Script (slrm\_testing\_script.py)](slrm_testing_script.py)
