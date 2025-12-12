# Segmented Neural Architecture (SNA)

> **This project introduces the Segmented Neural Architecture (SNA), a model characterized by high operational efficiency and extremely high data compaction. Unlike Artificial Neural Networks (ANNs), the SNA relies on a modular, autonomous structure that enables "Instantaneous Training," "Hot Swapping" without collateral effects, and the activation of non-linear functions. Its "Multimodal Compression" core generates an optimized dictionary of weights and biases, ensuring full traceability (Transparent Box) and controlled precision loss ($\epsilon$).**

---

## I. The Input Set

We assume that a DataFrame (Input Set) is a Database, where each input $X$ is associated with a single output $Y$ (i.e., it is a function from $X$ to $Y$). Furthermore, the dataset contains no duplicate entries, which guarantees the uniqueness of the model's control points (the "no Density or Depth" property).

In this project, our toy Dataframe (Input Set) is as follows:

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

Where the first value ($X$) represents the input and the second value ($Y$) represents the associated output. Note that the DataFrame is presented unsorted to simulate unprocessed data input.

---

## II. Generating the Base Dictionary (Instantaneous Training)

The creation of the Base Dictionary simply consists of sorting the $(X, Y)$ pairs of the DataFrame from least to greatest, using $X$ as the sorting key.

In our toy model, the Base Dictionary is as follows:

```
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

The Base Dictionary is already functional (although not yet optimized or compressed) as it allows for inference through two procedures:

1.  **Direct Search:** If the input $X$ matches a value $X_n$ in the Base Dictionary, the output is the $Y$ value associated with $X_n$.
2.  **Linear Inference (Interpolation):** If the input $X$ does not match a known point and falls within the Dictionary limits, the output $Y$ is given by the following equation:

$$Y = (X - X_1) \cdot \left[ \frac{Y_2 - Y_1}{X_2 - X_1} \right] + Y_1$$

Where $X_1$ is the nearest value less than $X$, $X_2$ is the nearest value greater than $X$, and $Y_1, Y_2$ are their associated outputs in the Base Dictionary.

For example, in our toy model, if the input $X$ is $5$, the output $Y$ is given by:

$$Y = (5 - 4) \cdot \left[ \frac{18 - 10}{6 - 4} \right] + 10$$

$$Y = 14$$

---

## III. Dictionary Transformation (Weights and Biases)

Optimizing the Base Dictionary consists of making the Base Dictionary more functional according to one or more criteria.

The initial optimization criterion consists of transforming the data points $(X, Y)$ into the definitions of the linear segments. This is achieved by calculating the Slope ($m$, or Weight) and the Y-Intercept ($b$, or Bias) for each segment defined by pairs of contiguous points.

In our example, the results are:

```
[-8.00,-4.00]
				(-1.00,-12.0) // Slope, Y-Intercept for segment [-8.00, -6.00]
[-6.00,-6.00]
				(-0.01,-6.06) // Slope, Y-Intercept for segment [-6.00, -5.00]
[-5.00,-6.01]
				(+0.01,-5.96) // ...
[-4.00,-6.00]
				(+1.00,-2.00) // ...
[-2.00,-4.00]
				(+2.00,+0.00)
[+0.00,+0.00]
				(+2.00,+0.00)
[+2.00,+4.00]
				(+3.00,-2.00)
[+3.00,+7.00]
				(+3.00,-2.00)
[+4.00,+10.0]
				(+4.00,-6.00)
[+6.00,+18.0]
```

Each pair $(m, b)$ is associated with the starting point $X$ of the segment. In this way, the Optimized Dictionary stores the "Activation" of each Neuron (point $X$) in the form of its segment parameters $(m, b)$, eliminating the need to store the original $Y$ value.

The Optimized Dictionary is as follows:

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
*(Correction: The point [+6.00] is removed because it does not start a new segment.)*

This new Optimized Dictionary is the basis for compression. It is also functional, just like the Base Dictionary, and ready for use.

For any input $X$ within the Dictionary limits, the Inference Procedure states that the output $Y$ is given by the SNA Master Equation:

$$Y = X \cdot \text{Slope}(\text{Weight}) + \text{Y-Intercept}(\text{Bias})$$

Where the $\text{Slope}(\text{Weight})$ and the $\text{Y-Intercept}(\text{Bias})$ are those associated with the nearest $X_n$ less than or equal to $X$ in the Optimized Dictionary (the start of the current segment).

For example, in our toy model, if the input $X$ is $5$, the output $Y$ is given by:

$$Y = 5 \cdot 4 - 6$$

$$Y = 14$$

---

## IV. Redundancy Compression (Lossless)

Redundancy Compression eliminates redundant information from the Optimized Dictionary. This stage formalizes the network, leaving only the points where the Slope undergoes a change.

Basically, if the line segment starting at point $X_{n+1}$ has the same Slope ($m$) and Y-Intercept ($b$) as the segment starting at the previous point $X_n$ (i.e., there is **Slope Continuity**), then $X_{n+1}$ is a redundant point. The information for this point is already implicitly contained by the neuron $X_n$, so it is removed.

In our toy model, the Redundant Data or Neurons are: `[+0.00] (+2.00,+0.00)` and `[+3.00] (+3.00,-2.00)`. The Lossless Compressed Optimized Dictionary is as follows:

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

---

## V. Algorithmic Compression (Controlled Loss $\epsilon$)

Algorithmic Compression allows for further reduction in dictionary size by eliminating Neurons whose contribution to the global error is below a predefined threshold.

When a Data point or Neuron is locally irrelevant depends on the **Error Tolerance ($\epsilon$)** desired for the Segmented Neural Architecture. Essentially, Neuron $X_n$ is removed if, when substituted by an interpolation between its neighbors, the maximum absolute error does not exceed the established tolerance.

In our toy model, an Error Tolerance ($\epsilon$) of $0.03$ is set. This means that any point whose absolute deviation is less than or equal to $\epsilon$ is considered "non-relevant information" and can be removed.

Therefore, in our toy model, the Redundant Data or Neuron is: `[-5.00] (+0.01,-5.96)`, resulting in the Optimized and Compressed Dictionary of our toy model with loss of non-relevant information as follows:

```
// Optimized Dictionary (Final Lossy Compression)
[-8.00] (-1.00,-12.0)
[-6.00] (+0.00,-6.00)
[-4.00] (+1.00,-2.00)
[-2.00] (+2.00,+0.00)
[+2.00] (+3.00,-2.00)
[+4.00] (+4.00,-6.00)
```

Now, in the Optimized and Compressed Dictionary of our toy model for $X$ equal to $-5$, the result is $-6$, yielding only a difference of $0.01$ with respect to the actual data from the DataFrame or Base Dictionary ($-6.01$).

---

## VI. Non-Linear Activation and Multimodal Compression

This section covers advanced compression methods that allow the SNA to move beyond linear segments. Thanks to the **local and autonomous nature of each neuron in the Optimized Dictionary**, **functional substitution** is possible, where a set of segments is locally replaced by a higher-order function, without affecting other areas of the network.

The goals of this compression are:

### 1. Non-Linear Substitution without Loss of Information

Consists of identifying a group of contiguous linear neurons and substituting them with a single higher-order function (e.g., quadratic, cubic, exponential, etc.) that passes **exactly** through all breakpoint points. The result is a much more compact Dictionary without introducing additional error.

* **Example:** Substitute a stretch of 5 linear neurons with a single parabolic function ($Y = aX^2 + bX + c$) that perfectly fits the points.

### 2. Non-Linear Substitution with Non-Relevant Loss

The most common procedure. The higher-order function substitutes the set of linear neurons if the maximum absolute error between the non-linear function and the original data does not exceed the Error Tolerance ($\epsilon$). This allows for extreme compaction.

* **Local Application:** This technique can be applied only to specific sectors of the network (e.g., between $X=-1$ and $X=+12$), leaving the rest of the network with linear functions if they are locally more efficient or if a change is not necessary.

### 3. Local Procedure Adjustments

In addition to functional substitution, fine adjustments can be applied to optimize the transition between neurons and the inference procedure. This includes using a variable $\epsilon$ (stricter in critical zones, more flexible in flat zones) for better local efficiency.

---

## VII. Boundary Management and Projection

This section defines how the SNA manages inputs $X$ that fall outside its known training range [$X_{\min}, X_{\max}$].

### 1. Bounding and Limit Imposition (Default Restriction)

By default, the system can choose to **impose its limits** and reject or bound the input.

* If $X < X_{\min}$, the output $Y$ is fixed to the $Y_{\min}$ value of the first known point.
* If $X > X_{\max}$, the output $Y$ is fixed to the $Y_{\max}$ value of the last known point.

### 2. Segmental Projection (Standard Extrapolation)

If the system agrees to perform projections, the neuron associated with $X_{\min}$ undergoes a **Boundary Transformation**.

* **Lower Extreme ($X \rightarrow -\infty$):** The neuron associated with $X_{\min}$ (e.g., $X = -8.00$) becomes the anchor of the initial segment and **functionally represents the limit $X = -\infty$**. This ensures that any input $X$ infinitely smaller than $X_{\min}$ (e.g., $-30$) correctly activates this first neuron, complying with the search rule "find the nearest $X_n$ less than or equal to $X$."
* **Upper Extreme:** The point $X_{\max}$ (the largest value in the original DataFrame) **disappears from the Optimized Dictionary** during Section III, as it does not initiate a new segment, but rather marks the end of the previous segment. Therefore, the last remaining neuron defines the segment that is projected towards $+\infty$.

The output $Y$ is calculated by applying the SNA Master Equation using the Weights ($m$) and Biases ($b$) of the active boundary neuron, resulting in a **short-distance**, controlled, and consistent projection.

### 3. Zonal or Global Projection (Metaprogression)

The SNA has the capacity to analyze the global behavior of the Slopes ($m$) and Biases ($b$) near the limits to detect higher-order patterns (metaprogression).

* **Application:** If there is a well-marked progression in the weights ($m=2, m=3, m=4$), the system can project the next Weight ($m=5$) and its associated Bias to extend the Dictionary. This projection uses information from the **network pattern** as a whole (zonal) rather than relying solely on the last segment, offering potentially more accurate **long-distance** projection.

---

## VIII. Operational and Architectural Properties

+ **Modularity and Neural Autonomy (Zero Collateral Effect):** The distinctive feature of the SNA is that each neuron or segment operates locally and autonomously. This allows a specific sector of the Optimized Dictionary to be modified, optimized, or trained without the need to re-evaluate or alter the rest of the network—something impossible to efficiently replicate in traditional neural architectures (where a weight change affects the entire network).

+ **Hot Swapping:** Derived from modularity, the Optimized Dictionary can be locally modified, updated, or re-trained **in real-time** (hot), without interrupting the inference operations of the rest of the network. This guarantees extremely high availability and operational flexibility of the model.

+ **Instantaneous Training:** Since the SNA's "training" consists purely of sorting the DataFrame and directly calculating weights and biases (Sections II and III), the model lacks the iterative backpropagation phase, resulting in almost instantaneous training speed.

+ **Interpretability (Transparent Box Model):** The Optimized Dictionary stores explicit knowledge. Each neuron represents a functional segment with fixed parameters (Weights and Biases), allowing for complete traceability of inference, unlike "black box" models.

+ It could be visualized that each Neuron is responsible for one sector of the Network and that its functioning does not alter the Network beyond that sector.

+ Furthermore, local Neurons (closest to each other) can be associated and/or unified to make the Neural Network more compact and efficient.

+ The ideal scenario is for the Optimized Dictionary to be as compact as possible, for the input/output speed to be as fast as possible with the lowest possible cost, and finally, for the Neural Network to evolve in the most flexible/simple way possible.

---

## IX. Bibliography

The following conceptual references inspire or contrast with the fundamental principles of the Segmented Neural Architecture (SNA):

1.  **Segmented Regression and Curve Fitting:** Works on the approximation of complex functions using piecewise defined regression models.
2.  **Model Quantization and Compression:** Techniques aimed at reducing the size of neural models for implementation on memory-constrained hardware.
3.  **White Box Models (Interpretability):** Studies on the traceability and understanding of a prediction model's decisions, contrasting with the "black box" of ANNs.
4.  **Modularity and Decoupled Architectures:** Software design principles that allow for local modification without collateral effects in complex systems.
5.  **Training Processes and Optimization Algorithms:** Concepts related to training efficiency, contrasting the SNA's Instantaneous Training with the iterative nature of Backpropagation.
6.  **Handling Sparse Data and Outliers:** Techniques to ensure model robustness against isolated points or inconsistencies in input data.
7.  **Associative Memory Systems:** The Optimized Dictionary as a form of efficient data structure for fast pattern storage and retrieval.
8.  **Fault-Tolerant System Design:** Principles that allow components to be modified or updated (Hot Swapping) without interrupting global service.
9.  **Time Series Theory:** Works on detecting progression patterns (Metaprogression) to perform more accurate long-range extrapolations.

---

Alex Kinetic

MIT LICENSE

---
