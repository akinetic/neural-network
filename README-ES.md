# Modelo de Regresión Lineal Segmentada (MRLS)

Este proyecto implementa el Modelo de Regresión Lineal Segmentada (MRLS), una alternativa a las redes neuronales artificiales (RNA) tradicionales. El MRLS modela conjuntos de datos con funciones lineales a trozos, utilizando un proceso de **compresión neuronal** para reducir la complejidad sin comprometer la precisión más allá de una tolerancia definida por el usuario.

El núcleo de la solución es el algoritmo de compresión, que transforma un conjunto de datos desordenado (`DataFrame` / `X, Y`) en un diccionario final, altamente optimizado, listo para realizar predicciones.

## Estructura del Proyecto

* **`model.py`**: Contiene la implementación completa del proceso de entrenamiento (Creación, Optimización, Compresión) y la función de predicción (`predict`). Este código genera el diccionario MRLS final que se consume en la web.
* **`index.html`**: Implementación de la visualización en D3.js y JavaScript Vanilla, que muestra el conjunto de datos, la curva de predicción del MRLS (la función lineal a trozos) y permite interactuar con la función de predicción en tiempo real.

---

## 🧠 Arquitectura del MRLS: El Proceso de Entrenamiento (Compresión)

El entrenamiento del MRLS se logra a través de cuatro secciones principales, implementadas secuencialmente en `model.py`:

### 1. Creación del Diccionario Base (Sección I y II)

El MRLS es un modelo no iterativo. El "entrenamiento" comienza ordenando el conjunto de datos de entrada (`X, Y`) de menor a mayor valor de `X`. Esta ordenación transforma el `DataFrame` inicial en la estructura fundamental del MRLS: un diccionario donde cada punto `(X, Y)` está indexado por su valor `X`.

### 2. Optimización (Sección III)

A partir del diccionario base ordenado, se calcula la función lineal que conecta cada par de puntos adyacentes `(x1, y1)` y `(x2, y2)`.

* **Pendiente (P)**: Representa el **Peso** (`W`) del segmento.
    $$P = \frac{y_2 - y_1}{x_2 - x_1}$$
* **Ordenada al Origen (O)**: Representa el **Sesgo** (`B`) del segmento.
    $$O = y_1 - P \cdot x_1$$

El resultado es un diccionario optimizado donde cada clave `Xn` (excepto la última) almacena la tupla `(P, O)` que define el segmento que comienza en `Xn`.

### 3. Compresión sin Pérdida (Invarianza Geométrica - Sección IV)

Este paso elimina la redundancia geométrica del modelo. Si tres puntos consecutivos `(X_{n-1}, X_n, X_{n+1})` se encuentran sobre la misma línea recta (es decir, el segmento de $X_{n-1}$ tiene la misma Pendiente que el segmento de $X_n$), el punto intermedio $X_n$ es redundante.

* **Criterio:** Si $\text{Pendiente}(X_{n-1}) \approx \text{Pendiente}(X_n)$, se elimina el punto $X_n$.
* **Resultado:** Se eliminan "neuronas" intermedias que no contribuyen a un cambio en la dirección de la curva, logrando una compresión del diccionario **sin pérdida** de información geométrica.

### 4. Compresión con Pérdida (Criterio Humano - Sección V)

Este es el paso de mayor compresión, donde se aplica un **criterio humano** (la tolerancia $\epsilon$) para eliminar puntos cuya contribución al modelo es mínima.

* **Tolerancia ($\epsilon$):** Un valor de error máximo aceptable (por ejemplo, $0.03$).
* **Proceso:** El algoritmo intenta eliminar un punto $X_{\text{actual}}$ y "estirar" el segmento lineal anterior (`P_{prev}, O_{prev}`) hasta $X_{\text{actual}}$.
* **Criterio de Permanencia:** El punto $X_{\text{actual}}$ se considera **Relevante** y se mantiene si la predicción del segmento anterior extendido (`Y_{\text{hat}}`) genera un error absoluto superior a la tolerancia $\epsilon$ respecto al valor original (`Y_{\text{true}}`) en ese punto.

$$\text{Error} = | Y_{\text{true}} - Y_{\text{hat}} |$$

Si $\text{Error} > \epsilon$, el punto se mantiene. Si $\text{Error} \le \epsilon$, se elimina (compresión con pérdida).

## 🎯 Predicción y Generalización (Sección VII)

La función `predict(X)` utiliza el diccionario MRLS final y comprimido.

1.  **Búsqueda del Segmento Activo:** Para una nueva entrada $X$, el modelo encuentra la clave $X_n$ más próxima y menor o igual a $X$ ($X_n \le X$). Esta $X_n$ define el segmento lineal activo `(P, O)`.
2.  **Ecuación Maestra:** Se aplica la fórmula lineal para obtener la predicción $Y_{\text{predicha}}$.

$$Y_{\text{predicha}} = X \cdot P + O$$

### Generalización (Extrapolación)

El MRLS maneja la extrapolación fuera de los límites de entrenamiento (Sección VII) de la siguiente manera:

* **Extremo Menor:** Si $X$ es menor que el valor mínimo de entrenamiento ($X < X_{\text{min}}$), se extiende el primer segmento lineal (definido por $X_{\text{min}}$) al infinito negativo.
* **Extremo Mayor:** Si $X$ es mayor que el valor máximo de entrenamiento ($X > X_{\text{max}}$), se extiende el último segmento lineal válido (definido por $X_{\text{max-1}}$) al infinito positivo.
