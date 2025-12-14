# Modelo de Regresión Lineal Segmentada (MRLS)

> Este proyecto implementa el Modelo de Regresión Lineal Segmentada (MRLS), una alternativa a las redes neuronales artificiales (RNA) tradicionales. El MRLS modela conjuntos de datos con funciones lineales a trozos, utilizando un proceso de **compresión neuronal** para reducir la complejidad sin comprometer la precisión más allá de una tolerancia definida por el usuario.

El núcleo de la solución es el algoritmo de compresión, que transforma un conjunto de datos desordenado (`DataFrame` / `X, Y`) en un diccionario final, altamente optimizado, listo para realizar predicciones.

## Estructura del Proyecto

* **`mrls-logos.py`**: Contiene la implementación completa del proceso de entrenamiento (Creación, Optimización, Compresión) y la función de predicción (`predict`). Este código genera el diccionario MRLS final que se consume en la web.
* **`index.html`**: Implementación de la visualización en D3.js y JavaScript Vanilla, que muestra el conjunto de datos, la curva de predicción del MRLS (la función lineal a trozos) y permite interactuar con la función de predicción en tiempo real.

---

## 🧠 Arquitectura del MRLS: El Proceso de Entrenamiento (Compresión)

El entrenamiento del MRLS se logra a través de cuatro secciones principales, implementadas secuencialmente en `mrls-logos.py`:

### 1. Creación del Diccionario Base (Sección I y II)

El MRLS es un modelo **no iterativo** (Entrenamiento Instantáneo). El "entrenamiento" comienza ordenando el conjunto de datos de entrada (`X, Y`) de menor a mayor valor de `X`. Esta ordenación transforma el `DataFrame` inicial en la estructura fundamental del MRLS: un diccionario donde cada punto `(X, Y)` está indexado por su valor `X`.

**Ejemplo de Conjunto de Entrada (Input Set):**

Para demostrar el proceso, usamos el siguiente conjunto de datos desordenado (Entrada $X$, Salida $Y$):

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
Una vez ordenado por $X$, este se convierte en el **Diccionario Base**:

```
// Diccionario Base (Ordenado por X)
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

### 2. Optimización (Sección III)

A partir del diccionario base ordenado, se calcula la función lineal que conecta cada par de puntos adyacentes $(x_1, y_1)$ y $(x_2, y_2)$. Este paso transforma los datos $(X, Y)$ en los parámetros del segmento:

* **Pendiente (P)**: Representa el **Peso** (`W`) del segmento.
    $$P = \frac{y_2 - y_1}{x_2 - x_1}$$
* **Ordenada al Origen (O)**: Representa el **Sesgo** (`B`) del segmento.
    $$O = y_1 - P \cdot x_1$$

El resultado es un **Diccionario Optimizado** donde cada clave $X_n$ (el inicio del segmento) almacena la tupla $(P, O)$. Este es el conocimiento explícito del modelo.

**Ejemplo de Diccionario Optimizado (Pesos y Sesgos):**

```
// Diccionario Optimizado (Pesos y Sesgos)
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

### 3. Compresión sin Pérdida (Invarianza Geométrica - Sección IV)

Este paso elimina la redundancia geométrica del modelo. Si tres puntos consecutivos $(X_{n-1}, X_n, X_{n+1})$ se encuentran sobre la misma línea recta, el punto intermedio $X_n$ se considera redundante.

* **Criterio:** Si $\text{Pendiente}(X_{n-1}) \approx \text{Pendiente}(X_n)$, se elimina el punto $X_n$ del diccionario.
* **Resultado:** Se eliminan "neuronas" intermedias que no contribuyen a un cambio en la dirección de la curva, logrando una compresión del diccionario **sin pérdida** de información geométrica.

**Ejemplo de Compresión sin Pérdida:**

Se eliminan `[+0.00]` y `[+3.00]` por redundancia de Pendiente, quedando:
```
// Diccionario Optimizado (Compresión sin Pérdida)
[-8.00] (-1.00,-12.0)
[-6.00] (-0.01,-6.06)
[-5.00] (+0.01,-5.96)
[-4.00] (+1.00,-2.00)
[-2.00] (+2.00,+0.00)
[+2.00] (+3.00,-2.00)
[+4.00] (+4.00,-6.00)
```

### 4. Compresión con Pérdida (Criterio Humano - Sección V)

Este es el paso de mayor compresión, donde se aplica un **criterio humano** (la tolerancia $\epsilon$) para eliminar puntos cuya contribución al error global es inferior a un umbral predefinido.

* **Tolerancia ($\epsilon$):** Un valor de error máximo aceptable (por ejemplo, $0.03$).
* **Criterio de Permanencia:** El punto $X_{\text{actual}}$ se mantiene si el error absoluto al interpolar entre sus vecinos es superior a $\epsilon$.

$$\text{Error} = | Y_{\text{true}} - Y_{\text{hat}} |$$

Si $\text{Error} > \epsilon$, el punto se mantiene. Si $\text{Error} \le \epsilon$, se elimina (compresión con pérdida).

**Ejemplo de Compresión con Pérdida Final ($\epsilon=0.03$):**

Se elimina `[-5.00]` al tener un error de $0.01 \le 0.03$ al ser interpolado entre `[-6.00]` y `[-4.00]`.

```
// Diccionario Optimizado (Compresión con Pérdida Final)
[-8.00] (-1.00,-12.0)
[-6.00] (+0.00,-6.00) // Parámetros ajustados por la interpolación
[-4.00] (+1.00,-2.00)
[-2.00] (+2.00,+0.00)
[+2.00] (+3.00,-2.00)
[+4.00] (+4.00,-6.00)
```

---

## 5. Extensiones y Propiedades Operacionales del MRLS

La naturaleza modular de los segmentos del MRLS le otorga propiedades operacionales que lo distinguen de los modelos de redes neuronales iterativas:

### 5.1 Modularidad e Intercambio en Caliente (Hot Swapping)
Dado que cada segmento es autónomo y no interactúa con los pesos de otros segmentos, el MRLS permite la **Modificación en Caliente**. Esto significa que se puede actualizar, optimizar o añadir un nuevo conjunto de datos en un sector específico del diccionario **en tiempo real**, sin interrumpir la operación de inferencia del resto de la red.

### 5.2 Activación No Lineal y Compresión Multimodal
El proceso de compresión puede extenderse para reemplazar localmente un conjunto de múltiples segmentos lineales por una única función de orden superior (ej. cuadrática o exponencial), siempre que el error de sustitución se mantenga dentro de la tolerancia ($\epsilon$). Esto genera una **Compresión Multimodal** y compacta aún más la arquitectura.

### 5.3 Caja Transparente (Interpretabilidad Total)
El MRLS es un modelo de "caja transparente". Almacena el conocimiento de forma explícita (Pendiente $P$ y Ordenada $O$ para cada segmento). Esto permite una trazabilidad completa de cada predicción y es ideal para entornos que requieren alta interpretabilidad y auditoría.

---

## 🎯 Predicción y Generalización (Sección VII)

La función `predict(X)` utiliza el diccionario MRLS final y comprimido.

1.  **Búsqueda del Segmento Activo:** Para una nueva entrada $X$, el modelo encuentra la clave $X_n$ más próxima y menor o igual a $X$ ($X_n \le X$). Esta $X_n$ define el segmento lineal activo $(P, O)$.
2.  **Ecuación Maestra:** Se aplica la fórmula lineal para obtener la predicción $Y_{\text{predicha}}$.

$$Y_{\text{predicha}} = X \cdot P + O$$

### Generalización (Extrapolación)

El MRLS maneja la extrapolación fuera de los límites de entrenamiento de la siguiente manera:

* **Extrapolación Segmental (Corta Distancia):** Se extiende el segmento lineal de frontera (el primero o el último) al infinito, utilizando los parámetros $(P, O)$ del segmento más cercano al límite.
* **Proyección Zonal (Metaprogresión Avanzada):** En modelos avanzados, el MRLS puede analizar la progresión de los Pesos ($P$) y Sesgos ($O$) cerca de los límites para detectar patrones de orden superior. Esto permite proyectar el siguiente segmento con base en el **patrón global de la red**, ofreciendo una extrapolación de larga distancia potencialmente más precisa.

---

## IX. Bibliografía Conceptual

Las siguientes referencias conceptuales inspiran o contrastan con los principios fundamentales del Modelo de Regresión Lineal Segmentada (MRLS):

1.  Regresión Segmentada y Ajuste de Curvas: Trabajos sobre la aproximación de funciones complejas mediante modelos de regresión definidos por tramos.
2.  Cuantización y Compresión de Modelos: Técnicas orientadas a reducir el tamaño de los modelos neuronales para implementación en hardware con restricciones de memoria.
3.  Modelos de Caja Blanca (Interpretabilidad): Estudios sobre la trazabilidad y la comprensión de las decisiones de un modelo de predicción.
4.  Modularidad y Arquitecturas Desacopladas: Principios de diseño de software que permiten la modificación local sin efectos colaterales.
5.  Procesos de Entrenamiento y Algoritmos de Optimización: Conceptos relacionados con la eficiencia de capacitación y el entrenamiento no iterativo.
6.  Gestión de Datos Dispersos y Outliers: Técnicas para garantizar la robustez del modelo frente a puntos aislados o inconsistencias en los datos de entrada.
7.  Sistemas de Memoria Asociativa: El Diccionario Optimizado como una forma de estructura de datos eficiente para el almacenamiento y recuperación rápida de patrones.
8.  Diseño de Sistemas Tolerantes a Fallos: Principios que permiten la modificación o actualización de componentes (Modificación en Caliente) sin interrupción del servicio global.
9.  Teoría de Series Temporales: Trabajos sobre la detección de patrones de progresión (Metaprogresión) para realizar extrapolaciones de largo alcance.
10.  

---

Archivos del Proyecto

[readme-demo](https://akinetic.github.io/neural-network) : La aplicación web para la visualización.

[slrm-logos.py](../slrm-logos.py) : 
El código fuente principal que contiene la lógica de entrenamiento y predicción (V5.10b).

[slrm_manual.html](https://akinetic.github.io/neural-network/slrm_manual.html)  : (Contenido de este archivo): Este manual técnico (V5.10b).

[slrm_visualizer.html](https://akinetic.github.io/neural-network/slrm_visualizer.html) : La aplicación web para la visualización (Utiliza lógica anterior - No actualizado).

---

Authors

Alex Kinetic and Logos

Project under MIT License
