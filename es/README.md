# Segmented Linear Regression Model (SLRM)

> Este proyecto implementa el Segmented Linear Regression Model (SLRM), una alternativa eficiente a las Redes Neuronales Artificiales (**ANN**). El SLRM modela conjuntos de datos mediante funciones lineales por tramos, utilizando un proceso de **neural compression** (compresión neuronal) para reducir la complejidad sin comprometer la precisión más allá de una tolerancia definida por el usuario.

El núcleo de la solución es el algoritmo de compresión, que transforma un conjunto de datos desordenado (`DataFrame` / `X, Y`) en un diccionario final altamente optimizado, listo para realizar predicciones.

## Estructura del Proyecto

* **`slrm-logos.py`**: Contiene la implementación completa del proceso de entrenamiento (Creación, Optimización, Compresión) y la función de predicción (`predict`). Este código genera el diccionario final SLRM que consume la aplicación web.
* **`index.html`**: Implementación de la visualización utilizando D3.js y JavaScript Vanilla, que muestra el conjunto de datos y la curva de predicción del SLRM (la función lineal por tramos).

---

## Arquitectura SLRM: El Proceso de Entrenamiento (Compression)

El entrenamiento del SLRM se logra a través de cuatro secciones principales, implementadas secuencialmente en `slrm-logos.py`:

### 1. Creación del Diccionario Base

El SLRM es un modelo **no iterativo** (Entrenamiento Instantáneo). El "entrenamiento" comienza ordenando el conjunto de datos de entrada (`X, Y`) desde el valor más bajo al más alto de `X`. Este ordenamiento transforma el `DataFrame` inicial en la estructura fundamental del SLRM: un diccionario donde cada punto `(X, Y)` está indexado por su valor `X`.

**Ejemplo de Conjunto de Entrada:**

Para demostrar el proceso, utilizamos el siguiente conjunto de datos desordenado (Entrada $X$, Salida $Y$):

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

---

### 2. Optimización

A partir del diccionario base ordenado, se calcula la función lineal que conecta cada par de puntos adyacentes $(x_1, y_1)$ y $(x_2, y_2)$. Este paso transforma los datos $(X, Y)$ en los parámetros del segmento:

* **Pendiente (P)**: Representa el **Peso** (`W`) del segmento.
    $$P = \frac{y_2 - y_1}{x_2 - x_1}$$
* **Ordenada al Origen (O)**: Representa el **Sesgo** (`B`) del segmento.
    $$O = y_1 - P \cdot x_1$$

El resultado es un **Diccionario Optimizado** donde cada clave $X_n$ (el inicio del segmento) almacena la tupla $(P, O)$. Este es el conocimiento explícito del modelo.

**Ejemplo de Diccionario Optimizado (Pesos y Sesgos):**

```text
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

---

### 3. Compresión sin Pérdida (Invarianza Geométrica)

Este paso elimina la redundancia geométrica del modelo. Si tres puntos consecutivos $(X_{n-1}, X_n, X_{n+1})$ se encuentran sobre la misma línea recta, el punto intermedio $X_n$ se considera redundante.

* **Criterio:** Si $\text{Pendiente}(X_{n-1}) \approx \text{Pendiente}(X_n)$, se elimina el punto $X_n$ del diccionario.
* **Resultado:** Se eliminan las "neuronas" intermedias que no contribuyen a un cambio en la dirección de la curva, logrando una compresión **sin pérdida** (lossless) de la información geométrica del diccionario.

**Ejemplo de Compresión sin Pérdida:**

`[+0.00]` y `[+3.00]` se eliminan debido a la redundancia de la Pendiente, resultando en:

```text
// Diccionario Optimizado (Compresión sin Pérdida)
[-8.00] (-1.00,-12.0)
[-6.00] (-0.01,-6.06)
[-5.00] (+0.01,-5.96)
[-4.00] (+1.00,-2.00)
[-2.00] (+2.00,+0.00)
[+2.00] (+3.00,-2.00)
[+4.00] (+4.00,-6.00)
```

---

### 4. Compresión con Pérdida (Criterio Humano)

Este es el paso para la compresión máxima, donde se aplica un **criterio humano** (la tolerancia $\epsilon$) para eliminar puntos cuya contribución al error global está por debajo de un umbral predefinido.

* **Tolerancia ($\epsilon$):** Un valor de error máximo aceptable (por ejemplo, $0.03$).
* **Criterio de Permanencia:** El punto $X_{\text{actual}}$ se considera **Relevante** y se conserva si el error absoluto al interpolar entre sus vecinos es mayor que $\epsilon$.

$$\text{Error} = | Y_{\text{true}} - Y_{\text{hat}} |$$

Si $\text{Error} > \epsilon$, el punto se mantiene. Si $\text{Error} \le \epsilon$, se elimina (compresión con pérdida).

**Ejemplo de Compresión con Pérdida Final ($\epsilon=0.03$):**

`[-5.00]` se elimina ya que su error es $0.01 \le 0.03$ al ser interpolado entre `[-6.00]` y `[-4.00]`.

```text
// Diccionario Optimizado (Compresión con Pérdida Final)
[-8.00] (-1.00,-12.0)
[-6.00] (+0.00,-6.00) // Parámetros ajustados debido a la interpolación
[-4.00] (+1.00,-2.00)
[-2.00] (+2.00,+0.00)
[+2.00] (+3.00,-2.00)
[+4.00] (+4.00,-6.00)
```

---

## 5. Predicción y Generalización

La inferencia en el SLRM es un proceso de búsqueda directa y cálculo lineal. Cuando se recibe una entrada $X$, el modelo no realiza activaciones complejas en capas ocultas; en su lugar, localiza el segmento de conocimiento correspondiente y aplica la función lineal almacenada.

1.  **Búsqueda del Segmento Activo:** Para una nueva entrada $X$, el modelo encuentra la clave $X_n$ más próxima y menor o igual a $X$ ($X_n \le X$). Esta $X_n$ define el segmento lineal activo $(P, O)$.
2.  **Ecuación Maestra:** Se aplica la fórmula lineal para obtener la predicción $Y_{\text{predicha}}$.

$$Y_{\text{predicha}} = X \cdot P + O$$

Este método garantiza una latencia de predicción constante ($O(1)$ o $O(\log n)$ dependiendo de la estructura de búsqueda), lo que lo hace ideal para sistemas de tiempo real y **IoT**.

### Generalización (Extrapolación)

El SLRM maneja la extrapolación fuera de los límites de entrenamiento de la siguiente manera:

* **Extrapolación Segmental (Corta Distancia):** Se extiende el segmento lineal de frontera (el primero o el último) al infinito, utilizando los parámetros $(P, O)$ del segmento más cercano al límite.
* **Proyección Zonal (Metaprogresión Avanzada):** En modelos avanzados, el SLRM puede analizar la progresión de los Pesos ($P$) y Sesgos ($O$) cerca de los límites para detectar patrones de orden superior. Esto permite proyectar el siguiente segmento con base en el **patrón global de la red**, ofreciendo una extrapolación de larga distancia potencialmente más precisa.

---

## 6. Superioridad del SLRM: Eficiencia vs. Modelos Estándar

Aunque el SLRM es fundamentalmente una arquitectura para la **Knowledge Compression** (Compresión de Conocimiento), su rendimiento al modelar datos complejos no lineales supera a los modelos paramétricos estándar y demuestra una eficiencia estructural frente a modelos jerárquicos complejos como los Árboles de Decisión.

Se realizó una prueba comparativa frente a modelos de *scikit-learn* utilizando un conjunto de datos no lineales de 15 puntos desafiantes ($\epsilon=0.5$).

### Métricas de Rendimiento y Complejidad (Hoja de Decisión)

Los resultados demuestran que el SLRM logra una precisión casi perfecta con la mayor compresión de datos, demostrando su superioridad estructural en términos de simplicidad e interpretabilidad.

| Modelo | $R^2$ (Coeficiente de Determinación) | Complejidad del Modelo | Tasa de Compresión |
| :--- | :--- | :--- | :--- |
| **SLRM (Segmentado)** | **0.9893** | **6 (Puntos Clave/Segmentos)** | **60.00%** |
| Árbol de Decisión (Profundidad 5) | **0.9964** | 9 (Nodos Hoja/Regiones) | 0% |
| Polinomial (Grado 3) | 0.9328 | 4 (Coeficientes) | 0% |
| SLR (Lineal Simple) | 0.7399 | 2 (Parámetros) | 0% |

> **Conclusión:** El SLRM logra un $R^2=0.9893$ con una **compresión de datos del 60%** utilizando solo **5 segmentos lineales** (6 puntos clave). El Árbol de Decisión logra una precisión similar ($R^2=0.9964$) pero requiere **9 regiones** para hacerlo, lo que confirma la **eficiencia geométrica** superior y la simplicidad inherente del SLRM.

---

## 7. Extensiones del SLRM y Propiedades Operativas

La naturaleza modular de los segmentos del SLRM proporciona propiedades operativas que lo distinguen de los modelos de redes neuronales iterativas:

### 7.1 Modularidad y Hot Swapping
Dado que cada segmento es autónomo y no interactúa con los pesos (weights) de otros segmentos, el SLRM permite la **Modificación en Caliente** (Hot Swapping). Esto significa que un sector específico del diccionario puede ser actualizado, optimizado o se pueden añadir nuevos datos **en tiempo real**, sin interrumpir la operación de inferencia del resto de la red.

### 7.2 Activación No Lineal y Compresión Multimodal
El proceso de compresión puede extenderse para reemplazar localmente un conjunto de múltiples segmentos lineales por una única función de orden superior (ej. cuadrática o exponencial), siempre que el error de sustitución se mantenga dentro de la tolerancia ($\epsilon$). Esto genera una **Compresión Multimodal** y compacta aún más la arquitectura.

### 7.3 Caja Transparente (Interpretabilidad Total)
El SLRM es un modelo de "caja transparente". Almacena el conocimiento de forma explícita (Pendiente $P$ y Ordenada $O$ para cada segmento). Esto permite una trazabilidad total de cada predicción y es ideal para entornos que requieren alta interpretabilidad y auditoría.

---

## 8. Instalación y Uso

El **SLRM-LOGOS** está diseñado para ser extremadamente ligero, sin dependencias externas.

### Instalación vía NPM

```bash
npm install slrm-logos-es
```

### Ejemplo de Uso en JavaScript (Node.js)

```javascript
const { train_slrm, predict_slrm } = require('slrm-logos-es');

// 1. Datos de entrenamiento (Formato: "x, y")
const data = "1,2\n2,4\n3,8\n4,16";

// 2. Entrenar el modelo con una tolerancia (Epsilon) de 0.5
const { model, originalData, maxError } = train_slrm(data, 0.5);

// 3. Realizar una predicción
const inputX = 2.5;
const prediction = predict_slrm(inputX, model, originalData);

console.log(`Predicción para X=${inputX}: Y=${prediction.y_pred}`);
console.log(`Error Máximo del Modelo: ${maxError}`);
```

---

## 9. Bibliografía Conceptual

Las siguientes referencias conceptuales inspiran o contrastan con los principios fundamentales del Segmented Linear Regression Model (SLRM):

1. **Segmented Regression and Curve Fitting:** Trabajos sobre la aproximación de funciones complejas utilizando modelos de regresión definidos por tramos (*piecewise*).
2. **Quantization and Model Compression:** Técnicas orientadas a reducir el tamaño de los modelos neuronales para su implementación en hardware con restricciones de memoria.
3. **White Box Models (Interpretability):** Estudios sobre la trazabilidad y la comprensión de las decisiones de un modelo de predicción.
4. **Modularity and Decoupled Architectures:** Principios de diseño de software que permiten la modificación local sin efectos colaterales.
5. **Time Series Theory:** Trabajos sobre la detección de patrones de progresión (Metaprogresión) para realizar extrapolaciones de largo alcance más precisas.

---

## Recursos del Proyecto y Navegación

* **[Visualizador Interactivo en Vivo](https://akinetic.github.io/neural-network/es/)**
    *Utiliza la aplicación web para probar el modelo SLRM y la compresión en tiempo real.*

* **[slrm-logos-es.py](slrm-logos-es.py)**
    *El motor principal de producción que contiene toda la lógica de entrenamiento y compresión.*

* **[Manual Técnico](slrm_manual_es.md)**
    *Inmersión profunda en los fundamentos matemáticos y la arquitectura.*

* **[Reporte de Rendimiento](slrm_reporte_rendimiento.md)**
    *Comparativa detallada frente a modelos de Scikit-Learn (Precisión y Eficiencia).*

---

## Autores

* **Alex Kinetic**
* **Logos**

---

## Licencia

Este proyecto está bajo la **Licencia MIT** - consulta el archivo [LICENSE](LICENSE) para más detalles.

> *"La simplicidad es la máxima sofisticación."* - Segmented Linear Regression Model (SLRM)
