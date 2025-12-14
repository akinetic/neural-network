# Arquitectura Neuronal Segmentada ANS

> **Este proyecto introduce la Arquitectura Neuronal Segmentada ANS, un modelo de alta eficiencia operacional y altísima compactación de datos. A diferencia de las Redes Neuronales Artificiales (RNA), el ANS se basa en una estructura modular y autónoma que permite el "Entrenamiento Instantáneo", la "Modificación en Caliente (Hot Swapping)" sin efectos colaterales, y la activación de funciones no lineales. Su núcleo de "Compresión Multimodal" genera un diccionario optimizado de pesos y sesgos, garantizando la trazabilidad total (Caja Transparente) y una pérdida de precisión controlada ($\epsilon$).**

---

## I. El Conjunto de Entrada (Input Set)

Consideremos que un DataFrame (Input Set) es una Base de Datos, donde cada entrada $X$ está asociada a una única salida $Y$ (es decir, es una función de $X$ a $Y$). Además, el conjunto de datos no contiene entradas duplicadas, lo que garantiza la unicidad de los puntos de control del modelo (propiedad "sin Densidad o Profundidad").

En este proyecto, nuestro Dataframe (Input Set) o modelo de juguete es el siguiente:

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

Donde el primer valor ($X$) representa la entrada y el segundo valor ($Y$) representa la salida asociada. Nótese que el DataFrame se presenta desordenado para simular un input de datos sin procesar.

---

## II. Generación del Diccionario Base (Entrenamiento Instantáneo)

La creación del Diccionario Base simplemente consiste en ordenar de menor a mayor los pares $(X, Y)$ del DataFrame, utilizando $X$ como clave de ordenamiento.

En nuestro modelo de juguete, el Diccionario Base es el siguiente:

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

El Diccionario Base ya es funcional (aunque aún no está optimizado ni comprimido) pues permite la inferencia mediante dos procedimientos:

1.  **Búsqueda Directa:** Si la entrada $X$ coincide con un valor $X_n$ del Diccionario Base, la salida es el valor $Y$ asociado a $X_n$.
2.  **Inferencia Lineal (Interpolación):** Si la entrada $X$ no coincide con un punto conocido y cae dentro de los límites del Diccionario, la salida $Y$ está dada por la siguiente ecuación:

$$Y = (X - X_1) \cdot \left[ \frac{Y_2 - Y_1}{X_2 - X_1} \right] + Y_1$$

Donde $X_1$ es el valor más próximo menor a $X$, $X_2$ es el valor más próximo mayor a $X$, y $Y_1, Y_2$ son sus salidas asociadas en el Diccionario Base.

Por ejemplo, en nuestro modelo de juguete, si la entrada $X$ es $5$ entonces la salida $Y$ está dada por:

$$Y = (5 - 4) \cdot \left[ \frac{18 - 10}{6 - 4} \right] + 10$$

$$Y = 14$$

---

## III. Transformación del Diccionario (Pesos y Sesgos)

La optimización del Diccionario Base consiste en hacer más funcional al Diccionario Base bajo un determinado criterio o varios.

El criterio de optimización inicial consiste en transformar los puntos de datos $(X, Y)$ en las definiciones de los segmentos lineales. Esto se logra calculando la Pendiente ($m$, o Peso) y la Ordenada al Origen ($b$, o Sesgo) para cada segmento definido por pares de puntos contiguos.

En nuestro ejemplo, resulta:

```
[-8.00,-4.00]
				(-1.00,-12.0)
[-6.00,-6.00]
				(-0.01,-6.06)
[-5.00,-6.01]
				(+0.01,-5.96)
[-4.00,-6.00]
				(+1.00,-2.00)
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

Cada par $(m, b)$ se asocia al punto $X$ de inicio del segmento. De esta manera, el Diccionario Optimizado almacena la "Activación" de cada Neurona (punto $X$) en forma de sus parámetros de segmento $(m, b)$, eliminando la necesidad de almacenar el valor $Y$ original.

El Diccionario Optimizado queda de la siguiente manera:

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
*(Corrección: El punto [+6.00] se elimina porque no inicia un nuevo segmento.)*

Este nuevo Diccionario Optimizado es la base para la compresión. Ya es funcional también como lo es el Diccionario Base para poder ser utilizado.

Para cualquier entrada $X$ dentro de los límites del Diccionario, el Procedimiento de Inferencia consiste en que la salida $Y$ está dada por la Ecuación Maestra del ANS:

$$Y = X \cdot \text{Pendiente}(\text{Peso}) + \text{Ordenada}(\text{Sesgo})$$

Donde la $\text{Pendiente}(\text{Peso})$ y la $\text{Ordenada}(\text{Sesgo})$ son las asociadas al $X_n$ más próximo menor o igual del Diccionario Optimizado (el inicio del segmento actual).

Por ejemplo, en nuestro modelo de juguete, si la entrada $X$ es $5$ entonces la salida $Y$ está dada por:

$$Y = 5 \cdot 4 - 6$$

$$Y = 14$$

---

## IV. Compresión de Redundancia (Sin Pérdida)

La Compresión de Redundancia elimina la información redundante del Diccionario Optimizado. Esta etapa formaliza la red, dejando solo los puntos donde la Pendiente sufre un cambio.

Básicamente, si el segmento de línea que comienza en el punto $X_{n+1}$ tiene la misma Pendiente ($m$) y Ordenada al Origen ($b$) que el segmento que comienza en el punto anterior $X_n$ (es decir, existe **Continuidad de Pendiente**), entonces $X_{n+1}$ es un punto redundante. La información de este punto ya está implícitamente contenida por la neurona de $X_n$, por lo que se elimina.

En nuestro modelo de juguete, los Datos o las Neuronas Redundantes son: `[+0.00] (+2.00,+0.00)` y `[+3.00] (+3.00,-2.00)`. El Diccionario Optimizado Comprimido sin pérdida queda de la siguiente manera:

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

---

## V. Compresión Algorítmica (Pérdida Controlada $\epsilon$)

La Compresión Algorítmica permite reducir aún más el tamaño del diccionario, eliminando Neuronas cuya contribución al error global es inferior a un umbral predefinido.

Cuando un Dato o una Neurona no es relevante localmente depende de la **Tolerancia de Error ($\epsilon$)** que se quiera obtener de la Arquitectura Neuronal Segmentada. Básicamente, se elimina la Neurona $X_n$ si, al ser sustituida por una interpolación entre sus vecinos, el error máximo absoluto no supera la tolerancia establecida.

En nuestro modelo de juguete, se establece una Tolerancia de Error ($\epsilon$) de $0.03$. Esto significa que cualquier punto cuya desviación absoluta sea menor o igual a $\epsilon$ es considerado "información no relevante" y puede ser eliminado.

Por lo tanto, en nuestro modelo de juguete, el Dato o Neurona Redundantes es: `[-5.00] (+0.01,-5.96)`, quedando, por lo tanto, el Diccionario Optimizado de nuestro modelo de juguete y con pérdida de información no relevante de la siguiente manera:

```
// Diccionario Optimizado (Compresión con Pérdida Final)
[-8.00] (-1.00,-12.0)
[-6.00] (+0.00,-6.00)
[-4.00] (+1.00,-2.00)
[-2.00] (+2.00,+0.00)
[+2.00] (+3.00,-2.00)
[+4.00] (+4.00,-6.00)
```

Ahora, en el Diccionario Optimizado y Comprimido de nuestro modelo de juguete para $X$ igual a $-5$, el resultado es $-6$, obteniéndose solamente una diferencia de $0.01$ con respecto al dato real del DataFrame o Diccionario Base ($-6.01$).

---

## VI. Activación No Lineal y Compresión Multimodal

Esta sección abarca métodos de compresión avanzada que permiten al ANS ir más allá de los segmentos lineales. Gracias a la **naturaleza local y autónoma de cada neurona del Diccionario Optimizado**, es posible la **sustitución funcional**, donde se reemplaza localmente un conjunto de segmentos por una función de orden superior, sin afectar otras áreas de la red.

El objetivo de esta compresión es:

### 1. Sustitución No Lineal sin Pérdida de Información

Consiste en identificar un grupo de neuronas lineales contiguas y sustituirlas por una única función de orden superior (p. ej., cuadrática, cúbica, exponencial, etc.) que pase **exactamente** por todos los puntos de quiebre. El resultado es un Diccionario mucho más compacto sin introducir error adicional.

* **Ejemplo:** Sustituir un tramo de 5 neuronas lineales por una única función parabólica ($Y = aX^2 + bX + c$) que ajusta perfectamente los puntos.

### 2. Sustitución No Lineal con Pérdida No Relevante

El procedimiento más común. La función de orden superior sustituye el conjunto de neuronas lineales si el error máximo absoluto entre la función no lineal y los datos originales no supera la Tolerancia de Error ($\epsilon$). Esto permite una compactación extrema.

* **Aplicación Local:** Esta técnica puede aplicarse solo a sectores específicos de la red (p. ej., entre $X=-1$ y $X=+12$), dejando el resto de la red con funciones lineales si estas son más eficientes localmente o si no es necesario un cambio.

### 3. Ajustes Locales del Procedimiento

Además de la sustitución funcional, se pueden aplicar ajustes finos para optimizar la transición entre las neuronas y el procedimiento de inferencia. Esto incluye el uso de un $\epsilon$ variable (más estricto en zonas críticas, más flexible en zonas planas) para una mejor eficiencia local.

---

## VII. Gestión de Frontera y Proyección

Esta sección define cómo el ANS gestiona las entradas $X$ que caen fuera de su rango de entrenamiento conocido [$X_{\min}, X_{\max}$].

### 1. Acotación e Imposición de Límites (Restricción por Defecto)

Por defecto, el sistema puede optar por **imponer sus límites** y rechazar o acotar la entrada.

* Si $X < X_{\min}$, la salida $Y$ es fijada al valor $Y_{\min}$ del primer punto conocido.
* Si $X > X_{\max}$, la salida $Y$ es fijada al valor $Y_{\max}$ del último punto conocido.

### 2. Proyección Segmental (Extrapolación Estándar)

Si el sistema acepta realizar proyecciones, la neurona asociada al $X_{\min}$ sufre una **Transformación de Frontera**.

* **Extremo Inferior ($X \rightarrow -\infty$):** La neurona asociada al $X_{\min}$ (e.g., $X = -8.00$) se convierte en el ancla del segmento inicial y **funcionalmente representa el límite $X = -\infty$**. Esto asegura que cualquier entrada $X$ infinitamente menor que el $X_{\min}$ (e.g., $-30$) activa correctamente esta primera neurona, cumpliendo la regla de búsqueda "encontrar el $X_n$ más próximo menor o igual a $X$".
* **Extremo Superior:** El punto $X_{\max}$ (el valor más grande del DataFrame original) **desaparece del Diccionario Optimizado** durante la Sección III, ya que no inicia un nuevo segmento, sino que marca el final del segmento anterior. Por lo tanto, la última neurona restante define el segmento que se proyecta hacia $+\infty$.

La salida $Y$ se calcula aplicando la Ecuación Maestra del ANS utilizando los Pesos ($m$) y Sesgos ($b$) de la neurona de frontera activa, lo que resulta en una proyección de **corta distancia**, controlada y consistente.

### 3. Proyección Zonal o Global (Metaprogresión)

El ANS tiene la capacidad de analizar el comportamiento global de las Pendientes ($m$) y Sesgos ($b$) cerca de los límites para detectar patrones de orden superior (metaprogresiones).

* **Aplicación:** Si existe una progresión bien marcada en los pesos ($m=2, m=3, m=4$), el sistema puede proyectar el siguiente Peso ($m=5$) y su Sesgo asociado para extender el Diccionario. Esta proyección utiliza la información del **patrón de la red** en su conjunto (zonal) en lugar de depender únicamente del último segmento, lo que ofrece una proyección de **larga distancia** potencialmente más precisa.

---

## VIII. Propiedades Operacionales y Arquitectónicas

+ **Modularidad y Autonomía Neuronal (Cero Efecto Colateral):** La característica distintiva del ANS es que cada neurona o segmento opera de forma local y autónoma. Esto permite modificar, optimizar o entrenar un sector específico del Diccionario Optimizado sin necesidad de reevaluar o alterar el resto de la red, algo imposible de replicar de manera eficiente en arquitecturas neuronales tradicionales (donde un cambio de peso afecta a toda la red).

+ **Modificación en Caliente (Hot Swapping):** Derivado de la modularidad, el Diccionario Optimizado puede ser modificado, actualizado o re-entrenado localmente **en tiempo real** (en caliente), sin interrumpir las operaciones de inferencia del resto de la red. Esto garantiza una altísima disponibilidad y flexibilidad operativa del modelo.

+ **Entrenamiento Instantáneo:** Dado que el "entrenamiento" del ANS consiste puramente en el ordenamiento del DataFrame y el cálculo directo de pesos y sesgos (Secciones II y III), el modelo carece de la fase iterativa de retropropagación (backpropagation), resultando en una velocidad de capacitación casi instantánea.

+ **Interpretabilidad (Modelo de Caja Transparente):** El Diccionario Optimizado almacena conocimiento explícito. Cada neurona representa un segmento funcional con parámetros fijos (Pesos y Sesgos), permitiendo una trazabilidad completa de la inferencia, a diferencia de los modelos de "caja negra".

+ Se podría visualizar como que cada Neurona se ocupa de un sector de la Red y que su funcionamiento no altera la Red más allá de ese sector de la Red.

+ Además, entre las Neuronas locales (más próximas entre sí) se pueden asociar y/o unificar para hacer de la Red Neuronal más compacta y eficiente.

+ Lo ideal sería que el Diccionario Optimizado sea lo más compacto posible, que la velocidad de entrada/salida sea lo más rápida posible y con el menor costo posible, y finalmente que la Red Neuronal pueda evolucionar de la manera más flexible/sencilla posible.

---

## IX. Bibliografía

Las siguientes referencias conceptuales inspiran o contrastan con los principios fundamentales de la Arquitectura Neuronal Segmentada (ANS):

1.  **Regresión Segmentada y Ajuste de Curvas:** Trabajos sobre la aproximación de funciones complejas mediante modelos de regresión definidos por tramos.
2.  **Cuantización y Compresión de Modelos:** Técnicas orientadas a reducir el tamaño de los modelos neuronales para implementación en hardware con restricciones de memoria.
3.  **Modelos de Caja Blanca (Interpretabilidad):** Estudios sobre la trazabilidad y la comprensión de las decisiones de un modelo de predicción, contrastando con la "caja negra" de las RNA.
4.  **Modularidad y Arquitecturas Desacopladas:** Principios de diseño de software que permiten la modificación local sin efectos colaterales en sistemas complejos.
5.  **Procesos de Entrenamiento y Algoritmos de Optimización:** Conceptos relacionados con la eficiencia de capacitación, contrastando el Entrenamiento Instantáneo del ANS con la naturaleza iterativa de la Retropropagación.
6.  **Gestión de Datos Dispersos y Outliers:** Técnicas para garantizar la robustez del modelo frente a puntos aislados o inconsistencias en los datos de entrada.
7.  **Sistemas de Memoria Asociativa:** El Diccionario Optimizado como una forma de estructura de datos eficiente para el almacenamiento y recuperación rápida de patrones.
8.  **Diseño de Sistemas Tolerantes a Fallos:** Principios que permiten la modificación o actualización de componentes (Modificación en Caliente) sin interrupción del servicio global.
9.  **Teoría de Series Temporales:** Trabajos sobre la detección de patrones de progresión (Metaprogresión) para realizar extrapolaciones de largo alcance con mayor precisión.

---

Authors

Alex Kinetic and Logos

Project under MIT License
