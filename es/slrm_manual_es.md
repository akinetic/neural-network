# Manual Técnico SLRM-LOGOS (V5.12)

## Arquitectura Central del Segmented Linear Regression Model (SLRM)

El Segmented Linear Regression Model (SLRM), impulsado por el algoritmo determinista Logos Core, es una solución rápida y eficiente para la compresión y el modelado de datos unidimensionales (Series Temporales, curvas de calibración, etc.).

Utiliza la técnica **Minimum Required Line Segments (MRLS)** para representar conjuntos de datos con el menor número posible de segmentos lineales, cumpliendo con una tolerancia de error ($\epsilon$) definida por el usuario.

---

### Características Clave (SLRM V5.12)

* **Verificación y Robustez:** Versión V5.12 verificada y probada con correcciones críticas en el manejo del error máximo en la compresión Lossy.
* **Velocidad:** El proceso de compresión central es no iterativo y determinista, apoyándose en NumPy para una velocidad de entrenamiento óptima, lo que lo convierte en un modelo de **Entrenamiento Instantáneo**.
* **Compresión (Sin Pérdida / Con Pérdida):** Implementa tanto la compresión geométrica exacta (`_lossless_compression`) como la compresión basada en tolerancia (`_lossy_compression`).
* **Predicción Optimizada:** Utiliza un caché de predicción **LRU (Least Recently Used)** para acelerar drásticamente la inferencia en tiempo real de puntos repetidos o cercanos.
* **Integridad de Datos:** Incluye una etapa inicial de purificación de datos (`_clean_and_sort_data`) para manejar duplicados (promediando valores de Y para la misma X) y ordenar los datos por X.
* **Interpretabilidad Total:** El modelo es una "caja transparente", que almacena conocimiento explícito (Pendiente $P$ e Intercepto $O$) para cada segmento.

### Etapas del Proceso de Entrenamiento (V5.12)

La función `train_slrm` ejecuta un flujo de compresión de tres etapas:

1.  **Purificación y Ordenamiento (`_clean_and_sort_data`):**
    * Convierte los datos de entrada en una secuencia de puntos limpia y ordenada.
    * Maneja duplicados en X promediando sus valores de Y correspondientes.
2.  **Compresión Sin Pérdida (`_lossless_compression`):**
    * **Invarianza Geométrica:** Identifica y elimina todos los puntos intermedios que son geométricamente colineales (puntos de ruptura no críticos).
    * El resultado es la **Base de Puntos de Ruptura** que preserva la precisión original.
3.  **Compresión Con Pérdida (MRLS, `_lossy_compression`):**
    * Aplica el criterio de error humano $\epsilon$ sobre la Base de Puntos de Ruptura.
    * Utiliza la técnica **Minimum Required Line Segments (MRLS)** para extender los segmentos hasta que el error de interpolación exceda $\epsilon$.
    * **El modelo final** ($P$, $O$, $X_{fin}$) es generado y se devuelve el error máximo real ($Error_{max}$) alcanzado.

### Guía Rápida de Uso

El modelo se entrena y utiliza a través de las funciones `train_slrm` y `predict_slrm` en el archivo `slrm-logos.py`.

#### Requisitos:

Es necesario tener instalado Python y la librería NumPy.

```bash
pip install numpy
```

#### 1. Entrenamiento (V5.12)

Define tus datos de entrada y la tolerancia ($\epsilon$).

```python
from slrm_logos import train_slrm

INPUT_DATA = [
    [-8.00, -4.00], [4.00, 5.0], [6.00, 18.0], [10.00, 27.00]
    # ... más puntos
]

# Entrenar el modelo. Devuelve el diccionario del modelo final, los puntos originales y el error máximo alcanzado.
final_model, original_points, max_error_achieved = train_slrm(INPUT_DATA, epsilon=0.03) 

# El resultado 'final_model' es un diccionario {X_inicio: [P, O, X_fin]} 
# que define explícitamente cada segmento.
```

#### 2. Predicción (V5.12)

Utiliza el diccionario del modelo (`final_model`) para obtener predicciones para cualquier punto X. El caché LRU se gestiona internamente dentro de la función `predict_slrm`.

```python
from slrm_logos import predict_slrm

x_test = 4.5
result = predict_slrm(x_test, final_model, original_points)

print(f"Predicción para X={result['x_in']}: Y={result['y_pred']:.4f}")
print(f"Segmento Activo: P={result['slope_P']:.4f}, O={result['intercept_O']:.4f}")
print(f"Cache Hit: {result['cache_hit']}") # Muestra si la predicción provino del caché
```

### Recursos y Scripts de Referencia:

* [Repositorio Principal del Proyecto](https://github.com/akinetic/neural-network/)
* [Script de Producción SLRM ES (slrm-logos-es.py)](slrm-logos-es.py)
* [Reporte de Rendimiento y Eficiencia (slrm\_reporte\_rendimiento.md)](slrm_reporte_rendimiento.md)
