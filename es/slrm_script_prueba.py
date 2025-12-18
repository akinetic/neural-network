# ==============================================================================
# NOTEBOOK DE GOOGLE COLAB: COMPARACIÓN SLRM vs. RLS/POLI/DT
# ==============================================================================
# Este archivo simula un Notebook de Colab en Python para comparar el rendimiento
# de nuestro algoritmo SLRM contra modelos estándar de scikit-learn, incluyendo
# Regresión Lineal Simple (RLS), Regresión Polinomial y Árbol de Decisión (DT).
#
# OBJETIVOS:
# 1. Integrar el núcleo del SLRM (V5.12) para garantizar la consistencia.
# 2. Entrenar y predecir con RLS, Polinomial y Decision Tree de scikit-learn.
# 3. Calcular métricas clave (MSE, R2, Segmentos SLRM, Tasa de Compresión).
# 4. Visualizar los resultados comparativos para la Hoja de Decisión.
# ==============================================================================

# --- CELDAS 0 & 1: INSTALACIONES E IMPORTACIONES ---
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import List, Tuple, Dict, Any, Optional
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor # Importación necesaria

# ==============================================================================
# --- SLRM CORE V5.12 (CÓDIGO INTEGRADO) ---
# Se mantiene el código del SLRM (train_slrm, predict_slrm, utilidades)
# para asegurar la funcionalidad central del modelo que se está probando.
# ==============================================================================

# Definición de tipos para el modelo SLRM
SLRMModel = Dict[float, List[float]]

# --- CONSTANTES GLOBALES ---
EPSILON = 0.50
CACHE_SIZE = 100
FLOAT_TOLERANCE = 1e-9

# --- 1. CACHÉ DE PREDICCIÓN (Caché LRU) ---

class LRUCache:
    """Caché LRU para la función de predicción SLRM."""
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: float) -> Optional[Dict[str, Any]]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: float, value: Dict[str, Any]):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = value

_prediction_cache = LRUCache(CACHE_SIZE)

# ==============================================================================
# 2. FUNCIONES DE UTILIDAD Y PREPROCESAMIENTO
# ==============================================================================

def _clean_and_sort_data(data_string: str) -> List[Tuple[float, float]]:
    """Limpia, parsea y ordena los datos, manejando duplicados de X promediando Y."""
    points_map: Dict[float, Tuple[float, int]] = {}

    for line in data_string.strip().split('\n'):
        parts = line.strip().replace(',', ' ').split()
        if len(parts) >= 2:
            try:
                x = float(parts[0])
                y = float(parts[1])

                if x in points_map:
                    current_y, count = points_map[x]
                    new_y = (current_y * count + y) / (count + 1)
                    points_map[x] = (new_y, count + 1)
                else:
                    points_map[x] = (y, 1)
            except ValueError:
                continue

    cleaned_data = [(x, y_count[0]) for x, y_count in points_map.items()]
    cleaned_data.sort(key=lambda p: p[0])

    return cleaned_data

# ==============================================================================
# 3. FUNCIONES DE COMPRESIÓN (NÚCLEO LOGOS V5.12)
# ==============================================================================

def _lossless_compression(data: List[Tuple[float, float]]) -> List[float]:
    """Compresión Sin Pérdida (Invariancia Geométrica)."""
    if len(data) < 3:
        return [p[0] for p in data]

    critical_x = [data[0][0]]

    for i in range(1, len(data) - 1):
        p0, p1, p2 = data[i - 1], data[i], data[i + 1]

        dx_a = p1[0] - p0[0]
        dx_b = p2[0] - p1[0]

        if dx_a != 0 and dx_b != 0:
            P_a = (p1[1] - p0[1]) / dx_a
            P_b = (p2[1] - p1[1]) / dx_b

            if abs(P_a - P_b) > FLOAT_TOLERANCE:
                critical_x.append(p1[0])
        else:
             critical_x.append(p1[0])

    if len(data) > 1:
        critical_x.append(data[-1][0])

    return sorted(list(set(critical_x)))


def _lossy_compression(initial_keys: List[float], epsilon: float, data: List[Tuple[float, float]]) -> Tuple[SLRMModel, float]:
    """Compresión con Pérdida (MRLS - Segmentos de Línea Mínimos Requeridos)."""
    if len(initial_keys) < 2:
        return {}, 0.0

    data_map = {x: y for x, y in data}
    data_x_list = [x for x, y in data]

    epsilon_threshold = max(epsilon, 1e-12) if epsilon == 0 else epsilon

    final_model: SLRMModel = {}
    i = 0
    max_overall_error = 0.0

    def _calculate_segment_max_error(x_s, x_e, P, O, data_x_list, data):
        """Ayudante para calcular el error máximo de un segmento COMPROMETIDO."""
        if math.isnan(P) or math.isnan(O):
            return 0.0

        start_idx = data_x_list.index(x_s)
        end_idx = data_x_list.index(x_e)
        max_err = 0.0

        for k in range(start_idx + 1, end_idx):
            x_mid, y_true_mid = data[k]

            y_hat_mid = P * x_mid + O
            error = abs(y_true_mid - y_hat_mid)

            max_err = max(max_err, error)
        return max_err

    while i < len(initial_keys) - 1:

        x_start = initial_keys[i]
        y_start = data_map[x_start]

        j = i + 1
        current_test_max_error = 0.0

        while j < len(initial_keys):
            x_end_candidate = initial_keys[j]
            y_end_candidate = data_map[x_end_candidate]

            dx = x_end_candidate - x_start

            if dx == 0:
                P_test, O_test = np.nan, np.nan
            else:
                P_test = (y_end_candidate - y_start) / dx
                O_test = y_start - P_test * x_start

            error_exceeded = False

            start_index = data_x_list.index(x_start)
            end_index = data_x_list.index(x_end_candidate)

            current_test_max_error = 0.0

            for k in range(start_index + 1, end_index):
                x_mid, y_true_mid = data[k]

                if math.isnan(P_test):
                    error = abs(y_true_mid - y_start)
                else:
                    y_hat_mid = P_test * x_mid + O_test
                    error = abs(y_true_mid - y_hat_mid)

                current_test_max_error = max(current_test_max_error, error)

                if error > epsilon_threshold:
                    error_exceeded = True
                    break

            if error_exceeded:
                x_end_committed = initial_keys[j - 1]
                y_end_committed = data_map[x_end_committed]

                dx_committed = x_end_committed - x_start
                if dx_committed == 0:
                    P, O = np.nan, np.nan
                else:
                    P = (y_end_committed - y_start) / dx_committed
                    O = y_start - P * x_start

                final_model[x_start] = [P, O, x_end_committed]

                committed_segment_max_error = _calculate_segment_max_error(x_start, x_end_committed, P, O, data_x_list, data)
                max_overall_error = max(max_overall_error, committed_segment_max_error)

                i = j - 1
                break

            elif j == len(initial_keys) - 1:
                x_end = initial_keys[j]
                y_end = data_map[x_end]

                dx = x_end - x_start
                if dx == 0:
                    P, O = np.nan, np.nan
                else:
                    P = (y_end - y_start) / dx
                    O = y_start - P * x_start

                final_model[x_start] = [P, O, x_end]

                max_overall_error = max(max_overall_error, current_test_max_error)

                i = j
                break

            j += 1

    if initial_keys:
        last_key = initial_keys[-1]
        if last_key not in final_model:
            final_model[last_key] = [np.nan, np.nan, np.nan]

    return final_model, max_overall_error

# ==============================================================================
# 4. FUNCIONES PRINCIPALES DE ENTRENAMIENTO Y PREDICCIÓN
# ==============================================================================

def train_slrm(input_data_string: str, epsilon: float = EPSILON) -> Tuple[SLRMModel, List[Tuple[float, float]], float]:
    """Entrena el SLRM a partir de la data de entrada."""
    global _prediction_cache

    original_points = _clean_and_sort_data(input_data_string)

    if len(original_points) < 2:
        _prediction_cache = LRUCache(CACHE_SIZE)
        return {}, original_points, 0.0

    initial_breakpoints_x = _lossless_compression(original_points)
    final_model, max_error = _lossy_compression(initial_breakpoints_x, epsilon, original_points)

    _prediction_cache = LRUCache(CACHE_SIZE)

    return final_model, original_points, max_error


def predict_slrm(x_in: float, slrm_model: SLRMModel, original_points: List[Tuple[float, float]]) -> Dict[str, Any]:
    """Predice el valor Y para un X de entrada usando el modelo SLRM."""
    if not slrm_model or not original_points:
        return {'x_in': x_in, 'y_pred': np.nan, 'slope_P': np.nan, 'intercept_O': np.nan, 'cache_hit': False}

    cached_result = _prediction_cache.get(x_in)
    if cached_result is not None:
        cached_result['cache_hit'] = True
        return cached_result

    segment_starts = sorted([x for x, segment in slrm_model.items() if not math.isnan(segment[0])])

    if not segment_starts:
        return {'x_in': x_in, 'y_pred': np.nan, 'slope_P': np.nan, 'intercept_O': np.nan, 'cache_hit': False}

    min_x = original_points[0][0]
    max_x = original_points[-1][0]

    active_key = None

    if x_in < min_x:
        # Extrapolación Izquierda: usar el primer segmento
        active_key = segment_starts[0]
    elif x_in >= max_x:
        # Extrapolación Derecha o punto final exacto: usar el último segmento
        active_key = segment_starts[-1]
    else:
        # Interpolación: encontrar el segmento donde x_inicio <= x_in < x_fin
        for x_start in segment_starts:
            x_end = slrm_model[x_start][2]
            if x_in >= x_start and x_in < x_end:
                active_key = x_start
                break

    if active_key is None:
        P, O = np.nan, np.nan
    else:
        P, O, _ = slrm_model[active_key]

    y_pred = x_in * P + O if not math.isnan(P) else np.nan

    result = {
        'x_in': x_in,
        'y_pred': y_pred,
        'slope_P': P,
        'intercept_O': O,
        'cache_hit': False
    }

    _prediction_cache.put(x_in, result)

    return result

# ==============================================================================
# --- CELDA 2: PREPARACIÓN DE DATOS Y ENTRENAMIENTO SLRM ---
# ==============================================================================

# Datos de muestra (los mismos usados en tu visualizador V5.8)
SAMPLE_DATA_STRING = """
1, 1
2, 1.5
3, 1.7
4, 3.5
5, 5
6, 4.8
7, 4.5
8, 4.3
9, 4.1
10, 4.2
11, 4.3
12, 4.6
13, 5.5
14, 7
15, 8.5
"""

# Parámetro de tolerancia para el SLRM
SLRM_EPSILON = 0.5

# 1. Ejecutar el entrenamiento SLRM
slrm_model, original_points, slrm_max_error = train_slrm(SAMPLE_DATA_STRING, SLRM_EPSILON)

# Convertir los puntos originales a arrays de numpy para scikit-learn
X = np.array([p[0] for p in original_points]).reshape(-1, 1)
Y = np.array([p[1] for p in original_points])
N = len(original_points) # Número total de puntos

# 2. SLRM: Calcular Tasa de Compresión y Segmentos
slrm_segments = sum(1 for P, O, X_end in slrm_model.values() if not math.isnan(P))
slrm_breakpoints = slrm_segments + 1 if slrm_segments > 0 else 0
compression_rate = (N - slrm_breakpoints) / N * 100 if N > 0 else 0

print("--- RESULTADOS DEL SLRM (Nuestro Modelo) ---")
print(f"Puntos Originales (N): {N}")
print(f"Segmentos SLRM: {slrm_segments}")
print(f"Máximo Error (Epsilon): {slrm_max_error:.4f} (Objetivo: <= {SLRM_EPSILON})")
print(f"Tasa de Compresión: {compression_rate:.2f}% (Se usaron solo {slrm_breakpoints} puntos clave)")

# ==============================================================================
# --- CELDA 3: ENTRENAMIENTO DE MODELOS ESTÁNDAR (Scikit-learn) ---
# ==============================================================================

# --- Modelo A: Regresión Lineal Simple (RLS) ---
rls_model = LinearRegression()
rls_model.fit(X, Y)
Y_pred_rls = rls_model.predict(X)

# --- Modelo B: Regresión Polinomial (Grado 3 para un ajuste más cercano) ---
poly_features = PolynomialFeatures(degree=3)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, Y)
Y_pred_poly = poly_model.predict(X_poly)
# El modelo Polinomial tiene 4 coeficientes (grado 3 + intercepto)

# --- Modelo C: Regresión con Árboles de Decisión (Decision Tree) ---
# Usamos una profundidad máxima para evitar sobreajuste extremo
dt_model = DecisionTreeRegressor(max_depth=5)
dt_model.fit(X, Y)
Y_pred_dt = dt_model.predict(X)
# dt_complexity = dt_model.get_depth() + 1 if dt_model.get_depth() else 2 # Métrica Original : Profundidad del árbol + 1 (mínimo 2)
dt_complexity = dt_model.tree_.n_leaves # Métrica Corregida : Las cuentas echan hojas los nodos (las regiones de la predicción), asegurando una comparación estructural justa con los segmentos de SLRM.

# ==============================================================================
# --- CELDA 4: PREDICCIÓN Y CÁLCULO DE MÉTRICAS (HOJA DE DECISIÓN) ---
# ==============================================================================

# Generar puntos de predicción para el SLRM para el gráfico
X_plot = X.flatten()
Y_pred_slrm = np.array([predict_slrm(x, slrm_model, original_points)['y_pred'] for x in X_plot])

# --- Métricas para RLS ---
mse_rls = mean_squared_error(Y, Y_pred_rls)
r2_rls = r2_score(Y, Y_pred_rls)

# --- Métricas para Polinomial ---
mse_poly = mean_squared_error(Y, Y_pred_poly)
r2_poly = r2_score(Y, Y_pred_poly)

# --- Métricas para Árbol de Decisión ---
mse_dt = mean_squared_error(Y, Y_pred_dt)
r2_dt = r2_score(Y, Y_pred_dt)

# --- Métricas para SLRM ---
mse_slrm = mean_squared_error(Y, Y_pred_slrm)
r2_slrm = r2_score(Y, Y_pred_slrm)


print("\n--- COMPARATIVA DE MÉTRICAS DE ERROR (Hoja de Decisión) ---")
print(f"| Modelo | MSE (Error Cuadrático Medio) | R2 (Coeficiente de Determinación) | Parámetros del Modelo |")
print(f"|:---|:---|:---|:---|")
print(f"| **SLRM (Segmentado)** | {mse_slrm:.4f} | {r2_slrm:.4f} | {slrm_breakpoints} (Puntos Clave) |")
print(f"| RLS (Lineal Simple) | {mse_rls:.4f} | {r2_rls:.4f} | 2 (Pendiente + Intercepto) |")
print(f"| Polinomial (Grado 3) | {mse_poly:.4f} | {r2_poly:.4f} | 4 (Coeficientes) |")
print(f"| Árbol de Decisión (Profundidad Máx 5) | {mse_dt:.4f} | {r2_dt:.4f} | {dt_complexity} (Niveles/Profundidad) |")


print("\n--- CONCLUSIÓN ---")
print(f"El SLRM logró un R2 de {r2_slrm:.4f} con una compresión del {compression_rate:.2f}%.")
print(f"El Árbol de Decisión (Decision Tree) logró un R2 de {r2_dt:.4f} con {dt_complexity} niveles, superando al RLS y Polinomial en precisión,")
print(f"pero lo hizo con un modelo de estructura jerárquica compleja, mientras que el SLRM lo logra con la simplicidad geométrica de {slrm_segments} segmentos lineales.")


# ==============================================================================
# --- CELDA 5: VISUALIZACIÓN DE RESULTADOS ---
# ==============================================================================

plt.figure(figsize=(12, 6))

# 1. Puntos Originales
plt.scatter(X, Y, color='black', label='Puntos Originales ($N=15$)', zorder=2)

# 2. Predicción RLS
plt.plot(X, Y_pred_rls, color='orange', linestyle='--', label=f'RLS (R2={r2_rls:.2f})', linewidth=1.5, zorder=1)

# 3. Predicción Polinomial (Grado 3)
plt.plot(X, Y_pred_poly, color='red', linestyle=':', label=f'Polinomial G3 (R2={r2_poly:.2f})', linewidth=1.5, zorder=1)

# 4. Predicción Árbol de Decisión
plt.plot(X, Y_pred_dt, color='green', linestyle='-.', label=f'Árbol de Decisión (R2={r2_dt:.2f})', linewidth=1.5, zorder=1)

# 5. Predicción SLRM Segmentada
# Dibujar cada segmento del SLRM
segment_keys = sorted([x for x, seg in slrm_model.items() if not math.isnan(seg[0])])

for x_start in segment_keys:
    P, O, x_end = slrm_model[x_start]

    # Puntos X del segmento
    X_segment = np.array([x_start, x_end])
    # Puntos Y predichos
    Y_segment_pred = P * X_segment + O

    plt.plot(X_segment, Y_segment_pred, color='#0077B6', linewidth=4, label='SLRM' if x_start == segment_keys[0] else '', zorder=3)
    # Dibujar los breakpoints para hacerlos visibles
    plt.scatter([x_start, x_end], [P * x_start + O, P * x_end + O], color='#0077B6', marker='o', s=60, zorder=4)


plt.title(f'Comparación de Ajuste: SLRM (Epsilon={SLRM_EPSILON}) vs. Modelos Estándar', fontsize=16)
plt.xlabel('Variable X', fontsize=14)
plt.ylabel('Variable Y', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
