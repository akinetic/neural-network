# slrm-logos-es.py
# Modelo de Regresión Lineal Segmentada (SLRM) - Núcleo Logos
# Versión: V5.12 (Versión Final Verificada)
# Autores: Alex Kinetic y Logos
#
# Implementación completa del proceso de entrenamiento (compresión) y
# predicción optimizada del SLRM. Utiliza la robusta compresión bifásica:
# Sin Pérdida (Invarianza Geométrica) seguida de Con Pérdida (MRLS, Criterio Humano).
# Incorpora un caché LRU para la velocidad de predicción.

import numpy as np
import math
from collections import OrderedDict
from typing import List, Tuple, Dict, Any, Optional

# Definición de tipos para el modelo SLRM
SLRMModel = Dict[float, List[float]]

# --- CONSTANTES GLOBALES ---
# Tolerancia de Error por Defecto (Epsilon) para la compresión con pérdida.
EPSILON = 0.50
# Tamaño del Caché LRU para la función de predicción.
CACHE_SIZE = 100
# Tolerancia utilizada para comparaciones de coma flotante (Invarianza Geométrica o Epsilon=0)
FLOAT_TOLERANCE = 1e-9

# --- 1. CACHÉ DE PREDICCIÓN (Caché LRU) ---

class LRUCache:
    """
    Caché Simple Menos Recientemente Usado (LRU) optimizado para la predicción SLRM.
    Almacena las últimas predicciones para evitar búsquedas repetidas de segmentos.
    """
    def __init__(self, capacity: int):
        # OrderedDict mantiene el orden de inserción, útil para la política LRU.
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: float) -> Optional[Dict[str, Any]]:
        """Recupera un valor y lo mueve al final (más reciente)."""
        if key not in self.cache:
            return None
        # Mueve la clave al final para marcarla como usada recientemente
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: float, value: Dict[str, Any]):
        """Inserta o actualiza un valor. Si se excede la capacidad, elimina el elemento menos recientemente usado."""
        if key in self.cache:
            # CORRECCIÓN CRÍTICA : Mover la CLAVE, no el VALOR.
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # Eliminar el primer elemento (el menos reciente)
                self.cache.popitem(last=False)
            self.cache[key] = value

# Inicializar el caché global.
_prediction_cache = LRUCache(CACHE_SIZE)

# ==============================================================================
# 2. FUNCIONES DE UTILIDAD DE ENTRENAMIENTO Y PREPROCESAMIENTO
# ==============================================================================

def _clean_and_sort_data(data_string: str) -> List[Tuple[float, float]]:
    """
    Analiza y limpia la cadena de datos de entrada.
    1. Maneja el formato (comas, espacios).
    2. Ordena por el valor X.
    3. Purifica: Maneja duplicados de X promediando sus valores Y.
    Devuelve una lista limpia y ordenada de tuplas (X, Y).
    """
    points_map: Dict[float, Tuple[float, int]] = {}
    
    for line in data_string.strip().split('\n'):
        # Separar por coma o espacio
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
    """
    Sección IV: Compresión Sin Pérdida (Invarianza Geométrica).
    Elimina puntos intermedios colineales con sus vecinos.
    Devuelve una lista de puntos de quiebre X críticos.
    """
    if len(data) < 3:
        return [p[0] for p in data]

    critical_x = [data[0][0]]
    
    for i in range(1, len(data) - 1):
        p0, p1, p2 = data[i - 1], data[i], data[i + 1]

        dx_a = p1[0] - p0[0]
        dx_b = p2[0] - p1[0]

        # Usar verificación absoluta de colinealidad si los segmentos no son verticales
        if dx_a != 0 and dx_b != 0:
            P_a = (p1[1] - p0[1]) / dx_a
            P_b = (p2[1] - p1[1]) / dx_b

            # Criterio: Si las pendientes NO son iguales (usando tolerancia), es un punto de quiebre.
            if abs(P_a - P_b) > FLOAT_TOLERANCE:
                critical_x.append(p1[0])
        else:
             # Caso de segmentos verticales o puntos coincidentes (ya manejados por la limpieza)
             critical_x.append(p1[0])
    
    # Siempre conservar el último punto
    if len(data) > 1:
        critical_x.append(data[-1][0])

    return sorted(list(set(critical_x)))


def _lossy_compression(initial_keys: List[float], epsilon: float, data: List[Tuple[float, float]]) -> Tuple[SLRMModel, float]:
    """
    Sección V: Compresión Con Pérdida (MRLS - Segmentos de Línea Mínimos Requeridos).
    Encuentra el segmento más largo posible desde cada punto de quiebre que respete el épsilon.
    Devuelve: (Modelo SLRM: {X_inicio: [P, O, X_fin]}, Máximo_Error_Alcanzado)
    """
    if len(initial_keys) < 2:
        return {}, 0.0

    data_map = {x: y for x, y in data}
    data_x_list = [x for x, y in data]
    
    # Lógica de Épsilon Crítico: Si el usuario establece epsilon=0, forzamos la verificación estricta (1e-12).
    epsilon_threshold = max(epsilon, 1e-12) if epsilon == 0 else epsilon

    final_model: SLRMModel = {}
    i = 0  # Índice del punto de quiebre inicial en initial_keys
    max_overall_error = 0.0
    
    def _calculate_segment_max_error(x_s, x_e, P, O, data_x_list, data):
        """Ayudante para calcular el error máximo de un segmento COMPROMETIDO."""
        # Verificar si P y O están calculados (es decir, no son NaN)
        if math.isnan(P) or math.isnan(O):
            return 0.0 
            
        start_idx = data_x_list.index(x_s)
        end_idx = data_x_list.index(x_e)
        max_err = 0.0
        
        # Iterar sobre los puntos intermedios (excluyendo el punto inicial y final)
        for k in range(start_idx + 1, end_idx):
            x_mid, y_true_mid = data[k]
            
            y_hat_mid = P * x_mid + O
            error = abs(y_true_mid - y_hat_mid)
            
            max_err = max(max_err, error)
        return max_err

    while i < len(initial_keys) - 1:
        
        x_start = initial_keys[i]
        y_start = data_map[x_start]
        
        j = i + 1  # Índice del punto de quiebre final candidato (x_end_candidate)
        
        current_test_max_error = 0.0 

        while j < len(initial_keys):
            x_end_candidate = initial_keys[j]
            y_end_candidate = data_map[x_end_candidate]

            dx = x_end_candidate - x_start
            
            # 1. Calcular la línea de prueba (P_test, O_test)
            if dx == 0:
                P_test, O_test = np.nan, np.nan
            else:
                P_test = (y_end_candidate - y_start) / dx
                O_test = y_start - P_test * x_start
            
            error_exceeded = False
            
            # Encontrar índices de puntos para la verificación de límites
            start_index = data_x_list.index(x_start)
            end_index = data_x_list.index(x_end_candidate)

            # 2. Verificar todos los puntos intermedios contra la línea de prueba (i -> j)
            # Restablecer el seguimiento del error máximo para el NUEVO segmento de línea de prueba (i -> j)
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
                # El Segmento i -> j FALLÓ. Comprometer el segmento válido anterior (i -> j-1).
                x_end_committed = initial_keys[j - 1]
                y_end_committed = data_map[x_end_committed]
                
                # Recalcular P y O para el segmento COMPROMETIDO (i -> j-1)
                dx_committed = x_end_committed - x_start
                if dx_committed == 0:
                    P, O = np.nan, np.nan
                else:
                    P = (y_end_committed - y_start) / dx_committed
                    O = y_start - P * x_start
                
                final_model[x_start] = [P, O, x_end_committed]
                
                # CORRECCIÓN CRÍTICA (Lógica V5.9): Recalcular el error máximo para el segmento COMPROMETIDO (i -> j-1)
                committed_segment_max_error = _calculate_segment_max_error(x_start, x_end_committed, P, O, data_x_list, data)
                max_overall_error = max(max_overall_error, committed_segment_max_error)

                i = j - 1 # El siguiente segmento comienza en j-1 (el punto final comprometido)
                break 
                
            elif j == len(initial_keys) - 1:
                # Alcanzado el último punto. Comprometer el segmento (i -> j).
                x_end = initial_keys[j]
                y_end = data_map[x_end]
                
                dx = x_end - x_start
                if dx == 0:
                    P, O = np.nan, np.nan
                else:
                    P = (y_end - y_start) / dx
                    O = y_start - P * x_start
                    
                final_model[x_start] = [P, O, x_end]
                
                # El error máximo para este segmento final válido se almacena en current_test_max_error
                max_overall_error = max(max_overall_error, current_test_max_error)
                
                i = j # El ciclo termina
                break
            
            j += 1 # Intentar extender el segmento aún más

    # Agregar el marcador final NaN si el ciclo no lo agregó explícitamente
    if initial_keys:
        last_key = initial_keys[-1]
        if last_key not in final_model:
            final_model[last_key] = [np.nan, np.nan, np.nan]

    return final_model, max_overall_error

# ==============================================================================
# 4. FUNCIONES PRINCIPALES DE ENTRENAMIENTO Y PREDICCIÓN
# ==============================================================================

def train_slrm(input_data_string: str, epsilon: float = EPSILON) -> Tuple[SLRMModel, List[Tuple[float, float]], float]:
    """
    Entrena el Modelo de Regresión Lineal Segmentada (SLRM) a partir de los datos.

    Argumentos:
        input_data_string (str): Puntos de datos (X, Y) separados por líneas.
        epsilon (float): Tolerancia de error máxima para la compresión con pérdida.

    Devuelve:
        Tupla: (Modelo SLRM, Puntos Originales Limpiados, Máximo Error Alcanzado)
    """
    global _prediction_cache 
    
    # 1. Limpieza y Ordenamiento
    original_points = _clean_and_sort_data(input_data_string)
    
    if len(original_points) < 2:
        _prediction_cache = LRUCache(CACHE_SIZE)
        return {}, original_points, 0.0
        
    # 2. Compresión Sin Pérdida (Invarianza Geométrica)
    initial_breakpoints_x = _lossless_compression(original_points)
    
    # 3. Compresión Con Pérdida (MRLS)
    final_model, max_error = _lossy_compression(initial_breakpoints_x, epsilon, original_points)
    
    # Limpiar el caché de predicción al entrenar un nuevo modelo (CRÍTICO para modelos nuevos)
    _prediction_cache = LRUCache(CACHE_SIZE)
    
    return final_model, original_points, max_error


def predict_slrm(x_in: float, slrm_model: SLRMModel, original_points: List[Tuple[float, float]]) -> Dict[str, Any]:
    """
    Predice el valor Y para una entrada X utilizando el modelo SLRM comprimido.
    """
    if not slrm_model or not original_points:
        return {'x_in': x_in, 'y_pred': np.nan, 'slope_P': np.nan, 'intercept_O': np.nan, 'cache_hit': False}

    # Intentar obtener la predicción del caché
    cached_result = _prediction_cache.get(x_in)
    if cached_result is not None:
        cached_result['cache_hit'] = True
        return cached_result
        
    # Obtener claves para segmentos válidos (aquellos con P calculado)
    segment_starts = sorted([x for x, segment in slrm_model.items() if not math.isnan(segment[0])])
    
    if not segment_starts:
        return {'x_in': x_in, 'y_pred': np.nan, 'slope_P': np.nan, 'intercept_O': np.nan, 'cache_hit': False}

    min_x = original_points[0][0]
    max_x = original_points[-1][0]
    
    active_key = None

    if x_in < min_x:
        # Extrapolación (Izquierda): usar el primer segmento
        active_key = segment_starts[0]
    elif x_in >= max_x:
        # Extrapolación (Derecha) o punto final exacto: usar el último segmento
        active_key = segment_starts[-1]
    else:
        # Interpolación: encontrar el segmento donde x_inicio <= x_in < x_fin
        for x_start in segment_starts:
            x_end = slrm_model[x_start][2] # X_fin del segmento
            
            # Usar X_fin del segmento. Si x_in es igual al último punto, cae en el caso max_x anterior.
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

    # Guardar en caché
    _prediction_cache.put(x_in, result)
    
    return result

# ==============================================================================
# 5. EJEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    import time
    
    # Datos utilizados en el visualizador interactivo V5.8
    SAMPLE_DATA = """
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
    
    print("--- Ejemplo de Entrenamiento y Predicción SLRM (V5.12 - Refinamiento de Prueba) ---")
    
    # --------------------------------------------------------------------------
    # PRUEBA 1: Compresión Con Pérdida (epsilon=0.5)
    # --------------------------------------------------------------------------
    epsilon_test = 0.5
    print(f"\n[PRUEBA 1] Entrenando con Épsilon = {epsilon_test:.6f}")
    
    start_time = time.time()
    # Nota: train_slrm limpia el caché automáticamente
    model, points, max_error = train_slrm(SAMPLE_DATA, epsilon_test)
    training_duration = time.time() - start_time
    
    segment_count = sum(1 for P, O, X_end in model.values() if not math.isnan(P))
    breakpoint_count = segment_count + 1 if segment_count > 0 else 0
    
    print(f"Tiempo Empleado: {training_duration:.4f} segundos.")
    print(f"Puntos Originales: {len(points)}")
    print(f"Puntos de Quiebre Finales: {breakpoint_count}")
    print(f"Segmentos Generados: {segment_count}")
    print(f"Máximo Error Alcanzado: {max_error:.7f}") 
    
    print("\nResultado del Modelo (X_inicio: [P, O, X_fin]):")
    for x_start, segment in model.items():
        if not math.isnan(segment[0]):
            print(f"  {x_start:+.2f}: P={segment[0]:+.4f}, O={segment[1]:+.4f}, X_fin={segment[2]:+.2f}")
    
    # Prueba de Predicción
    X_TEST_VALUES = [0.0, 5.5, 9.5, 15.0, 16.0]
    print("\nPrueba de Predicción:")
    
    # Ejecución 1: Todos los Fallos de Caché (Rellena el Caché)
    print("--- Ejecución 1 (Rellenando Caché) ---")
    for x in X_TEST_VALUES:
        result = predict_slrm(x, model, points)
        print(f"Predecir X={result['x_in']:+.2f} | Y={result['y_pred']:+.6f} | Segmento Activo P={result['slope_P']:+.4f} | Caché: {'Acertado' if result['cache_hit'] else 'Fallado'}")

    # Ejecución 2: Se esperan Aciertos de Caché
    print("--- Ejecución 2 (Probando Aciertos) ---")
    for x in X_TEST_VALUES:
        result = predict_slrm(x, model, points)
        print(f"Predecir X={result['x_in']:+.2f} | Y={result['y_pred']:+.6f} | Segmento Activo P={result['slope_P']:+.4f} | Caché: {'Acertado' if result['cache_hit'] else 'Fallado'}")
        
    print(f"\n[INFO] Estado del Caché LRU después de la Prueba 1 (Tamaño: {len(_prediction_cache.cache)}/{CACHE_SIZE})")

    # --------------------------------------------------------------------------
    # PRUEBA 2: Compresión Sin Pérdida (epsilon=0)
    # --------------------------------------------------------------------------
    epsilon_zero = 0.0
    print(f"\n[PRUEBA 2] Entrenando con Épsilon = {epsilon_zero:.6f} (Forzando Invarianza Geométrica)")
    
    # train_slrm aquí limpia el caché
    model_zero, _, max_error_zero = train_slrm(SAMPLE_DATA, epsilon_zero)
    
    segment_count_zero = sum(1 for P, O, X_end in model_zero.values() if not math.isnan(P))
    
    print(f"Segmentos Generados: {segment_count_zero}")
    print(f"Máximo Error Alcanzado: {max_error_zero:.7f} (Debería ser cercano a cero)")

    # Verificación final del Estado del Caché después del reinicio por train_slrm
    print(f"\n[INFO] Estado del Caché LRU después del Entrenamiento de la Prueba 2 (Caché reiniciado por train_slrm): {len(_prediction_cache.cache)}/{CACHE_SIZE}")
