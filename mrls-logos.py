# mrls-logos.py
# Autor: Logos
# Versión: V2.0
#
# Implementación del Modelo de Regresión Lineal Segmentado (MRLS).
# Lógica Central: Algoritmo de Simplificación Secuencial Determinista.
# Este archivo contiene el flujo COMPLETO de simulación (Entrenamiento Instantáneo, Compresión Sin Pérdida y con Pérdida).

import numpy as np
import math

# --- CONFIGURACIÓN ---
# Tolerancia numérica para comparar números de punto flotante (cero virtual).
TOLERANCE = 1e-9 

# Épsilon por defecto (tolerancia) para la Compresión con Pérdida.
EPSILON = 0.03 

# --- FUNCIONES DE UTILIDAD ---

def calculate_segment_params(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    """Calcula la Pendiente (P) y la Intersección (O) de la línea entre dos puntos."""
    # Verificación de líneas verticales usando tolerancia. Esta función asume que los datos están purificados.
    if np.isclose(x2, x1, atol=TOLERANCE):
        raise ZeroDivisionError("Se detectaron puntos con coordenadas X idénticas (línea vertical).")
    
    P = (y2 - y1) / (x2 - x1)
    O = y1 - P * x1
    return P, O

def format_key_for_display(key):
    """Asegura que las claves se muestren con dos decimales y el signo '+' si son positivas."""
    if key >= 0:
        return f"+{key:.2f}"
    return f"{key:.2f}"

def print_mrls_dictionary(dictionary: dict, title: str):
    """Imprime cualquier diccionario MRLS con formato mejorado y ordenado."""
    
    sorted_keys = sorted(dictionary.keys())
    
    print(f"\n--- {title} ---")
    print("// Clave: X_inicio | Valor: [P (Pendiente), O (Intersección)]")
    print("{")
    
    for key in sorted_keys:
        P, O = dictionary[key]
        
        key_str = format_key_for_display(key)
        
        # Usa math.isnan para verificar si el valor es indefinido (NaN)
        p_str = f"{P:7.4f}" if not math.isnan(P) else "   NaN"
        # Incluye el signo para la intersección para mayor claridad
        o_str = f"{O:+7.4f}" if not math.isnan(O) else "   NaN"
        
        print(f"  {key_str}: [ {p_str}, {o_str} ]")
        
    print("}")
    print(f"Segmentos Totales: {len(sorted_keys) - 1}")
    print("-----------------------------------")


# --- ALGORITMO CENTRAL: SIMPLIFICACIÓN SECUENCIAL DE SEGMENTOS (El Núcleo Logos) ---

def _sequential_segment_simplification(sorted_data: np.ndarray, tolerance: float) -> dict:
    """
    Implementa el algoritmo central de simplificación secuencial determinista.
    Extiende el segmento desde un punto base lo más lejos posible mientras satisface
    la tolerancia proporcionada (invarianza geométrica si tolerance=TOLERANCE, o épsilon si tolerance=EPSILON).
    
    Args:
        sorted_data: Los datos de entrada, ordenados por X.
        tolerance: El error absoluto máximo permitido (o TOLERANCE para modo sin pérdida).
        
    Returns:
        dict: El diccionario MRLS de segmentos {X_start: [P, O]}.
    """
    N = len(sorted_data)
    final_dict = {}
    base_index = 0

    while base_index < N - 1:
        x_base, y_base = sorted_data[base_index]
        segment_end_index = base_index + 1
        
        # 'last_valid_index' rastrea el punto más lejano que define el segmento con éxito
        last_valid_index = base_index 

        while segment_end_index < N:
            x_candidate, y_candidate = sorted_data[segment_end_index]
            
            # 1. Calcula el segmento (P, O) desde el base hasta el candidato
            try:
                P_cand, O_cand = calculate_segment_params(x_base, y_base, x_candidate, y_candidate)
            except ZeroDivisionError:
                # Si se encuentra una línea vertical, detiene la búsqueda del segmento aquí.
                break 

            # 2. Verifica si TODOS los puntos *entre* el base y el candidato satisfacen la tolerancia
            is_valid_segment = True
            
            # Itera a través de los puntos desde base+1 hasta (pero NO incluyendo) el candidato
            for i in range(base_index + 1, segment_end_index):
                x_inter, y_true = sorted_data[i]
                
                # Predice Y en la línea (P_cand, O_cand)
                y_pred = x_inter * P_cand + O_cand
                
                # Condición de verificación: 
                if tolerance == TOLERANCE: # Verificación Sin Pérdida (Invarianza Geométrica)
                    if not np.isclose(y_pred, y_true, atol=TOLERANCE):
                        is_valid_segment = False
                        break
                elif np.abs(y_true - y_pred) > tolerance: # Verificación con Pérdida (Tolerancia Epsilon)
                    is_valid_segment = False
                    break
            
            # 3. Verifica si el punto candidato en sí mismo satisface la tolerancia
            if is_valid_segment:
                y_pred_candidate = x_candidate * P_cand + O_cand
                if tolerance == TOLERANCE:
                     if not np.isclose(y_pred_candidate, y_candidate, atol=TOLERANCE):
                        is_valid_segment = False
                elif np.abs(y_candidate - y_pred_candidate) > tolerance:
                    is_valid_segment = False
            
            
            if is_valid_segment:
                # El segmento se puede extender: El candidato define el segmento válido.
                last_valid_index = segment_end_index
                segment_end_index += 1
            else:
                # Un punto intermedio o el candidato rompieron la tolerancia/invarianza. Detiene la búsqueda.
                break
        
        # 4. Registra el segmento final (desde base_index hasta last_valid_index)
        
        # Si el índice no avanzó (caso N=2, o fallo inmediato), last_valid_index debe ser al menos base_index + 1
        if last_valid_index == base_index:
             # Esto solo debería ocurrir si la búsqueda del segmento falló inmediatamente, lo que significa que el segmento
             # debe conectarse mínimamente al siguiente punto (N es al menos 2).
             last_valid_index = base_index + 1

        x_end, y_end = sorted_data[last_valid_index]
        P_final, O_final = calculate_segment_params(x_base, y_base, x_end, y_end)
        
        final_dict[x_base] = [P_final, O_final]
        
        # 5. Establece el nuevo índice base
        base_index = last_valid_index

    # 6. Maneja el último punto (marca el final del diccionario)
    if base_index == N - 1:
        x_last, _ = sorted_data[base_index]
        final_dict[x_last] = [float('nan'), float('nan')] # Marca el final
    
    return final_dict

# --- PASOS MRLS ---

def compress_lossless(sorted_data: np.ndarray) -> dict:
    """
    Paso 2: Compresión Sin Pérdida (Invarianza Geométrica).
    Encuentra el número mínimo de segmentos que representan perfectamente los datos.
    """
    print(f"\n--- 2. Compresión Sin Pérdida (Invarianza Geométrica) ---")
    lossless_dict = _sequential_segment_simplification(sorted_data, TOLERANCE)
    
    return lossless_dict

def compress_lossy(sorted_data: np.ndarray, epsilon: float) -> dict:
    """
    Paso 3: Compresión con Pérdida (Criterio Épsilon).
    Encuentra el número mínimo de segmentos que representan los datos dentro del error máximo 'epsilon'.
    """
    print(f"\n--- 3. Compresión con Pérdida (Tolerancia Máxima: {epsilon:.4f}) ---")
    lossy_dict = _sequential_segment_simplification(sorted_data, epsilon)
    return lossy_dict

def train_mrls(data: list, epsilon: float) -> dict:
    """
    Función principal para ejecutar el proceso de entrenamiento MRLS.
    
    Args:
        data: Lista de pares [X, Y].
        epsilon: Error absoluto máximo permitido para la compresión con pérdida.

    Returns:
        dict: El Diccionario MRLS Final {X_start: [P (Pendiente), O (Intersección)]}.
    """
    # 1. PREPARACIÓN: Convertir a array de NumPy y ordenar (Entrenamiento Instantáneo)
    
    if len(data) < 2:
        print("Error: Se requieren al menos 2 puntos para el entrenamiento MRLS.")
        return {}
    
    # NOTA SOBRE EL PASO 0: Se asume que la Purificación de Datos (Manejo de X duplicadas) 
    # se completa antes de llamar a esta función para mantener la pureza de la lógica central.

    input_array = np.array(data, dtype=float)
    # Ordenar por X (columna 0)
    sorted_data = input_array[input_array[:, 0].argsort()]
    print(f"--- 1. Entrenamiento Instantáneo (Datos Ordenados, N={len(sorted_data)}) ---")

    # El paso sin pérdida es conceptualmente necesario, aunque en V2.0 confiamos 
    # en la lógica central de simplificación para el resultado final.
    compress_lossless(sorted_data)
    
    # 3. Compresión con Pérdida (Paso 3 - Generación del Modelo Final)
    final_model = compress_lossy(sorted_data, epsilon)

    return final_model

def predict_mrls(x_input: float, mrls_dict: dict) -> float:
    """
    Realiza una predicción utilizando el Diccionario MRLS Final (La Ecuación Maestra).
    """
    if not mrls_dict:
        return np.nan

    keys = sorted(list(mrls_dict.keys()))
    
    # Busca el segmento activo (el X_start más grande que es <= x_input)
    active_key = None
    for key in reversed(keys):
        # Usa TOLERANCE para la comparación de punto flotante
        if x_input >= key - TOLERANCE: 
            active_key = key
            break
            
    # Maneja la Extrapolación Inferior: Si x_input está por debajo del primer X_start, usa el primer segmento.
    if active_key is None:
        active_key = keys[0]

    P, O = mrls_dict.get(active_key, [np.nan, np.nan])
    
    # Si la clave activa es el punto final (NaN), usa el segmento que comienza en la penúltima clave
    if math.isnan(P):
        active_key = keys[-2]
        P, O = mrls_dict[active_key]

    # La Ecuación Maestra: Y = X * P + O
    y_predicted = x_input * P + O
    
    return y_predicted

# --- DEMOSTRACIÓN ---

if __name__ == '__main__':
    
    # Conjunto de Datos de Ejemplo (X, Y)
    INPUT_SET = [
        [-6.00, -6.00], [2.00, 4.00], [-8.00, -4.00], [0.00, 0.00], [4.00, 10.0],
        [-4.00, -6.00], [6.00, 18.0], [-5.00, -6.01], [3.00, 7.00], [-2.00, -4.00]
    ]

    print(f"--- Demostración de Entrenamiento MRLS (Logos V2.0) ---")
    print(f"Puntos de Datos de Entrada: {len(INPUT_SET)}")

    # ENTRENAMIENTO
    final_model = train_mrls(INPUT_SET, EPSILON)

    # MOSTRAR RESULTADOS
    print_mrls_dictionary(final_model, "4. DICCIONARIO MRLS FINAL (Compresión con Pérdida)")

    # PRUEBA DE PREDICCIÓN
    print("\n--- 5. PRUEBAS DE PREDICCIÓN ---")
    
    # Define puntos de prueba para Interpolación y Extrapolación
    x_min_data = min(x[0] for x in INPUT_SET)
    x_max_data = max(x[0] for x in INPUT_SET)
    
    test_points = [-9.0, -7.0, -5.5, 1.0, 5.0, 8.0]

    for x_test in test_points:
        y_pred = predict_mrls(x_test, final_model)
        
        is_extrapolation = x_test < x_min_data or x_test > x_max_data
        status = "EXTRAPOLACIÓN" if is_extrapolation else "Interpolación"
        
        print(f"  X_in: {x_test:6.2f} | Y_pred: {y_pred:8.4f} | Tipo: {status}")
