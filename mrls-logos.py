# -*- coding: utf-8 -*-
# mrls-logos.py: Implementación del Modelo de Regresión Lineal Segmentada (MRLS)
# con un algoritmo de compresión neuronal con pérdida (lossy compression).
#
# Este archivo contiene el flujo COMPLETO de simulación para generar el
# diccionario comprimido que se utiliza en la función de predicción.

import math

# --- FUNCIONES DE UTILIDAD Y DISPLAY ---

def format_key_for_display(key):
    """Asegura que las claves se muestren con dos decimales y el signo '+' si son positivas."""
    if key >= 0:
        return f"+{key:.2f}"
    return f"{key:.2f}"

def print_mrls_dictionary(dictionary, title="Diccionario MRLS"):
    """Imprime cualquier diccionario MRLS con formato mejorado y ordenado."""
    
    # Ordenar las claves numéricamente
    sorted_keys = sorted(dictionary.keys())
    
    print(f"\n--- {title} ---")
    print("// Clave: X_inicio | Valor: [Pendiente (P), Ordenada (O)]")
    print("{")
    
    for key in sorted_keys:
        P, O = dictionary[key]
        
        key_str = format_key_for_display(key)
        
        # Usamos math.isnan para verificar si el valor es indefinido (NaN)
        p_str = f"{P:7.2f}" if not math.isnan(P) else "   NaN"
        # Incluimos el signo en la ordenada para claridad en la impresión
        o_str = f"{O:+7.2f}" if not math.isnan(O) else "   NaN"
        
        print(f"  {key_str}: [ {p_str}, {o_str} ]")
        
    print("}")
    print("-----------------------------------")


# --- SECCIONES DE LA COMPRESIÓN MRLS ---

def create_base_dictionary(dataframe):
    """
    Sección I & II: Crea el Diccionario Base (Entrada: X, Salida: Y) ordenando 
    los datos de menor a mayor por la Entrada (X).
    """
    print("--- 1. Creando Diccionario Base (Entrenamiento Instantáneo) ---")
    # Aseguramos que las claves y valores sean float
    base_dict = {float(x): float(y) for x, y in dataframe} 
    sorted_dict = dict(sorted(base_dict.items()))
    
    print(f"Diccionario Base Creado (Total de Puntos: {len(sorted_dict)})")
    return sorted_dict

def optimize_dictionary(base_dict):
    """
    Sección III: Optimiza el Diccionario Base calculando la Pendiente (Peso) y 
    la Ordenada al Origen (Sesgo) para cada segmento entre pares de puntos.
    """
    print("\n--- 2. Optimizando Diccionario (Calculando Pesos y Sesgos) ---")
    keys = list(base_dict.keys())
    optimized_dict = {}

    for i in range(len(keys) - 1):
        x1, y1 = keys[i], base_dict[keys[i]]
        x2, y2 = keys[i+1], base_dict[keys[i+1]]
        
        if x2 - x1 == 0:
            pendiente = float('nan')
            ordenada = float('nan')
        else:
            # Fórmula de la Pendiente (Peso)
            pendiente = (y2 - y1) / (x2 - x1)
            # Fórmula de la Ordenada al Origen (Sesgo)
            ordenada = y1 - pendiente * x1
        
        optimized_dict[x1] = (pendiente, ordenada)

    # El último punto: Marca el límite superior, su segmento es indefinido.
    optimized_dict[keys[-1]] = (float('nan'), float('nan'))
    
    print(f"Número de segmentos iniciales: {len(optimized_dict) - 1}")
    return optimized_dict

def compress_lossless(optimized_dict, epsilon=1e-6):
    """
    Sección IV: Compresión sin Pérdida (Invarianza Geométrica).
    Elimina los puntos intermedios donde la Pendiente del segmento adyacente es idéntica.
    """
    print("\n--- 3. Compresión sin Pérdida (Aplicando Invarianza Geométrica) ---")
    keys = list(optimized_dict.keys())
    compressed_dict = {}
    
    if not keys:
        return {}

    # El primer punto siempre se mantiene como punto de partida
    compressed_dict[keys[0]] = optimized_dict[keys[0]]

    for i in range(1, len(keys) - 1):
        x_current = keys[i]
        
        pendiente_current, _ = optimized_dict[x_current]
        pendiente_prev, _ = optimized_dict[keys[i-1]]
        
        # Comprobación de la Invarianza: Si la diferencia de pendientes es mayor que epsilon, mantenemos el punto.
        # Esto reemplaza el uso de np.isclose
        if abs(pendiente_current - pendiente_prev) > epsilon:
            compressed_dict[x_current] = optimized_dict[x_current]
    
    # Agregamos el extremo mayor
    compressed_dict[keys[-1]] = optimized_dict[keys[-1]]
    
    print(f"Segmentos Redundantes Eliminados (Puntos restantes): {len(compressed_dict) - 1}")
    return compressed_dict


def compress_lossy(compressed_dict, base_dict, tolerance=0.03):
    """
    Sección V: Compresión con Pérdida (Criterio Humano).
    Elimina Neuronas/puntos cuya eliminación no produce un error superior a la 
    tolerancia en los puntos originales del Diccionario Base.
    """
    print(f"\n--- 4. Compresión con Pérdida (Tolerancia Máxima: {tolerance:.3f}) ---")
    
    # Transformamos el diccionario a una lista de puntos que representan segmentos activos
    compressed_list = [(x, p, o) for x, (p, o) in compressed_dict.items() if not math.isnan(p)]
    
    final_compressed = []
    
    if not compressed_list:
        # Retorna el diccionario con solo el punto final (NaN, NaN) si estaba vacío
        if compressed_dict:
             return {list(compressed_dict.keys())[-1]: (float('nan'), float('nan'))}
        return {}

    # El primer punto siempre se mantiene como punto de partida
    final_compressed.append(compressed_list[0])

    for i in range(1, len(compressed_list)):
        x_current, p_current, o_current = compressed_list[i]
        
        # Intentamos 'saltar' el punto actual (x_current) y usar la pendiente del punto anterior (x_prev)
        x_prev, p_prev, o_prev = final_compressed[-1] 
        
        # 1. Y_true: El valor original de la salida en X_current (o su predicción si ya fue eliminado antes)
        y_true = base_dict.get(x_current)
        if y_true is None:
             # Si el punto ya fue eliminado en Secc IV, usamos su predicción sin pérdida como Y_true
             y_true = x_current * p_current + o_current 
        
        # 2. Y_hat: La predicción si se utiliza el segmento extendido anterior (p_prev, o_prev)
        y_hat = x_current * p_prev + o_prev
        
        error = abs(y_true - y_hat)
        
        # Criterio: Si el error es mayor que la tolerancia, el punto es Relevante y se mantiene.
        if error > tolerance:
            final_compressed.append(compressed_list[i])

    # Reconstruimos el diccionario con la compresión con pérdida.
    lossy_dict = {x: (p, o) for x, p, o in final_compressed}
    
    # Agregamos el extremo mayor
    if compressed_dict:
        lossy_dict[list(compressed_dict.keys())[-1]] = (float('nan'), float('nan')) 
    
    print(f"Neuronas No Relevantes Eliminadas. Segmentos finales: {len(lossy_dict) - 1}")
    return lossy_dict


# --- FUNCIÓN DE PREDICCIÓN (Secciones III y VII - El Núcleo Operativo Limpio) ---

def predict(x: float, dictionary: dict) -> tuple:
    """
    Realiza una predicción Y para un valor X dado utilizando el modelo MRLS comprimido.
    Aplica generalización (extrapolación) si X está fuera del rango de entrenamiento.
    
    Args:
        x: El valor de entrada para el cual se requiere la predicción.
        dictionary: El diccionario MRLS que contiene los segmentos lineales.
        
    Returns:
        Una tupla (y_predicted, segment_info, description)
    """
    
    # 1. Preparar Claves y Límites
    keys = sorted(dictionary.keys())
    if len(keys) < 2:
        return None, "Error: El diccionario MRLS debe tener al menos dos puntos.", "ERROR"

    min_x = keys[0]             # El límite inferior de entrenamiento
    final_x_limit = keys[-1]    # El límite superior absoluto de entrenamiento
    
    # El penúltimo punto es la clave del último segmento válido (P y O definidos)
    max_segment_key = keys[-2] 

    target_x = None
    description = "INTERPOLACIÓN (Dentro del Rango)"

    # 2. Manejo de Generalización (Extrapolación)
    
    # Extremo Menor: Si X es menor que el límite inferior, se usa el primer segmento.
    if x < min_x:
        target_x = min_x
        description = "EXTRAPOLACIÓN Menor (< X_min)"
    
    # Búsqueda del segmento activo (para interpolación o extrapolación mayor)
    else:
        # Encontrar la clave X_n más próxima y menor o igual a X
        # Se busca la clave de atrás hacia adelante para encontrar el segmento activo (Xn <= X)
        for key in reversed(keys):
            # Solo buscamos claves que marcan el inicio de un segmento válido (no NaN)
            if x >= key and not math.isnan(dictionary.get(key, (None, None))[0]): 
                target_x = key
                break
        
        # Extremo Mayor: Si el segmento encontrado es el último válido (max_segment_key) 
        # y X excede el límite final (final_x_limit), se mantiene ese segmento.
        if target_x == max_segment_key and x >= final_x_limit:
            description = "EXTRAPOLACIÓN Mayor (> X_max)"
        # Si x es exactamente el punto final (keys[-1]), target_x será max_segment_key 
        # (ya que keys[-1] tiene NaN) y se trata como la última interpolación válida.

    # 3. Cálculo de la Predicción
    
    if target_x is None:
        # Esto ocurre si el segmento más bajo es inválido o si hay un error lógico
        return None, "Error: No se pudo encontrar un segmento activo para X.", "ERROR"
        
    # Obtener los parámetros del segmento activo
    P, O = dictionary[target_x]
    
    # Manejar el caso muy improbable de que target_x sea el punto final (NaN) pero no se haya detectado arriba.
    if math.isnan(P) or math.isnan(O):
        P, O = dictionary[max_segment_key]
        target_x = max_segment_key

    # Calcular Y
    y_predicted = x * P + O
    
    # 4. Información del Segmento
    segment_info = f"Segmento: [X={target_x:.2f}] con P={P:.2f} y O={O:+.2f}"
    
    return y_predicted, segment_info, description


# --- EJECUCIÓN DEL MODELO DE JUGUETE ---
# Este bloque demuestra la generación del diccionario final y las pruebas de predicción.

if __name__ == "__main__":
    
    print("--- Demostración COMPLETA del Modelo de Regresión Lineal Segmentada (MRLS) ---")
    
    # I. DataFrame Purificado (Datos de entrenamiento)
    # Los datos se ordenarán internamente: [-8, -4], [-6, -6], [-5, -6.01], [-4, -6], [-2, -4], [0, 0], [2, 4], [3, 7], [4, 10], [6, 18]
    dataframe = [
        [-6.00, -6.00], [2.00, 4.00], [-8.00, -4.00], [0.00, 0.00],
        [4.00, 10.0], [-4.00, -6.00], [6.00, 18.0], [-5.00, -6.01],
        [3.00, 7.00], [-2.00, -4.00]
    ]

    # 1. Creación del diccionario base
    base_dict = create_base_dictionary(dataframe)

    # 2. Optimización (Pesos/Sesgos)
    optimized_dict = optimize_dictionary(base_dict)

    # 3. Compresión sin Pérdida (Invarianza Geométrica)
    compressed_lossless_dict = compress_lossless(optimized_dict)

    # 4. Compresión con Pérdida (Criterio Humano)
    COMPRESSION_TOLERANCE = 0.03
    compressed_lossy_dict = compress_lossy(compressed_lossless_dict, base_dict, tolerance=COMPRESSION_TOLERANCE)
    
    # Mostrar el resultado final de la compresión (Diccionario que usa la web)
    # Se espera que se eliminen los puntos: -5.00, -4.00, 0.00, 3.00
    print_mrls_dictionary(compressed_lossy_dict, "4. Diccionario FINAL Comprimido (MRLS)")


    print("\n--- 5. PRUEBAS DE PREDICCIÓN CON EL MODELO FINAL ---")
    
    test_values = [
        -10.0, # Extrapolación Menor
        -7.0,  # Interpolación Segmento -8.0
        -5.0,  # Interpolación Segmento -6.0 (punto eliminado en la pérdida)
        -3.0,  # Prueba de seguridad
        0.0,   # Interpolación Segmento -2.0 (punto eliminado en la pérdida)
        3.0,   # Interpolación Segmento +2.0 (punto eliminado en la pérdida)
        5.0,   # Interpolación Segmento +4.0
        6.0,   # Límite Máximo
        8.0    # Extrapolación Mayor
    ]

    for x_val in test_values:
        y, info, desc = predict(x_val, compressed_lossy_dict)
        
        if y is not None:
            # Imprimimos el resultado de la predicción
            print(f"X = {x_val:6.2f} | Y pred = {y:7.3f} | {desc:25} | {info}")
        else:
            print(f"X = {x_val:6.2f} | Error al predecir: {info}")
            
    print("\n-------------------------------------------")
