
import numpy as np

# NOTA: En este proyecto, los datos se representan como un diccionario
# donde la clave es la Entrada (X) y el valor asociado es una tupla
# (Pendiente, Ordenada). La Rueda Mágica transforma el DataFrame en una
# función lineal a trozos.

def create_base_dictionary(dataframe):
    """
    Sección I & II: Crea el Diccionario Base (Entrada: X, Salida: Y) ordenando 
    los datos de menor a mayor por la Entrada (X). Este paso reemplaza el 
    proceso iterativo de 'entrenamiento' por una simple ordenación.
    """
    print("--- 1. Creando Diccionario Base (Entrenamiento Instantáneo) ---")
    base_dict = {x: y for x, y in dataframe}
    sorted_dict = dict(sorted(base_dict.items()))
    
    print(f"Diccionario Base Creado (Entradas: {list(sorted_dict.keys())})")
    return sorted_dict

def optimize_dictionary(base_dict):
    """
    Sección III: Optimiza el Diccionario Base calculando la Pendiente (Peso) y 
    la Ordenada al Origen (Sesgo) para cada segmento entre pares de puntos adyacentes.
    Cada clave (Xn) almacena el Peso y Sesgo del segmento que comienza en Xn.
    """
    print("\n--- 2. Optimizando Diccionario (Calculando Pesos y Sesgos) ---")
    keys = list(base_dict.keys())
    optimized_dict = {}

    for i in range(len(keys) - 1):
        x1, y1 = keys[i], base_dict[keys[i]]
        x2, y2 = keys[i+1], base_dict[keys[i+1]]
        
        if x2 - x1 == 0:
            pendiente = np.nan
            ordenada = np.nan
        else:
            # Fórmula de la Pendiente (Peso)
            pendiente = (y2 - y1) / (x2 - x1)
            # Fórmula de la Ordenada al Origen (Sesgo)
            ordenada = y1 - pendiente * x1
        
        optimized_dict[x1] = (pendiente, ordenada)

    # El último punto: Marca el límite superior, su segmento es indefinido por ahora.
    optimized_dict[keys[-1]] = (np.nan, np.nan)
    
    print(f"Número de segmentos iniciales: {len(optimized_dict) - 1}")
    return optimized_dict

def compress_lossless(optimized_dict):
    """
    Sección IV: Compresión sin Pérdida de Información (Invarianza Geométrica).
    Elimina los puntos intermedios donde la Pendiente del segmento adyacente es idéntica.
    """
    print("\n--- 3. Compresión sin Pérdida (Aplicando Invarianza Geométrica) ---")
    keys = list(optimized_dict.keys())
    compressed_dict = {}
    
    if keys:
        compressed_dict[keys[0]] = optimized_dict[keys[0]]

    for i in range(1, len(keys) - 1):
        x_current = keys[i]
        
        pendiente_current, _ = optimized_dict[x_current]
        pendiente_prev, _ = optimized_dict[keys[i-1]]
        
        # Comprobación de la Invarianza: Si las pendientes son diferentes, mantenemos el punto.
        if not np.isclose(pendiente_current, pendiente_prev):
             compressed_dict[x_current] = optimized_dict[x_current]
    
    # Agregamos el extremo mayor
    compressed_dict[keys[-1]] = optimized_dict[keys[-1]]
    
    print(f"Segmentos Redundantes Eliminados (Puntos restantes): {len(compressed_dict) - 1}")
    return compressed_dict


def compress_lossy(compressed_dict, base_dict, tolerance=0.03):
    """
    Sección V: Compresión con Pérdida de Información No Relevante (Criterio Humano).
    Elimina Neuronas/puntos cuya eliminación no produce un cambio superior a 
    la tolerancia (error máximo) en los puntos originales del Diccionario Base.
    """
    print(f"\n--- 4. Compresión con Pérdida (Tolerancia: {tolerance}) ---")
    
    # Transformamos el diccionario a una lista de puntos para facilitar la manipulación de índices
    compressed_list = [(x, p, o) for x, (p, o) in compressed_dict.items() if not np.isnan(p)]
    
    final_compressed = []
    
    # El primer punto siempre se mantiene como punto de partida
    if compressed_list:
        final_compressed.append(compressed_list[0])

    for i in range(1, len(compressed_list)):
        x_current, p_current, o_current = compressed_list[i]
        
        # Intentamos 'saltar' el punto actual (x_current) y usar la pendiente del punto anterior (x_prev)
        x_prev, p_prev, o_prev = final_compressed[-1] 
        
        # 1. Y_true: El valor original de la salida en X_current
        y_true = base_dict.get(x_current)
        if y_true is None:
             # Si el punto ya fue eliminado en Secc IV, usamos su predicción sin pérdida como Y_true
             y_true = x_current * p_current + o_current 
        
        # 2. Y_hat: La predicción si se utiliza el segmento extendido anterior
        y_hat = x_current * p_prev + o_prev
        
        error = abs(y_true - y_hat)
        
        # Criterio: Si el error es mayor que la tolerancia, el punto es Relevante y se mantiene.
        if error > tolerance:
            final_compressed.append(compressed_list[i])

    # Reconstruimos el diccionario con la compresión con pérdida.
    lossy_dict = {x: (p, o) for x, p, o in final_compressed}
    # Agregamos el extremo mayor
    lossy_dict[list(compressed_dict.keys())[-1]] = (np.nan, np.nan) 
    
    print(f"Neuronas No Relevantes Eliminadas. Segmentos finales: {len(lossy_dict) - 1}")
    return lossy_dict


def predict(x, dictionary):
    """
    Sección III (Uso) y VII (Generalización): Realiza la predicción.
    Busca la clave (Xn) más próxima y aplica la Ecuación Maestra (Y = X * P + O).
    Aplica Generalización (Extrapolación) fuera de los extremos.
    """
    keys = list(dictionary.keys())
    
    # --- Generalización: Extremo Menor (Sección VII) ---
    if x < keys[0]:
        print(f"PREDICCIÓN EXTRAPOLADA: Extremo Menor (Usando segmento que comienza en {keys[0]})")
        pendiente, ordenada = dictionary[keys[0]]
        return x * pendiente + ordenada

    # 1. Búsqueda del segmento activo (Xn más próximo menor o igual)
    target_x = None
    for key in reversed(keys):
        if x >= key:
            target_x = key
            break
            
    # --- Generalización: Extremo Mayor (Sección VII) ---
    if target_x == keys[-1]:
        # Si x cae en el último punto o más allá, usamos el segmento que comienza en el penúltimo punto
        print(f"PREDICCIÓN EXTRAPOLADA: Extremo Mayor (Usando segmento que comienza en {keys[-2]})")
        # El penúltimo punto es keys[-2], que contiene el último segmento válido
        pendiente, ordenada = dictionary[keys[-2]] 
        return x * pendiente + ordenada

    # 2. Predicción dentro del rango conocido
    if target_x is not None:
        pendiente, ordenada = dictionary[target_x]
        y_predicted = x * pendiente + ordenada
        return y_predicted
    
    return "ERROR: No se encontró segmento válido."

# --- EJECUCIÓN DEL MODELO DE JUGUETE ---

# I. DataFrame Purificado 
dataframe = [
    [-6.00, -6.00], [2.00, 4.00], [-8.00, -4.00], [0.00, 0.00],
    [4.00, 10.0], [-4.00, -6.00], [6.00, 18.0], [-5.00, -6.01],
    [3.00, 7.00], [-2.00, -4.00]
]

# 1. Creación
base_dict = create_base_dictionary(dataframe)

# 2. Optimización (Pesos/Sesgos)
optimized_dict = optimize_dictionary(base_dict)

# 3. Compresión sin Pérdida (Sección IV)
compressed_lossless_dict = compress_lossless(optimized_dict)

# 4. Compresión con Pérdida (Sección V)
# Se espera que el punto -5.00 sea eliminado (error 0.01 < 0.03)
compressed_lossy_dict = compress_lossy(compressed_lossless_dict, base_dict, tolerance=0.03)


print("\n--- 5. PRUEBAS DE GENERALIZACIÓN (SECCIÓN VII) ---")
print(f"Rango de entrenamiento (X): [{list(base_dict.keys())[0]} a {list(base_dict.keys())[-1]}]")

# A. Extrapolación Extremo Menor
x_test_menor = -10.0
y_pred_menor = predict(x_test_menor, compressed_lossy_dict)
print(f"\nExtremo Menor X={x_test_menor}: Y = {y_pred_menor:.2f}")

# B. Extrapolación Extremo Mayor
x_test_mayor = 8.0
y_pred_mayor = predict(x_test_mayor, compressed_lossy_dict)
print(f"Extremo Mayor X={x_test_mayor}: Y = {y_pred_mayor:.2f}")

# C. Predicción dentro del rango (X=-5.00, eliminado en compresión con pérdida)
x_test_mid = -5.00
y_pred_mid = predict(x_test_mid, compressed_lossy_dict)
print(f"Dentro de Rango X={x_test_mid}: Y = {y_pred_mid:.2f} (Error: {abs(y_pred_mid - -6.01):.2f})")
