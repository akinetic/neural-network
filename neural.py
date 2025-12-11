import numpy as np

# NOTA: En este proyecto, los datos se representan como un diccionario
# donde la clave es la Entrada (X) y el valor asociado es una tupla
# (Pendiente, Ordenada) o, en el caso del Diccionario Base, el valor de Salida (Y).

def create_base_dictionary(dataframe):
    """
    Sección I & II: Crea el Diccionario Base ordenando los datos de menor a mayor
    por la Entrada (X). Este paso reemplaza el proceso iterativo de 'entrenamiento'.
    """
    print("--- 1. Creando Diccionario Base (Entrenamiento Instantáneo) ---")
    # Convertimos la lista de pares [X, Y] a un diccionario {X: Y} para facilitar el ordenamiento y búsqueda.
    base_dict = {x: y for x, y in dataframe}
    
    # Ordenamos el diccionario por la clave (X)
    sorted_dict = dict(sorted(base_dict.items()))
    
    print(f"Diccionario Base Creado (Entradas: {list(sorted_dict.keys())})")
    return sorted_dict

def optimize_dictionary(base_dict):
    """
    Sección III: Optimiza el Diccionario Base calculando la Pendiente (Peso) y 
    la Ordenada al Origen (Sesgo) para cada segmento definido por pares de puntos
    adyacentes.
    """
    print("\n--- 2. Optimizando Diccionario (Calculando Pesos y Sesgos) ---")
    keys = list(base_dict.keys())
    optimized_dict = {}

    # El bucle recorre todos los puntos excepto el último, ya que definen el segmento
    # entre ese punto y el siguiente.
    for i in range(len(keys) - 1):
        x1, y1 = keys[i], base_dict[keys[i]]
        x2, y2 = keys[i+1], base_dict[keys[i+1]]
        
        # Rigor: Evitar división por cero si X2 == X1 (asumimos Dataframe Purificado)
        if x2 - x1 == 0:
            pendiente = np.nan
            ordenada = np.nan
        else:
            # Fórmula de la Pendiente (Peso)
            pendiente = (y2 - y1) / (x2 - x1)
            # Fórmula de la Ordenada al Origen (Sesgo)
            # y = m*x + b => b = y - m*x
            ordenada = y1 - pendiente * x1
        
        # El punto keys[i] ahora almacena la información del SEGMENTO que COMIENZA en X1.
        optimized_dict[x1] = (pendiente, ordenada)

    # El último punto define el segmento "Más allá del Extremo Mayor" (Pendiente/Ordenada desconocida/futura)
    optimized_dict[keys[-1]] = (np.nan, np.nan)
    
    print(f"Número de segmentos iniciales: {len(optimized_dict) - 1}")
    return optimized_dict

def compress_lossless(optimized_dict):
    """
    Sección IV: Compresión sin Pérdida. Elimina Neuronas/Datos redundantes.
    Una Neurona es redundante si su pendiente es igual a la pendiente de la Neurona anterior.
    """
    print("\n--- 3. Compresión sin Pérdida (Aplicando Invarianza) ---")
    keys = list(optimized_dict.keys())
    compressed_dict = {}
    
    # Inicializamos con el primer punto (que siempre es necesario para definir el primer segmento)
    if keys:
        x_start = keys[0]
        compressed_dict[x_start] = optimized_dict[x_start]

    # Iteramos a partir del segundo punto
    for i in range(1, len(keys) - 1): # Excluimos el último NaN
        x_prev = keys[i-1]
        x_current = keys[i]
        
        # Obtenemos la pendiente del segmento actual (comienza en x_current)
        pendiente_current, _ = optimized_dict[x_current]
        
        # Obtenemos la pendiente del segmento anterior (comienza en el punto anterior, x_prev)
        pendiente_prev, _ = optimized_dict[x_prev]
        
        # El principio de Invarianza: Si las pendientes son iguales, el punto intermedio (x_current)
        # es redundante porque el segmento ya está cubierto por el segmento anterior.
        if not np.isclose(pendiente_current, pendiente_prev):
             compressed_dict[x_current] = optimized_dict[x_current]
    
    # Agregamos el último punto para mantener la consistencia (el extremo mayor)
    compressed_dict[keys[-1]] = optimized_dict[keys[-1]]
    
    print(f"Segmentos Redundantes Eliminados. Segmentos restantes: {len(compressed_dict) - 1}")
    return compressed_dict

def predict(x, compressed_dict):
    """
    Sección III (Uso): Predicción extremadamente rápida basada en el Diccionario Optimizado.
    La predicción es la búsqueda binaria de la clave (Xn) más próxima y la aplicación de la Pendiente/Ordenada asociada.
    """
    keys = list(compressed_dict.keys())
    
    # 1. Búsqueda de la Neurona/Segmento (Xn más próximo menor o igual)
    target_x = None
    for key in reversed(keys):
        if x >= key:
            target_x = key
            break
    
    # 2. Aplicación de la Ecuación Maestra (Y = X * Pendiente + Ordenada)
    if target_x is not None and target_x != keys[-1]:
        pendiente, ordenada = compressed_dict[target_x]
        
        # Si la Pendiente o la Ordenada son NaN (por ejemplo, en el extremo mayor),
        # se devuelve un error o se maneja el caso de Extremos (Sección VII).
        if np.isnan(pendiente):
            return "ERROR: Extremo no definido"
            
        y_predicted = x * pendiente + ordenada
        return y_predicted
    
    elif x < keys[0]:
        return "ERROR: Más allá del Extremo Menor"
    else:
        return "ERROR: Extremo Mayor no definido"

# --- EJECUCIÓN DEL MODELO DE JUGUETE ---

# I. DataFrame Purificado (Sección I)
# [Entrada (X), Salida (Y)]
dataframe = [
    [-6.00, -6.00], [2.00, 4.00], [-8.00, -4.00], [0.00, 0.00],
    [4.00, 10.0], [-4.00, -6.00], [6.00, 18.0], [-5.00, -6.01],
    [3.00, 7.00], [-2.00, -4.00]
]

# 1. Creación
base_dict = create_base_dictionary(dataframe)

# 2. Optimización (Pesos/Sesgos)
optimized_dict = optimize_dictionary(base_dict)

# 3. Compresión sin Pérdida (Eliminando redundancias)
compressed_dict = compress_lossless(optimized_dict)

print("\n--- 4. Demostración de Predicción (Sección III) ---")

# Ejemplo del README: X = 5
x_ejemplo = 5
y_predicha = predict(x_ejemplo, compressed_dict)
print(f"Entrada X = {x_ejemplo}")
print(f"Salida Y  = {y_predicha}")
# La salida es Y = 14.0, coincidiendo con tu cálculo: Y = 5 * 4 - 6 = 14.0
print(f"Cálculo: Y = X * P + O => 5 * 4.0 + (-6.0) = {y_predicha}")

# Ejemplo: X = -5.01 (debería caer en el segmento de [-6.00, -5.00])
x_ejemplo_2 = -5.5
y_predicha_2 = predict(x_ejemplo_2, compressed_dict)
print(f"\nEntrada X = {x_ejemplo_2}")
print(f"Salida Y  = {y_predicha_2:.3f}")
# El segmento [-6.00, -4.00] tiene pendiente 1.0 y ordenada -2.0.
# Sin compresión: El segmento [-6.00, -5.00] tiene pendiente 0.01 y ordenada -5.96.
# Con tu compresión (si asumimos la pérdida de la Sección V no aplicada aún),
# el segmento que inicia en -6.00 se encarga de la predicción.
# En la versión SIN pérdida (Sección IV), el segmento es [-5.00, -4.00] (P: 0.01, O: -5.96)
# El algoritmo busca Xn más proximo menor o igual: es -6.00.
# Pendiente del segmento que comienza en -6.00 es -0.01 y Ordenada es -6.06.
# Y = -5.5 * (-0.01) + (-6.06) = 0.055 - 6.06 = -6.005
print(f"Segmento Activo: {predict(-6.00, optimized_dict)}")
print(f"Predicción esperada: -6.005") # ¡El código funciona!