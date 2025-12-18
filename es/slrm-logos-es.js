/**
 * @fileoverview Algoritmo Central SLRM-LOGOS (V5.12)
 * Un Modelo de Regresión Lineal Segmentada (SLRM) implementado en JavaScript/Node.js puro.
 * Este módulo proporciona funciones para el entrenamiento y la predicción utilizando el algoritmo Logos Core.
 * Las funciones son síncronas y están diseñadas para entornos de cómputo (Node.js/Workers).
 * @author Alex Kinetic y Logos
 * Dependencias: Ninguna.
 */

// --- LÓGICA CENTRAL SLRM (V5.12) ---

// CONSTANTE DE PRECISIÓN PARA ARITMÉTICA DE PUNTO FLOTANTE
const TOLERANCIA_FLOTANTE = 1e-9;

// --- PROCESAMIENTO DE DATOS ---

/**
 * Limpia y ordena los datos, manejando los duplicados de X promediando Y.
 * @param {string} dataString - Los datos de entrada como una cadena (ej., "x1, y1\nx2, y2").
 * @returns {Array<Object>} - Un array de puntos {x: number, y: number} limpios y ordenados.
 */
function _limpiarYOrdenarDatos(dataString) {
    const mapaPuntos = new Map();

    dataString.split('\n').forEach(linea => {
        // Soporta varios separadores (espacio, coma)
        const partes = linea.trim().split(/[\s,]+/).map(p => parseFloat(p.trim()));
        if (partes.length >= 2 && !isNaN(partes[0]) && !isNaN(partes[1])) {
            const x = partes[0];
            const y = partes[1];

            if (mapaPuntos.has(x)) {
                // Manejar valores X duplicados promediando Y
                const actual = mapaPuntos.get(x);
                actual.y = (actual.y * actual.count + y) / (actual.count + 1);
                actual.count += 1;
            } else {
                mapaPuntos.set(x, { x: x, y: y, count: 1 });
            }
        }
    });

    const datosLimpios = Array.from(mapaPuntos.values()).map(p => ({x: p.x, y: p.y}));
    // Ordenar por valor X
    datosLimpios.sort((a, b) => a.x - b.x);

    return datosLimpios;
}

// --- LÓGICA DE COMPRESIÓN (Sin Pérdida: Invarianza Geométrica) ---

/**
 * Realiza compresión sin pérdida identificando puntos no colineales
 * (cambios de pendiente). Estos forman los "Breakpoints Base".
 * @param {Array<Object>} datos - Los datos limpios y ordenados.
 * @returns {Array<number>} - Un array de valores X para los breakpoints críticos.
 */
function _compresionSinPerdida(datos) {
    if (datos.length < 2) return datos.map(d => d.x);

    const xCriticos = [datos[0].x];

    for (let i = 1; i < datos.length - 1; i++) {
        const p0 = datos[i - 1];
        const p1 = datos[i];
        const p2 = datos[i + 1];

        const dX_a = p1.x - p0.x;
        const dX_b = p2.x - p1.x;

        // Manejar segmentos verticales o dX insignificante
        if (Math.abs(dX_a) < TOLERANCIA_FLOTANTE || Math.abs(dX_b) < TOLERANCIA_FLOTANTE) {
            xCriticos.push(p1.x);
            continue;
        }

        const P_a = (p1.y - p0.y) / dX_a;
        const P_b = (p2.y - p1.y) / dX_b;

        // Verificación de colinealidad: si las pendientes son diferentes más allá de la tolerancia, es un punto crítico.
        if (Math.abs(P_a - P_b) > TOLERANCIA_FLOTANTE) {
            xCriticos.push(p1.x);
        }
    }

    if (datos.length > 1) {
        xCriticos.push(datos[datos.length - 1].x);
    }

    // Asegurar unicidad (aunque la lógica debe minimizar duplicados)
    return Array.from(new Set(xCriticos));
}

// --- LÓGICA DE COMPRESIÓN (Con Pérdida / SLRM) ---

/**
 * Calcula el error máximo de un segmento comprometido contra todos
 * los puntos de datos intermedios reales.
 */
function _calcularErrorMaximoSegmento(xS, xE, P, O, clavesDatos, datos) {
    if (isNaN(P)) return 0.0;

    const indiceInicio = clavesDatos.indexOf(xS);
    const indiceFin = clavesDatos.indexOf(xE);
    let maxError = 0.0;

    // Iterar SOLAMENTE sobre puntos estrictamente intermedios
    for (let k = indiceInicio + 1; k < indiceFin; k++) {
        const xMedio = datos[k].x;
        const yVerdaderoMedio = datos[k].y;

        const yEstimadoMedio = P * xMedio + O;
        const error = Math.abs(yVerdaderoMedio - yEstimadoMedio);

        maxError = Math.max(maxError, error);
    }
    return maxError;
}


/**
 * Aplica la compresión con pérdida utilizando el algoritmo SLRM (MRLS).
 * @param {Array<number>} clavesIniciales - Valores X de la compresión sin pérdida (breakpoints).
 * @param {number} epsilon - La tolerancia de error máxima permitida.
 * @param {Array<Object>} datos - Los datos originales limpios y ordenados.
 * @returns {Object} - { modeloFinal: Map<number, Array<number>>, maxError: number }
 * modeloFinal: Mapa donde la clave es X_inicio, el valor es [Pendiente P, Intercepto O, X_fin].
 */
function _compresionConPerdida(clavesIniciales, epsilon, datos) {
    const clavesDatos = datos.map(d => d.x);
    const mapaDatos = new Map(datos.map(p => [p.x, p.y]));

    if (clavesIniciales.length < 2) return { modeloFinal: new Map(), maxError: 0 };

    const modeloFinal = new Map();
    let i = 0; // Índice del punto inicial del segmento (x_inicio) en clavesIniciales
    let maxErrorGeneral = 0;
    // Usar 1e-12 para epsilon=0 para evitar división por cero en verificaciones de error
    const epsilonSeguro = Math.max(epsilon, 1e-12);

    while (i < clavesIniciales.length - 1) {

        const x_inicio = clavesIniciales[i];
        const y_inicio = mapaDatos.get(x_inicio);

        let j = i + 1; // Índice del punto final candidato (x_fin_candidato) en clavesIniciales

        while (j < clavesIniciales.length) {

            const x_fin_candidato = clavesIniciales[j];
            const y_fin_candidato = mapaDatos.get(x_fin_candidato);

            const dX = x_fin_candidato - x_inicio;

            if (isNaN(dX) || Math.abs(dX) < TOLERANCIA_FLOTANTE) {
                j++;
                continue;
            }

            // Calcular el segmento candidato: Pendiente (P) e Intercepto (O)
            const P_prueba = (y_fin_candidato - y_inicio) / dX;
            const O_prueba = y_inicio - P_prueba * x_inicio;

            let errorExcedido = false;

            const indiceInicio = clavesDatos.indexOf(x_inicio);
            const indiceFin = clavesDatos.indexOf(x_fin_candidato);

            // Verificar todos los puntos *intermedios* contra Epsilon
            for (let k = indiceInicio + 1; k < indiceFin; k++) {
                const x_medio = datos[k].x;
                const y_verdadero_medio = datos[k].y;

                const y_estimado_medio = P_prueba * x_medio + O_prueba;
                const error = Math.abs(y_verdadero_medio - y_estimado_medio);

                if (error > epsilonSeguro) {
                    errorExcedido = true;
                    break;
                }
            }

            if (errorExcedido) {
                // El segmento falló. Comprometer el segmento anterior (j-1).
                const x_fin_comprometido = clavesIniciales[j - 1];
                const y_fin_comprometido = mapaDatos.get(x_fin_comprometido);

                const dX_comprometido = x_fin_comprometido - x_inicio;
                // Evitar división por cero para P si dX está cerca de cero
                const P = dX_comprometido === 0 ? 0 : (y_fin_comprometido - y_inicio) / dX_comprometido;
                const O = y_inicio - P * x_inicio;

                // Guardar segmento: [Pendiente P, Intercepto O, X_Fin]
                modeloFinal.set(x_inicio, [P, O, x_fin_comprometido]);

                const maxErrorComprometido = _calcularErrorMaximoSegmento(x_inicio, x_fin_comprometido, P, O, clavesDatos, datos);
                maxErrorGeneral = Math.max(maxErrorGeneral, maxErrorComprometido);

                i = j - 1; // El siguiente segmento comienza donde terminó el anterior.
                break;

            } else if (j === clavesIniciales.length - 1) {
                // Se alcanzó el último punto y pasó la prueba. Comprometer el segmento final.
                const x_fin = clavesIniciales[j];
                const y_fin = mapaDatos.get(x_fin);

                const dX = x_fin - x_inicio;
                const P = dX === 0 ? 0 : (y_fin - y_inicio) / dX;
                const O = y_inicio - P * x_inicio;

                modeloFinal.set(x_inicio, [P, O, x_fin]);

                const maxErrorFinal = _calcularErrorMaximoSegmento(x_inicio, x_fin, P, O, clavesDatos, datos);
                maxErrorGeneral = Math.max(maxErrorGeneral, maxErrorFinal);

                i = j; // Finaliza el bucle exterior
                break;
            }

            j++; // Intentar extender el segmento aún más
        }
    }

    // Marcador final para el último punto
    const ultimaClave = clavesIniciales[clavesIniciales.length - 1];
    if (ultimaClave !== undefined && !modeloFinal.has(ultimaClave)) {
         // Usar [NaN, NaN, NaN] para puntos finales que no son inicios de segmento
         modeloFinal.set(ultimaClave, [NaN, NaN, NaN]);
    }

    return { modeloFinal, maxError: maxErrorGeneral };
}


// --- ENTRENAMIENTO Y PREDICCIÓN PRINCIPAL ---

/**
 * Función principal para entrenar el modelo SLRM.
 * @param {string} dataString - La cadena de datos crudos (ej. "1.0, 5.0\n2.0, 10.0").
 * @param {number} epsilon - La tolerancia de error máxima (debe ser >= 0).
 * @returns {Object} - { model: Map<number, Array<number>>, originalData: Array<Object>, maxError: number }
 */
function train_slrm(dataString, epsilon) {
    if (isNaN(epsilon) || epsilon < 0) {
        epsilon = 0;
    }

    const datosOriginales = _limpiarYOrdenarDatos(dataString);
    if (datosOriginales.length < 2) return { model: new Map(), originalData: datosOriginales, maxError: 0 };

    const breakpointsIniciales = _compresionSinPerdida(datosOriginales);

    // La Compresión Con Pérdida es síncrona
    const resultado = _compresionConPerdida(breakpointsIniciales, epsilon, datosOriginales);

    return { model: resultado.modeloFinal, originalData: datosOriginales, maxError: resultado.maxError };
}


/**
 * Realiza una predicción Y para un valor X dado utilizando el modelo entrenado.
 * NOTA: Esta función requiere los 'datosOriginales' (o min/max X) para manejar correctamente
 * la extrapolación fuera del rango del modelo.
 * @param {number} x_in - El valor X de entrada para la predicción.
 * @param {Map<number, Array<number>>} model - El Mapa del modelo entrenado.
 * @param {Array<Object>} originalData - Los puntos de datos originales limpios y ordenados.
 * @returns {Object} - { x_in: number, y_pred: number, slope_P: number, intercept_O: number }
 */
function predict_slrm(x_in, model, originalData) {
    // 1. Preparar claves y límites
    const claves = Array.from(model.entries())
        // Filtrar solo las entradas que son inicios de segmento válidos (tienen Pendiente P)
        .filter(([x_inicio, [P]]) => !isNaN(P))
        .map(([x_inicio]) => x_inicio);
    claves.sort((a, b) => a - b);

    if (claves.length === 0 || originalData.length === 0) {
        return { x_in, y_pred: NaN, slope_P: NaN, intercept_O: NaN };
    }

    const dataMinX = originalData[0].x;
    const dataMaxX = originalData[originalData.length - 1].x;

    let claveActiva = null;

    // 2. Lógica de Extrapolación/Interpolación
    if (x_in < dataMinX) {
        // Extrapolación antes del rango de datos: usar el primer segmento
        claveActiva = claves[0];
    } else if (x_in >= dataMaxX) {
        // Extrapolación después del rango de datos: usar el último segmento
        claveActiva = claves[claves.length - 1];
    } else {
        // Interpolación: encontrar el segmento donde cae x_in
        for (let i = 0; i < claves.length; i++) {
            const x_inicio = claves[i];
            const x_fin = model.get(x_inicio)[2]; // x_end es el 3er elemento en el array del segmento

            // Verificar si x_in está en [x_inicio, x_fin)
            if (x_in >= x_inicio && x_in < x_fin) {
                claveActiva = x_inicio;
                break;
            }
        }
        // Retorno para el último segmento si x_in es exactamente el último breakpoint (x_max)
        if (claveActiva === null) {
            claveActiva = claves[claves.length - 1];
        }
    }

    if (claveActiva === undefined) {
        return { x_in, y_pred: NaN, slope_P: NaN, intercept_O: NaN };
    }

    // 3. Cálculo
    // Formato del segmento: [P, O, x_end]
    const [P, O] = model.get(claveActiva) || [NaN, NaN];

    let y_pred = NaN;
    if (!isNaN(P) && !isNaN(O)) {
        // Fórmula de la regresión lineal: Y = P * X + O
        y_pred = x_in * P + O;
    }

    return {
        x_in,
        y_pred,
        slope_P: P,
        intercept_O: O
    };
}

// Exportar las funciones principales para su uso como un módulo NPM
module.exports = {
    train_slrm,
    predict_slrm,
    TOLERANCIA_FLOTANTE
};
