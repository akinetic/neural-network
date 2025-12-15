/**
 * SLRM-LOGOS (V5.10b) - Módulo JavaScript para Regresión Lineal Segmentada.
 * Implementación del algoritmo determinista Logos Core para Compresión de Conocimiento.
 * * Autores: Alex Kinetic and Logos
 */

// --- FUNCIONES AUXILIARES DE COMPRESIÓN ---

/**
 * Limpia y ordena los datos de entrada, manejando duplicados de X promediando sus valores Y.
 * @param {Array<Array<number>>} data - Array de puntos [[x, y], ...].
 * @returns {Array<{x: number, y: number}>} - Array de puntos limpios y ordenados.
 */
function _cleanAndSortData(data) {
    const map = new Map();
    
    // 1. Agrupar Y por X para manejar duplicados (promediado)
    data.forEach(([x, y]) => {
        if (!map.has(x)) {
            map.set(x, { sum: 0, count: 0 });
        }
        const entry = map.get(x);
        entry.sum += y;
        entry.count++;
    });

    const cleanPoints = Array.from(map.keys()).map(x => ({
        x,
        y: map.get(x).sum / map.get(x).count
    }));

    // 2. Ordenar por X
    cleanPoints.sort((a, b) => a.x - b.x);

    return cleanPoints;
}

/**
 * Realiza compresión sin pérdida: elimina puntos colineales (Invarianza Geométrica).
 * @param {Array<{x: number, y: number}>} points - Puntos limpios y ordenados.
 * @returns {Array<{x: number, y: number}>} - Puntos clave (breakpoints) sin colinealidad.
 */
function _losslessCompression(points) {
    if (points.length <= 2) return points;

    const breakpoints = [points[0]];
    let startPoint = points[0];
    const EPS = 1e-9; // Pequeña tolerancia para la comparación de flotantes

    for (let i = 1; i < points.length - 1; i++) {
        const midPoint = points[i];
        const endPoint = points[i + 1];

        // Calcular pendientes
        const deltaX1 = midPoint.x - startPoint.x;
        const deltaY1 = midPoint.y - startPoint.y;
        const deltaX2 = endPoint.x - midPoint.x;
        const deltaY2 = endPoint.y - midPoint.y;

        let slope1, slope2;

        // Manejo de pendientes infinitas (verticales)
        if (Math.abs(deltaX1) < EPS) slope1 = Infinity;
        else slope1 = deltaY1 / deltaX1;
        
        if (Math.abs(deltaX2) < EPS) slope2 = Infinity;
        else slope2 = deltaY2 / deltaX2;
        
        // Si las pendientes son diferentes, hay un quiebre.
        if (Math.abs(slope1 - slope2) > EPS || (slope1 === Infinity && slope2 !== Infinity) || (slope1 !== Infinity && slope2 === Infinity)) {
            breakpoints.push(midPoint);
            startPoint = midPoint;
        }
    }
    // Siempre añadir el último punto
    breakpoints.push(points[points.length - 1]);
    return breakpoints;
}

/**
 * Realiza compresión con pérdida (MRLS): Segmentos de Línea Mínimos Requeridos.
 * Extiende segmentos hasta que el error de interpolación exceda epsilon.
 * @param {Array<{x: number, y: number}>} breakpoints - Puntos clave de la compresión sin pérdida.
 * @param {number} epsilon - Tolerancia de error máxima.
 * @returns {{model: Object, maxError: number}} - Modelo final y error máximo alcanzado.
 */
function _lossyCompression(breakpoints, epsilon) {
    if (breakpoints.length <= 1) {
        return { model: {}, maxError: 0 };
    }

    const finalModel = {};
    let currentMaxError = 0;
    let i = 0; // Índice del punto inicial del segmento

    while (i < breakpoints.length - 1) {
        let start = breakpoints[i];
        let j = i + 1; // Índice del punto final potencial del segmento

        // Extender el segmento lo más posible (algoritmo MRLS)
        while (j < breakpoints.length) {
            let end = breakpoints[j];
            let segmentMaxError = 0;
            let P, O;

            // 1. Calcular la línea (Pendiente P y Intercepto O)
            if (end.x === start.x) { 
                P = Infinity;
                O = NaN;
            } else {
                P = (end.y - start.y) / (end.x - start.x);
                O = start.y - P * start.x;
            }

            // 2. Verificar el error de todos los puntos intermedios [i+1, j]
            let isWithinTolerance = true;
            for (let k = i + 1; k <= j; k++) {
                let midPoint = breakpoints[k];
                let y_pred = P * midPoint.x + O;
                let error = Math.abs(midPoint.y - y_pred);

                if (error > epsilon) {
                    isWithinTolerance = false;
                    break;
                }
                segmentMaxError = Math.max(segmentMaxError, error);
            }

            if (isWithinTolerance) {
                // El segmento se puede extender hasta el punto j.
                currentMaxError = Math.max(currentMaxError, segmentMaxError);
                j++; // Intenta incluir el siguiente punto
            } else {
                // El punto j excedió el error. El segmento finaliza en j-1.
                j--; 
                break;
            }
        }
        
        // El segmento final válido es [i, j]. Si j se salió del array, se ajusta a breakpoints.length - 1.
        if (j === breakpoints.length) {
             j = breakpoints.length - 1;
        }

        // Si el segmento no pudo avanzar (i == j, ocurre si j=i+1 falla), avanzar un punto a la vez.
        if (i === j) {
            j = i + 1;
        }
        
        let end = breakpoints[j];
        
        // Recalcular los parámetros finales para el segmento [i, j] usando sus puntos de inicio y fin
        let P, O;
        if (end.x === start.x) {
            P = Infinity;
            O = NaN;
        } else {
            P = (end.y - start.y) / (end.x - start.x);
            O = start.y - P * start.x;
        }

        // Almacenar el segmento en el modelo: {X_inicio: {P, O, X_end}}
        finalModel[start.x] = { P, O, X_end: end.x };

        // El nuevo punto inicial es el punto final del segmento actual
        i = j;
    }

    return { model: finalModel, maxError: currentMaxError };
}


// --- FUNCIONES PRINCIPALES DEL MODELO ---

/**
 * Entrena el Modelo de Regresión Lineal Segmentada (SLRM).
 * Aplica purificación, compresión sin pérdida y compresión con pérdida (MRLS).
 * @param {Array<Array<number>>} data - Datos de entrada [[x, y], ...].
 * @param {number} [epsilon=0.05] - Tolerancia de error máxima deseada.
 * @returns {{finalModel: Object, originalPoints: Array<{x: number, y: number}>, maxErrorAchieved: number}} - Resultados del entrenamiento.
 */
export function train_slrm(data, epsilon = 0.05) {
    // 1. Purificación y Ordenamiento
    const cleanPoints = _cleanAndSortData(data);
    if (cleanPoints.length < 2) {
        console.error("SLRM-LOGOS: Se requieren al menos dos puntos únicos para el entrenamiento.");
        return { finalModel: {}, originalPoints: cleanPoints, maxErrorAchieved: 0 };
    }

    // 2. Compresión Sin Pérdida (Invarianza Geométrica)
    const breakpoints = _losslessCompression(cleanPoints);

    // 3. Compresión Con Pérdida (MRLS)
    const { model, maxError } = _lossyCompression(breakpoints, epsilon);

    return {
        finalModel: model,
        originalPoints: cleanPoints,
        maxErrorAchieved: maxError
    };
}

/**
 * Realiza una predicción para un valor X usando el modelo SLRM.
 * Nota: La implementación completa del caché LRU se omite por simplicidad en este módulo.
 * @param {number} x_test - El valor X para el cual se desea la predicción.
 * @param {Object} finalModel - El modelo SLRM (objeto {X_inicio: {P, O, X_end}}).
 * @returns {{x_in: number, y_pred: number|null, slope_P: number|null, intercept_O: number|null, cache_hit: boolean}} - Resultado de la predicción.
 */
export function predict_slrm(x_test, finalModel) {
    let slope_P = null;
    let intercept_O = null;
    let y_pred = null;
    const cache_hit = false; // Bandera de caché siempre false en esta implementación simple.

    // Encontrar el segmento activo
    const xStarts = Object.keys(finalModel).map(Number).sort((a, b) => a - b);
    
    // Buscar el segmento
    for (const xStart of xStarts) {
        const segment = finalModel[xStart];
        
        // Verifica si x_test está dentro del rango [X_start, X_end]
        if (x_test >= xStart && x_test <= segment.X_end) {
            slope_P = segment.P;
            intercept_O = segment.O;
            
            if (slope_P === Infinity) {
                // Error: segmento vertical
                y_pred = null; 
            } else {
                // Cálculo de la regresión lineal: Y = P * X + O
                y_pred = slope_P * x_test + intercept_O;
            }
            break;
        }
    }
    
    if (y_pred === null && slope_P !== Infinity) {
         // Manejo de extrapolación o fuera de rango (devuelve null)
        // console.warn(`SLRM-LOGOS: X=${x_test} está fuera del rango modelado.`);
    }

    return {
        x_in: x_test,
        y_pred: y_pred,
        slope_P: slope_P,
        intercept_O: intercept_O,
        cache_hit: cache_hit
    };
}

// Exportar las funciones auxiliares también si el módulo se usa en un entorno de prueba
// export { _cleanAndSortData, _losslessCompression, _lossyCompression };
