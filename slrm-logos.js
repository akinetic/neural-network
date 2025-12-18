/**
 * @fileoverview SLRM-LOGOS Core Algorithm (V5.12)
 * A Segmented Linear Regression Model (SLRM) implemented in pure JavaScript/Node.js.
 * This module provides functions for training and predicting using the Logos Core algorithm.
 * * Functions are synchronous and designed for compute environments (Node.js/Workers).
 * @author Alex Kinetic and Logos
 * Dependencies: None.
 */

// --- SLRM CORE LOGIC (V5.12) ---

// PRECISION CONSTANT FOR FLOATING POINT ARITHMETIC
const FLOAT_TOLERANCE = 1e-9; 

// --- DATA PROCESSING ---

/**
 * Cleans and sorts data, handling X duplicates by averaging Y.
 * @param {string} dataString - The input data as a string (e.g., "x1, y1\nx2, y2").
 * @returns {Array<Object>} - An array of cleaned and sorted {x: number, y: number} points.
 */
function _cleanAndSortData(dataString) {
    const pointsMap = new Map();
    
    dataString.split('\n').forEach(line => {
        const parts = line.trim().split(/[\s,]+/).map(p => parseFloat(p.trim()));
        if (parts.length >= 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
            const x = parts[0];
            const y = parts[1];
            
            if (pointsMap.has(x)) {
                // Handle duplicate X values by averaging Y
                const current = pointsMap.get(x);
                current.y = (current.y * current.count + y) / (current.count + 1);
                current.count += 1;
            } else {
                pointsMap.set(x, { x: x, y: y, count: 1 });
            }
        }
    });

    const cleanedData = Array.from(pointsMap.values()).map(p => ({x: p.x, y: p.y}));
    // Sort by X value
    cleanedData.sort((a, b) => a.x - b.x);

    return cleanedData;
}

// --- COMPRESSION LOGIC (Lossless: Geometric Invariance) ---

/**
 * Performs lossless compression by identifying non-collinear points
 * (slope changes). These form the "Base Breakpoints".
 * @param {Array<Object>} data - The cleaned and sorted data.
 * @returns {Array<number>} - An array of X values for the critical breakpoints.
 */
function _losslessCompression(data) {
    if (data.length < 2) return data.map(d => d.x);

    const criticalX = [data[0].x]; 
    
    for (let i = 1; i < data.length - 1; i++) {
        const p0 = data[i - 1]; 
        const p1 = data[i];     
        const p2 = data[i + 1]; 

        const dX_a = p1.x - p0.x;
        const dX_b = p2.x - p1.x;
        
        // Handle vertical segments or insignificant dX
        if (Math.abs(dX_a) < FLOAT_TOLERANCE || Math.abs(dX_b) < FLOAT_TOLERANCE) {
            criticalX.push(p1.x); 
            continue; 
        }

        const P_a = (p1.y - p0.y) / dX_a; 
        const P_b = (p2.y - p1.y) / dX_b; 

        // Collinearity check: if slopes are different beyond tolerance, it's a critical point.
        if (Math.abs(P_a - P_b) > FLOAT_TOLERANCE) {
            criticalX.push(p1.x);
        }
    }

    if (data.length > 1) {
        criticalX.push(data[data.length - 1].x);
    }

    // Ensure uniqueness, although the logic should minimize duplicates
    return Array.from(new Set(criticalX));
}

// --- COMPRESSION LOGIC (Lossy / SLRM) ---
        
/**
 * Calculates the maximum error of a committed segment against all 
 * intermediate true data points.
 */
function _calculateSegmentMaxError(xS, xE, P, O, dataKeys, data) {
    if (isNaN(P)) return 0.0;
        
    const startIndex = dataKeys.indexOf(xS);
    const endIndex = dataKeys.indexOf(xE);
    let maxErr = 0.0;
    
    // Iterate ONLY over strictly intermediate points
    for (let k = startIndex + 1; k < endIndex; k++) { 
        const xMid = data[k].x;
        const yTrueMid = data[k].y;
        
        const yHatMid = P * xMid + O;
        const error = Math.abs(yTrueMid - yHatMid);
        
        maxErr = Math.max(maxErr, error);
    }
    return maxErr;
}


/**
 * Applies the lossy compression using the SLRM (MRLS) algorithm.
 * @param {Array<number>} initialKeys - X values from the lossless compression (breakpoints).
 * @param {number} epsilon - The maximum allowed error tolerance.
 * @param {Array<Object>} data - The cleaned and sorted original data.
 * @returns {Object} - { finalModel: Map<number, Array<number>>, maxError: number }
 * finalModel: Map where key is X_start, value is [Slope P, Intercept O, X_end].
 */
function _lossyCompression(initialKeys, epsilon, data) {
    const dataKeys = data.map(d => d.x);
    const dataMap = new Map(data.map(p => [p.x, p.y]));
    
    if (initialKeys.length < 2) return { finalModel: new Map(), maxError: 0 };
    
    const finalModel = new Map();
    let i = 0; // index of the start point (x_start) in initialKeys
    let maxOverallError = 0;
    // Use 1e-12 for epsilon=0 to prevent division by zero in error checks
    const safeEpsilon = Math.max(epsilon, 1e-12); 

    while (i < initialKeys.length - 1) {
        
        const x_start = initialKeys[i];
        const y_start = dataMap.get(x_start);
        
        let j = i + 1; // index of the candidate end point (x_end_candidate) in initialKeys

        while (j < initialKeys.length) {
            
            const x_end_candidate = initialKeys[j];
            const y_end_candidate = dataMap.get(x_end_candidate);

            const dX = x_end_candidate - x_start;
            
            if (isNaN(dX) || Math.abs(dX) < FLOAT_TOLERANCE) { 
                j++; 
                continue; 
            } 

            // Calculate candidate segment: Slope (P) and Intercept (O)
            const P_test = (y_end_candidate - y_start) / dX;
            const O_test = y_start - P_test * x_start;

            let errorExceeded = false;
            
            const startIndex = dataKeys.indexOf(x_start);
            const endIndex = dataKeys.indexOf(x_end_candidate);

            // Check all *intermediate* points against Epsilon
            for (let k = startIndex + 1; k < endIndex; k++) {
                const x_mid = data[k].x;
                const y_true_mid = data[k].y;
                
                const y_hat_mid = P_test * x_mid + O_test;
                const error = Math.abs(y_true_mid - y_hat_mid);

                if (error > safeEpsilon) { 
                    errorExceeded = true;
                    break; 
                }
            }
            
            if (errorExceeded) {
                // Segment failed. Commit the previous segment (j-1).
                const x_end_committed = initialKeys[j - 1];
                const y_end_committed = dataMap.get(x_end_committed);
                
                const dX_committed = x_end_committed - x_start;
                // Avoid division by zero for P if dX is near zero
                const P = dX_committed === 0 ? 0 : (y_end_committed - y_start) / dX_committed;
                const O = y_start - P * x_start;
                
                // Save segment: [Slope, Intercept, X_End]
                finalModel.set(x_start, [P, O, x_end_committed]);
                
                const committedMaxError = _calculateSegmentMaxError(x_start, x_end_committed, P, O, dataKeys, data);
                maxOverallError = Math.max(maxOverallError, committedMaxError);

                i = j - 1; // Next segment starts where the previous one ended.
                break; 
                
            } else if (j === initialKeys.length - 1) {
                // Last point reached and passed the test. Commit the final segment.
                const x_end = initialKeys[j];
                const y_end = dataMap.get(x_end);
                
                const dX = x_end - x_start;
                const P = dX === 0 ? 0 : (y_end - y_start) / dX; 
                const O = y_start - P * x_start;
                    
                finalModel.set(x_start, [P, O, x_end]);
                
                const finalMaxError = _calculateSegmentMaxError(x_start, x_end, P, O, dataKeys, data);
                maxOverallError = Math.max(maxOverallError, finalMaxError);
                
                i = j; // Ends the outer loop
                break;
            }
            
            j++; // Try to extend the segment further
        }
    }

    // Final marker for the last point
    const lastKey = initialKeys[initialKeys.length - 1];
    if (lastKey !== undefined && !finalModel.has(lastKey)) {
         // Use [NaN, NaN, NaN] for end points that are not segment starts
         finalModel.set(lastKey, [NaN, NaN, NaN]);
    }

    return { finalModel, maxError: maxOverallError };
}


// --- MAIN TRAINING AND PREDICTION ---

/**
 * Main function to train the SLRM model.
 * @param {string} dataString - The raw data string.
 * @param {number} epsilon - The maximum error tolerance (must be >= 0).
 * @returns {Object} - { model: Map<number, Array<number>>, originalData: Array<Object>, maxError: number }
 */
function train_slrm(dataString, epsilon) {
    if (isNaN(epsilon) || epsilon < 0) {
        epsilon = 0;
    }
    
    const originalPoints = _cleanAndSortData(dataString);
    if (originalPoints.length < 2) return { model: new Map(), originalData: originalPoints, maxError: 0 };
    
    const initialBreakpoints = _losslessCompression(originalPoints);
    
    // Lossy Compression is now synchronous
    const result = _lossyCompression(initialBreakpoints, epsilon, originalPoints);

    return { model: result.finalModel, originalData: originalPoints, maxError: result.maxError };
}


/**
 * Performs a Y prediction for a given X value using the trained model.
 * * NOTE: This function requires the 'originalData' (or min/max X) to correctly handle 
 * extrapolation outside the model's range.
 * * @param {number} x_in - The input X value for prediction.
 * @param {Map<number, Array<number>>} model - The trained model Map.
 * @param {Array<Object>} originalData - The clean and sorted original data points.
 * @returns {Object} - { x_in: number, y_pred: number, slope_P: number, intercept_O: number }
 */
function predict_slrm(x_in, model, originalData) {
    // 1. Prepare keys and bounds
    const keys = Array.from(model.entries())
        // Filter only entries that are valid segment starts (have Slope P)
        .filter(([x_start, [P]]) => !isNaN(P)) 
        .map(([x_start]) => x_start); 
    keys.sort((a, b) => a - b); 

    if (keys.length === 0 || originalData.length === 0) {
        return { x_in, y_pred: NaN, slope_P: NaN, intercept_O: NaN };
    }

    const dataMinX = originalData[0].x;
    const dataMaxX = originalData[originalData.length - 1].x;
    
    let activeKey = null;

    // 2. Extrapolation/Interpolation logic
    if (x_in < dataMinX) {
        // Extrapolation before data range: use the first segment
        activeKey = keys[0];
    } else if (x_in >= dataMaxX) {
        // Extrapolation after data range: use the last segment
        activeKey = keys[keys.length - 1]; 
    } else {
        // Interpolation: find the segment where x_in falls
        for (let i = 0; i < keys.length; i++) {
            const x_start = keys[i];
            const x_end = model.get(x_start)[2]; // x_end is the 3rd element in the segment array
            
            // Check if x_in is in [x_start, x_end)
            if (x_in >= x_start && x_in < x_end) { 
                activeKey = x_start;
                break;
            }
        }
        // Fallback for the last segment if x_in is exactly the last breakpoint (x_max)
        if (activeKey === null) {
            activeKey = keys[keys.length - 1];
        }
    }
    
    if (activeKey === undefined) {
         return { x_in, y_pred: NaN, slope_P: NaN, intercept_O: NaN };
    }

    // 3. Calculation
    // Segment format: [P, O, x_end]
    const [P, O] = model.get(activeKey) || [NaN, NaN];
    
    let y_pred = NaN;
    if (!isNaN(P) && !isNaN(O)) {
        y_pred = x_in * P + O;
    }

    return {
        x_in,
        y_pred,
        slope_P: P,
        intercept_O: O
    };
}

// Export the primary functions for use as an NPM module
module.exports = {
    train_slrm,
    predict_slrm,
    FLOAT_TOLERANCE
};
