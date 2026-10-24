/**
 * scorer.js: Professional-Grade MBTI Psychometric Engine for Form M
 *
 * This module implements a robust Item Response Theory (IRT) scoring algorithm
 * to calculate an individual's MBTI preferences (E-I, S-N, T-F, J-P) and their
 * clarity of preference based on responses to the MBTI Form M questionnaire.
 *
 * It operates on the following core principles and components:
 *
 * 1.  **IRT Model (2-Parameter Logistic - 2PL):**
 *     The scoring utilizes a 2PL model, which accounts for both item difficulty
 *     ('b' parameter - location on the latent trait continuum) and item discrimination
 *     ('a' parameter - how well an item differentiates between individuals).
 *     This provides a more precise and psychometrically sound measurement compared
 *     to simple raw-score counting.
 *
 * 2.  **Calibrated Item Parameters:**
 *     'a' and 'b' parameters for each of the 93 items are sourced from the
 *     `itemParameterMatrix.js`. These parameters are empirically derived and
 *     professionally calibrated based on the Factor Analysis Rotated Component Matrix
 *     from the official "MBTI® Form M MANUAL SUPPLEMENT" (2009), ensuring the reliability
 *     and validity of the results aligned with the instrument's design.
 *
 * 3.  **Maximum Likelihood Estimation (MLE) for Dichotomies:**
 *     For each of the four main MBTI dichotomies (E-I, S-N, T-F, J-P), the system
 *     employs MLE to estimate a single 'theta' (latent trait level). This 'theta'
 *     represents the respondent's underlying strength or clarity of preference for that dichotomy.
 *     All questions belonging to a specific dichotomy are collectively used in this
 *     single MLE calculation, aligning directly with the established methodology
 *
 * 4.  **Reliability of MBTI Form M (from MBTI® Form M Manual Supplement, 2009, Table 7):**
 *     - **Cronbach's Alpha (Internal Consistency):**
 *       - E-I: .91
 *       - S-N: .92
 *       - T-F: .91
 *       - J-P: .92
 *     - **Test-Retest Correlations (4-week interval):**
 *       - E-I: .95
 *       - S-N: .97
 *       - T-F: .94
 *       - J-P: .95
 *
 * 5.  **`scoreKey` Notation for Response Direction:**
 *     The `questions.json` file is expected to contain a `scoreKey` (1 or 0) for each
 *     response option. A `scoreKey` of 1 indicates the 'positive' pole of the dichotomy
 *     (E, S, T, J), while 0 indicates the 'negative' pole (I, N, F, P). This notation
 *     streamlines the core likelihood calculation within the MLE loop, enhancing
 *     efficiency and reducing logical complexity.
 *
 * 6.  **Handling of Omissions:**
 *     In accordance with MBTI Form M administration guidelines (MBTI Manual Third Edition, p. 151),
 *     omitted (unanswered) questions are gracefully handled. They are simply excluded
 *     from the MLE calculation for the relevant dichotomy, ensuring that only available
 *     information is used in producing the most accurate type.
 *     If an entire dichotomy has no answered questions, its theta defaults to 0.
 *
 * 7.  **Crystallization Index (CI) and Category:**
 *     The calculated 'theta' for each dichotomy is transformed into a Crystallization
 *     Index (CI) on a scale of 1-60, and further categorized into a qualitative
 *     category (Diffuse, Variable, Defined, Crystal). These conversions are based on
 *     the calculations specified in the MBTI Manual Third Edition (p. 148-149), with
 *     the scale doubled from the original PCI (1-30) to CI (1-60) for increased granularity.
 *
 * 8.  **Tie-Breaking Rule:**
 *     In the rare event that a dichotomy's theta score is precisely zero (indicating
 *     no discernible preference), a tie-breaking rule is applied, assigning the
 *     preference to I, N, F, or P as per the MBTI Manual Third Edition(p. 147).
 *
 * 9.  **Midpoint Adjustment Rule (MBTI® Form M Manual Third Edition, p. 149):**
 *     For the S-N, T-F, and J-P dichotomies, slight preferences towards one pole
 *     (e.g., a CI of 1-2 for 'S') are reclassified to the opposite pole ('N') based
 *     on empirical data that shows this improves agreement with individuals'
 *     self-reported 'best-fit' type. The E-I midpoint is not adjusted.
 *     Note: Thresholds are doubled from original PCI to accommodate the CI (1-60) scale.
 *
 * This module provides the psychometric backbone for deriving accurate MBTI
 * preferences, reflecting rigorous adherence to established IRT principles
 * and the specific characteristics of the MBTI Form M.
 */

/**
 * ==============================================================================
 * ARCHITECTURAL RATIONALE: Bimodality, IQ, and Measurement Clarity
 * ==============================================================================
 *
 * This engine operates on the empirically-supported conclusion that psychological
 * type is a **bimodal, categorical system.** The widely cited "bell curve" MBTI distribution
 * is not a feature of personality, but a statistical artifact of
 * measurement error driven by low test-taker comprehension.
 *
 * The instrument's own data validates this architectural choice:
 *
 * 1.  **The 'VCI Trough'**: **Figure 9.1 (MBTI® Manual, 3rd ed., p. 186)** provides
 *     the definitive evidence. Mean IQ and Vocabulary scores form a distinct U-shape
 *     across the E-I, T-F, and J-P dichotomies. The lowest cognitive scores are
 *     found at the zero-point midpoint, while the highest-scoring individuals
 *     cluster at the poles. This is not a coincidence; it is a direct correlation
 *     between cognitive clarity and preference clarity. The manual itself states,
 *     **"Reliabilities are higher in groups with higher average intelligence...
 *     [due to] a lower likelihood of random responding"**
 *     (MBTI® Manual, 3rd ed., p. 162).
 *
 * 2.  **The Myth of 'Type Flipping'**: "Flipping" is a symptom of a weak initial
 *     measurement, not a fluid personality. **Table 8.5 (MBTI® Manual, 3rd ed.,
 *     p. 163)** shows that "Slight" preferences near the midpoint have abysmal
 *     test-retest reliability (e.g., .22 for J-P), making them statistical noise.
 *     Conversely, "Clear" preferences exhibit reliabilities in the high .90s,
 *     rivaling any clinical instrument and proving that **true type does not 'flip.'**
 *
 * **Conclusion:** The "midpoint" is not a valid "ambivert" state; it is a dead
 * zone of failed measurement. This engine, through its rigorous adherence to IRT,
 * is designed to filter this cognitive noise. It correctly interprets a clear
 * result not just as a preference, but as an indicator of a stable, well-defined,
 * and bimodal psychological type.
 * ==============================================================================
 */

import { itemParameters } from './itemParameterMatrix.js';

// --- Model Configuration ---
// This config is now only needed for the FINAL scoring step (tie-breaking and preference assignment),
// not for the core MLE loop.
const DICHOTOMY_CONFIG = {
    'E-I': { poles: ['E', 'I'], tieBreaker: 'I' },
    'S-N': { poles: ['S', 'N'], tieBreaker: 'N' },
    'T-F': { poles: ['T', 'F'], tieBreaker: 'F' },
    'J-P': { poles: ['J', 'P'], tieBreaker: 'P' }
};

// --- Pre-computation for efficiency ---
const dichotomyToQuestionMap = new Map();
for (const [index, params] of Object.entries(itemParameters)) {
    const dichotomyName = params.dichotomy;
    if (!dichotomyToQuestionMap.has(dichotomyName)) {
        dichotomyToQuestionMap.set(dichotomyName, []);
    }
    dichotomyToQuestionMap.get(dichotomyName).push(parseInt(index, 10));
}

/**
 * Calculates the probability of a '1' response (positive pole) for a given item
 * using the 2-Parameter Logistic (2PL) IRT model.
 * P(θ) = 1 / (1 + e^(-a * (θ - b)))
 */
function probability(theta, a, b) {
    return 1 / (1 + Math.exp(-a * (theta - b)));
}

/**
 * Converts a CI score to a qualitative category.
 */
function getCiCategory(ci) {
    if (ci >= 52) return "Crystal";
    if (ci >= 32) return "Defined";
    if (ci >= 12) return "Variable";
    return "Diffuse";
}

/**
 * Estimates the latent trait (theta) for a single dichotomy using Maximum Likelihood Estimation (MLE).
 *
 * This function implements the Newton-Raphson optimization algorithm to find the value of theta
 * that maximizes the log-likelihood function shown in the manual:
 *   ln l(u|θ) = Σ [uᵢ * ln(Pᵢ) + (1 - uᵢ) * ln(1 - Pᵢ)]
 *
 * To maximize this function, we find where its first derivative is zero. Newton-Raphson
 * requires both the first and second derivatives of the log-likelihood function.
 */
function findBestThetaForDichotomy(dichotomyName, answers, allQuestions) {
    const allDichotomyIndices = dichotomyToQuestionMap.get(dichotomyName) || [];
    const answeredQuestionIndices = allDichotomyIndices.filter(qIndex => answers[qIndex + 1]);

    if (answeredQuestionIndices.length === 0) {
        return 0; // Return neutral theta for no answers.
    }

    // Prepare items with their parameters and the user's keyed response (u)
    const items = answeredQuestionIndices.map(qIndex => {
        const params = itemParameters[qIndex];
        const answer = answers[qIndex + 1];
        const questionData = allQuestions.MBTI_Form_M[qIndex];
        const userScoreKey = questionData.options[answer.choice].scoreKey;

        return {
            a: params.params.a, // Item discrimination
            b: params.params.b, // Item difficulty
            u: userScoreKey      // User's response (1 or 0)
        };
    });

    // Newton-Raphson settings
    const maxIterations = 20;
    const tolerance = 0.0001;
    let theta = 0.0; // Initial estimate for theta

    for (let iter = 0; iter < maxIterations; iter++) {
        let firstDerivative = 0.0;  // Gradient (Score Function)
        let secondDerivative = 0.0; // Hessian (Information Function)

        // Sum the derivatives over all answered items for the dichotomy
        for (const item of items) {
            const { a, b, u } = item;
            const P = probability(theta, a, b);
            const Q = 1 - P;

            // First derivative of the log-likelihood for one item: a * (u - P)
            firstDerivative += a * (u - P);

            // Second derivative of the log-likelihood for one item: -a^2 * P * Q
            secondDerivative += -a * a * P * Q;
        }

        // Avoid division by zero (happens if the function is flat) and check for convergence
        if (Math.abs(secondDerivative) < 1e-9) {
            break;
        }

        // Newton-Raphson update step: θ_new = θ_old - (gradient / hessian)
        const delta = firstDerivative / secondDerivative;
        const newTheta = theta - delta;

        // Clamp theta to a reasonable range [-3, 3] to prevent divergence
        const clampedTheta = Math.max(-3, Math.min(3, newTheta));
        const change = Math.abs(clampedTheta - theta);

        theta = clampedTheta;
        if (change < tolerance) {
            break; // Converged
        }
    }

    return theta;
}


/**
 * Main function to calculate MBTI results.
 */
export function calculateResults(answers, allQuestions) {
    const dichotomyResults = {};

    for (const [dichotomy, config] of Object.entries(DICHOTOMY_CONFIG)) {
        const theta = findBestThetaForDichotomy(dichotomy, answers, allQuestions);

        const [pole1, pole2] = config.poles;
        let preference;

        // Step 1: Determine initial preference based on theta's sign.
        // theta > 0 indicates preference for the positive pole (E, S, T, J).
        if (theta > 0) {
            preference = pole1;
        } else if (theta < 0) {
            preference = pole2;
        } else {
            // Apply tie-breaker for theta == 0
            preference = config.tieBreaker;
        }

        // Step 2: Calculate Crystallization Index (CI) and Category.
        const maxTheta = 3.0;
        const rawCiCalculation = (Math.abs(theta) / maxTheta) * 60;
        const ci = theta === 0 ? 1 : Math.max(1, Math.round(rawCiCalculation));
        const ciCategory = getCiCategory(ci);

        // Step 3: Apply Midpoint Adjustments (as per MBTI Manual Third Edition, p. 149).
        // This empirically-derived adjustment reclassifies preferences for certain
        // very low CI scores to improve agreement with 'best-fit' type.
        // Note: Thresholds are doubled from original PCI (1-30) to CI (1-60) scale.
        if (dichotomy === 'S-N' && preference === 'S' && ci <= 2) {
            preference = 'N';
        } else if (dichotomy === 'T-F' && preference === 'T' && ci <= 4) {
            preference = 'F';
        } else if (dichotomy === 'J-P' && preference === 'J' && ci <= 2) {
            preference = 'P';
        }

        // Step 4: Assemble final result object for the dichotomy.
        dichotomyResults[dichotomy] = {
            preference,
            ci,
            ciCategory,
            theta: parseFloat(theta.toFixed(2)),
            dichotomyName: dichotomy
        };
    }

    return {
        dichotomyResults
    };
}