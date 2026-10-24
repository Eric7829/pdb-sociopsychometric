/**
 * itemParameterMatrix.js
 *
 * This matrix contains professionally calibrated 2-Parameter Logistic (2PL)
 * Item Response Theory (IRT) parameters for MBTI Form M items, reverse engineered from
 * thanks to my meticulous gathering of public academic research on the MBTI® Form M.
 *
 * METHODOLOGY:
 * - 'b' parameter (difficulty/location): Sourced from Evaluating the MBTI® Form M in a South African context
 *   (Taylor & Zyl, 2011).This parameter
 *   indicates the point on the latent trait continuum where a respondent has a
 *   50% probability of endorsing the item's positive pole.
 * - 'a' parameter (discrimination/slope): Calculated from the factor loadings
 *   published in the Official MBTI Form M Manual Supplement (Schaubhut et al., 2009). The sample size was 10,000 Americans.
 *   The conversion uses the standard formula: a = (lambda / sqrt(1 - lambda^2)) * 1.702 (McDonald, 1999). This ensures
 *   that items with higher empirical loadings have a stronger ability to
 *   discriminate between individuals in the IRT model.
 * - Dichotomy: Assigned based on the item's primary factor loading.
 */
export const itemParameters = {
    "0": {
        "dichotomy": "J-P",
        "params": {
            "a": 1.5785,
            "b": 0.93
        }
    },
    "1": {
        "dichotomy": "S-N",
        "params": {
            "a": 1.1504,
            "b": 1.1
        }
    },
    "2": {
        "dichotomy": "E-I",
        "params": {
            "a": 1.9903,
            "b": -0.01
        }
    },
    "3": {
        "dichotomy": "J-P",
        "params": {
            "a": 1.5361,
            "b": -0.24
        }
    },
    "4": {
        "dichotomy": "S-N",
        "params": {
            "a": 1.0361,
            "b": -0.75
        }
    },
    "5": {
        "dichotomy": "T-F",
        "params": {
            "a": 0.9063,
            "b": 0.62
        }
    },
    "6": {
        "dichotomy": "E-I",
        "params": {
            "a": 1.2118,
            "b": -0.09
        }
    },
    "7": {
        "dichotomy": "J-P",
        "params": {
            "a": 1.5785,
            "b": -0.22
        }
    },
    "8": {
        "dichotomy": "S-N",
        "params": {
            "a": 1.1209,
            "b": -0.84
        }
    },
    "9": {
        "dichotomy": "E-I",
        "params": {
            "a": 1.1504,
            "b": -0.65
        }
    },
    "10": {
        "dichotomy": "S-N",
        "params": {
            "a": 1.3449,
            "b": -0.31
        }
    },
    "11": {
        "dichotomy": "J-P",
        "params": {
            "a": 1.2118,
            "b": 0.72
        }
    },
    "12": {
        "dichotomy": "E-I",
        "params": {
            "a": 1.1504,
            "b": 0.76
        }
    },
    "13": {
        "dichotomy": "J-P",
        "params": {
            "a": 0.9827,
            "b": 0.67
        }
    },
    "14": {
        "dichotomy": "T-F",
        "params": {
            "a": 0.9063,
            "b": -0.8
        }
    },
    "15": {
        "dichotomy": "E-I",
        "params": {
            "a": 1.2118,
            "b": 0.42
        }
    },
    "16": {
        "dichotomy": "J-P",
        "params": {
            "a": 0.4964,
            "b": 0.13
        }
    },
    "17": {
        "dichotomy": "S-N",
        "params": {
            "a": 0.6778,
            "b": 0.88
        }
    },
    "18": {
        "dichotomy": "E-I",
        "params": {
            "a": 1.2437,
            "b": 0.98
        }
    },
    "19": {
        "dichotomy": "J-P",
        "params": {
            "a": 0.9827,
            "b": 0.44
        }
    },
    "20": {
        "dichotomy": "T-F",
        "params": {
            "a": 1.2437,
            "b": 0.51
        }
    },
    "21": {
        "dichotomy": "S-N",
        "params": {
            "a": 0.8576,
            "b": -0.4
        }
    },
    "22": {
        "dichotomy": "E-I",
        "params": {
            "a": 0.9063,
            "b": 0.52
        }
    },
    "23": {
        "dichotomy": "J-P",
        "params": {
            "a": 1.1209,
            "b": 1.18
        }
    },
    "24": {
        "dichotomy": "S-N",
        "params": {
            "a": 0.7428,
            "b": -1.58
        }
    },
    "25": {
        "dichotomy": "E-I",
        "params": {
            "a": 1.5785,
            "b": -0.61
        }
    },
    "26": {
        "dichotomy": "S-N",
        "params": {
            "a": 1.092,
            "b": 1.35
        }
    },
    "27": {
        "dichotomy": "J-P",
        "params": {
            "a": 1.5785,
            "b": 0.21
        }
    },
    "28": {
        "dichotomy": "T-F",
        "params": {
            "a": 1.0361,
            "b": -0.4
        }
    },
    "29": {
        "dichotomy": "S-N",
        "params": {
            "a": 1.3102,
            "b": 1.11
        }
    },
    "30": {
        "dichotomy": "T-F",
        "params": {
            "a": 1.4176,
            "b": 1.05
        }
    },
    "31": {
        "dichotomy": "E-I",
        "params": {
            "a": 2.1931,
            "b": -0.37
        }
    },
    "32": {
        "dichotomy": "T-F",
        "params": {
            "a": 1.2765,
            "b": -0.9
        }
    },
    "33": {
        "dichotomy": "S-N",
        "params": {
            "a": 1.1209,
            "b": -0.37
        }
    },
    "34": {
        "dichotomy": "T-F",
        "params": {
            "a": 1.3449,
            "b": -0.68
        }
    },
    "35": {
        "dichotomy": "J-P",
        "params": {
            "a": 0.9063,
            "b": 0.51
        }
    },
    "36": {
        "dichotomy": "T-F",
        "params": {
            "a": 1.1807,
            "b": -0.84
        }
    },
    "37": {
        "dichotomy": "E-I",
        "params": {
            "a": 1.6225,
            "b": -0.71
        }
    },
    "38": {
        "dichotomy": "S-N",
        "params": {
            "a": 0.8817,
            "b": -0.33
        }
    },
    "39": {
        "dichotomy": "T-F",
        "params": {
            "a": 1.2765,
            "b": 1.32
        }
    },
    "40": {
        "dichotomy": "J-P",
        "params": {
            "a": 0.7428,
            "b": -0.9
        }
    },
    "41": {
        "dichotomy": "E-I",
        "params": {
            "a": 1.6683,
            "b": 0.31
        }
    },
    "42": {
        "dichotomy": "T-F",
        "params": {
            "a": 0.7651,
            "b": -0.73
        }
    },
    "43": {
        "dichotomy": "S-N",
        "params": {
            "a": 1.1209,
            "b": 1.1
        }
    },
    "44": {
        "dichotomy": "T-F",
        "params": {
            "a": 0.6778,
            "b": -0.87
        }
    },
    "45": {
        "dichotomy": "S-N",
        "params": {
            "a": 1.1807,
            "b": 0.23
        }
    },
    "46": {
        "dichotomy": "T-F",
        "params": {
            "a": 0.9567,
            "b": 1.02
        }
    },
    "47": {
        "dichotomy": "S-N",
        "params": {
            "a": 1.2765,
            "b": 1.43
        }
    },
    "48": {
        "dichotomy": "T-F",
        "params": {
            "a": 1.0361,
            "b": 1.02
        }
    },
    "49": {
        "dichotomy": "S-N",
        "params": {
            "a": 0.9827,
            "b": -0.17
        }
    },
    "50": {
        "dichotomy": "T-F",
        "params": {
            "a": 1.1807,
            "b": -0.6
        }
    },
    "51": {
        "dichotomy": "S-N",
        "params": {
            "a": 0.8339,
            "b": -0.67
        }
    },
    "52": {
        "dichotomy": "T-F",
        "params": {
            "a": 1.4176,
            "b": -0.46
        }
    },
    "53": {
        "dichotomy": "S-N",
        "params": {
            "a": 0.9063,
            "b": -0.57
        }
    },
    "54": {
        "dichotomy": "J-P",
        "params": {
            "a": 0.8576,
            "b": 0.08
        }
    },
    "55": {
        "dichotomy": "T-F",
        "params": {
            "a": 1.2437,
            "b": -0.11
        }
    },
    "56": {
        "dichotomy": "E-I",
        "params": {
            "a": 1.2765,
            "b": -1.18
        }
    },
    "57": {
        "dichotomy": "T-F",
        "params": {
            "a": 1.1209,
            "b": 1.14
        }
    },
    "58": {
        "dichotomy": "J-P",
        "params": {
            "a": 1.3449,
            "b": -0.37
        }
    },
    "59": {
        "dichotomy": "S-N",
        "params": {
            "a": 1.1504,
            "b": 0.6
        }
    },
    "60": {
        "dichotomy": "T-F",
        "params": {
            "a": 1.1807,
            "b": -0.05
        }
    },
    "61": {
        "dichotomy": "E-I",
        "params": {
            "a": 1.2765,
            "b": 0.05
        }
    },
    "62": {
        "dichotomy": "S-N",
        "params": {
            "a": 1.0638,
            "b": -0.05
        }
    },
    "63": {
        "dichotomy": "J-P",
        "params": {
            "a": 0.7651,
            "b": -0.77
        }
    },
    "64": {
        "dichotomy": "S-N",
        "params": {
            "a": 1.2765,
            "b": 0.78
        }
    },
    "65": {
        "dichotomy": "T-F",
        "params": {
            "a": 1.3102,
            "b": -0.38
        }
    },
    "66": {
        "dichotomy": "S-N",
        "params": {
            "a": 1.3449,
            "b": -0.7
        }
    },
    "67": {
        "dichotomy": "E-I",
        "params": {
            "a": 1.0361,
            "b": 1.15
        }
    },
    "68": {
        "dichotomy": "S-N",
        "params": {
            "a": 0.8339,
            "b": -0.42
        }
    },
    "69": {
        "dichotomy": "T-F",
        "params": {
            "a": 1.0638,
            "b": -1.67
        }
    },
    "70": {
        "dichotomy": "S-N",
        "params": {
            "a": 1.0361,
            "b": -1.5
        }
    },
    "71": {
        "dichotomy": "T-F",
        "params": {
            "a": 1.0091,
            "b": -0.31
        }
    },
    "72": {
        "dichotomy": "S-N",
        "params": {
            "a": 1.3102,
            "b": 0.09
        }
    },
    "73": {
        "dichotomy": "E-I",
        "params": {
            "a": 1.3102,
            "b": 0.55
        }
    },
    "74": {
        "dichotomy": "T-F",
        "params": {
            "a": 1.1504,
            "b": 1.05
        }
    },
    "75": {
        "dichotomy": "J-P",
        "params": {
            "a": 1.6683,
            "b": 0.62
        }
    },
    "76": {
        "dichotomy": "E-I",
        "params": {
            "a": 1.092,
            "b": 0.79
        }
    },
    "77": {
        "dichotomy": "J-P",
        "params": {
            "a": 1.6683,
            "b": -1.2
        }
    },
    "78": {
        "dichotomy": "E-I",
        "params": {
            "a": 1.9299,
            "b": 0.24
        }
    },
    "79": {
        "dichotomy": "J-P",
        "params": {
            "a": 1.4558,
            "b": -0.98
        }
    },
    "80": {
        "dichotomy": "E-I",
        "params": {
            "a": 1.1807,
            "b": -0.3
        }
    },
    "81": {
        "dichotomy": "S-N",
        "params": {
            "a": 1.1504,
            "b": -0.66
        }
    },
    "82": {
        "dichotomy": "E-I",
        "params": {
            "a": 1.4952,
            "b": -0.99
        }
    },
    "83": {
        "dichotomy": "J-P",
        "params": {
            "a": 1.5361,
            "b": 0.63
        }
    },
    "84": {
        "dichotomy": "E-I",
        "params": {
            "a": 1.2118,
            "b": -1.16
        }
    },
    "85": {
        "dichotomy": "J-P",
        "params": {
            "a": 1.0091,
            "b": 0.21
        }
    },
    "86": {
        "dichotomy": "T-F",
        "params": {
            "a": 1.2118,
            "b": 0.32
        }
    },
    "87": {
        "dichotomy": "J-P",
        "params": {
            "a": 1.5361,
            "b": -1.05
        }
    },
    "88": {
        "dichotomy": "T-F",
        "params": {
            "a": 0.5749,
            "b": 0.75
        }
    },
    "89": {
        "dichotomy": "J-P",
        "params": {
            "a": 1.0638,
            "b": -1.01
        }
    },
    "90": {
        "dichotomy": "E-I",
        "params": {
            "a": 1.6683,
            "b": 0.3
        }
    },
    "91": {
        "dichotomy": "S-N",
        "params": {
            "a": 0.5353,
            "b": 0.61
        }
    },
    "92": {
        "dichotomy": "J-P",
        "params": {
            "a": 1.4558,
            "b": 0.43
        }
    }
};
