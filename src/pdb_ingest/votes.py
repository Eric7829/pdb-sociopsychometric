import math
from .models import Distribution

TRAINING_SYSTEMS = {'MBTI', 'ENNEAGRAM', 'INSTINCT', 'TRITYPE', 'AP'}

def build_distributions(counts, evidence):
    result = []
    for system, observed in counts.items():
        if any(type(v) is not int or v < 0 for v in observed.values()):
            raise ValueError('Vote counts must be nonnegative integers')
        adjusted = dict(observed)
        relevant = [e for e in evidence if e.system_name == system]
        labels = {e.label for e in relevant if e.confirmed}
        status = 'UNKNOWN'
        if len(labels) > 1:
            status = 'CONFLICT'
        elif labels:
            label = next(iter(labels))
            if label in adjusted and adjusted[label] > 0:
                adjusted[label] = max(0, adjusted[label] - 1)
                status = 'SUBTRACTED'
            else:
                status = 'INCONSISTENT'
        elif relevant:
            status = 'UNCONFIRMED'
        total = sum(adjusted.values())
        simplex = {k: v / total for k, v in adjusted.items()} if total else {}
        entropy = -sum(p * math.log2(p) for p in simplex.values() if p) if total else None
        result.append(Distribution(system, dict(observed), adjusted, total, simplex, entropy,
                                   status, system in TRAINING_SYSTEMS and total >= 10, relevant))
    return result
