"""Offline IPIP-NEO-120 questionnaire and empirical scoring. Python 3.10+."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

BASE = Path(__file__).resolve().parent


def load_items():
    """Load the original 120 items and reverse-scoring key."""
    return json.loads((BASE / 'items.json').read_text(encoding='utf-8'))


def load_norms():
    """Load pooled empirical histograms, not normal-curve approximations."""
    return json.loads((BASE / 'empirical-norms.json').read_text(encoding='utf-8'))


def midrank_percentile(raw, counts, minimum):
    """Percent below raw plus half of ties, on a 0..100 scale."""
    if type(raw) is not int or not minimum <= raw < minimum + len(counts):
        raise ValueError('Raw score outside the scale range.')
    index = raw - minimum
    return 100 * (sum(counts[:index]) + counts[index] / 2) / sum(counts)


def percentile_curve(scale):
    """Return every raw-score / percentile pair for N, E, O, A, C or a facet."""
    norms = load_norms()
    scale = scale.upper()
    if scale not in norms['scales']:
        raise ValueError('Unknown scale; use N, E, O, A, C or a facet such as N1.')
    s = norms['scales'][scale]
    return [{'raw': raw, 'percentile': midrank_percentile(raw, s['counts'], s['minimum'])}
            for raw in range(s['minimum'], s['maximum'] + 1)]


def validate_responses(responses, allow_partial=False):
    if not isinstance(responses, list) or len(responses) != 120:
        raise ValueError('Responses must be a list of exactly 120 answers in original item order.')
    for i, value in enumerate(responses, 1):
        if allow_partial and value is None:
            continue
        if type(value) is not int or not 1 <= value <= 5:
            raise ValueError(f'Question {i}: answer must be an integer from 1 to 5.')


def score(responses):
    """Score 120 ORIGINAL (not reverse-corrected) answers; return all 35 scales."""
    validate_responses(responses)
    norms = load_norms()
    raw = {key: 0 for key in norms['scales']}
    for item, answer in zip(load_items(), responses):
        value = 6 - answer if item['reverse'] else answer
        raw[item['facet']] += value
        raw[item['facet'][0]] += value
    scores = {}
    for key, s in norms['scales'].items():
        scores[key] = {'name': s['name'], 'raw': raw[key],
                       'minimum': s['minimum'], 'maximum': s['maximum'],
                       'percentile': midrank_percentile(raw[key], s['counts'], s['minimum'])}
    return {'instrument': 'Johnson IPIP-NEO-120', 'schema_version': 1,
            'scored_at_utc': datetime.now(timezone.utc).isoformat(),
            'reference': {key: norms[key] for key in
                          ('source', 'dataset', 'dataset_sha256', 'rows', 'complete_rows',
                           'filter', 'percentile')},
            'reference_note': 'Pooled self-selected internet sample; not population-representative. '
                              'Locally derived empirical norms, not author-published norms.',
            'responses': list(responses), 'scores': scores}


def read_responses(path, allow_partial=False):
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if isinstance(data, dict):
        data = data.get('responses')
    validate_responses(data, allow_partial)
    return data


def write_json(path, data):
    """Replace a saved file atomically so interruption doesn't truncate progress."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         delete=False, suffix='.tmp') as file:
            temporary = Path(file.name)
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write('\n')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def format_report(report):
    lines = ['IPIP-NEO-120 RESULTS',
             f"Reference: {report['reference']['complete_rows']:,} complete responses.",
             report['reference_note'],
             'Percentile = % scoring below you, with half of ties included.', '',
             f"{'Scale':<6} {'Name':<25} {'Raw':>8} {'Percentile':>12}"]
    for key, s in report['scores'].items():
        lines.append(f"{key:<6} {s['name']:<25} {s['raw']:>3}/{s['maximum']:<4} {s['percentile']:>10.2f}")
    lines += ['', 'Domains are scored from raw items, not by averaging facet percentiles.',
              'Higher percentiles mean more of the named trait; they are not better/worse grades.']
    return '\n'.join(lines) + '\n'


def save_report(report, path):
    path = Path(path)
    write_json(path, report)
    path.with_suffix('.txt').write_text(format_report(report), encoding='utf-8')


def take(answers_path, output_path):
    answers_path = Path(answers_path)
    responses = read_responses(answers_path, allow_partial=True) if answers_path.exists() else [None] * 120
    print('IPIP-NEO-120 | Describe yourself as you generally are now.')
    print('1 Very Inaccurate | 2 Moderately Inaccurate | 3 Neither Accurate nor Inaccurate')
    print('4 Moderately Accurate | 5 Very Accurate')
    print('Enter back to change the previous answer; q to save and quit. Answers stay on this computer.\n')
    index = next((i for i, x in enumerate(responses) if x is None), 120)
    items = load_items()
    while index < 120:
        try:
            value = input(f"{index+1}/120. {items[index]['text']}\nAnswer [1-5]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print('\nSession paused.')
            value = 'q'
        if value in ('q', 'quit'):
            write_json(answers_path, {'instrument': 'Johnson IPIP-NEO-120', 'responses': responses})
            print(f'Saved progress to {answers_path}. Run the same command to resume.')
            return
        if value == 'back':
            index = max(0, index - 1)
            continue
        if value not in ('1', '2', '3', '4', '5'):
            print('Enter 1, 2, 3, 4, 5, back, or q.')
            continue
        responses[index] = int(value)
        write_json(answers_path, {'instrument': 'Johnson IPIP-NEO-120', 'responses': responses})
        index += 1
    report = score(responses)
    save_report(report, output_path)
    print('\n' + format_report(report))
    print(f'Saved {output_path} and {Path(output_path).with_suffix(".txt")}.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    interactive = sub.add_parser('take', help='Take or resume the 120-question assessment.')
    interactive.add_argument('--answers', type=Path, default=Path('answers.json'))
    interactive.add_argument('--output', type=Path, default=Path('results.json'))
    scorer = sub.add_parser('score', help='Score a saved JSON list or object with responses.')
    scorer.add_argument('answers', type=Path)
    scorer.add_argument('--output', type=Path, default=Path('results.json'))
    curve = sub.add_parser('curve', help='Print a complete empirical percentile curve or one raw score.')
    curve.add_argument('scale')
    curve.add_argument('--raw', type=int)
    args = parser.parse_args(argv)
    try:
        if args.command in ('take', 'score'):
            if args.output.suffix.lower() != '.json':
                raise ValueError('--output must have a .json extension (a .txt report is also saved).')
            if args.answers.resolve() == args.output.resolve():
                raise ValueError('Answers and results must use different paths.')
        if args.command == 'take':
            take(args.answers, args.output)
        elif args.command == 'score':
            report = score(read_responses(args.answers))
            save_report(report, args.output)
            print(format_report(report))
            print(f'Saved {args.output} and {args.output.with_suffix(".txt")}.')
        else:
            points = percentile_curve(args.scale)
            if args.raw is not None:
                points = [p for p in points if p['raw'] == args.raw]
                if not points:
                    raise ValueError('Raw score outside the scale range.')
            print('Raw  Percentile')
            for point in points:
                print(f"{point['raw']:3}  {point['percentile']:10.2f}")
    except (ValueError, OSError) as error:
        parser.exit(2, f'Error: {error}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
