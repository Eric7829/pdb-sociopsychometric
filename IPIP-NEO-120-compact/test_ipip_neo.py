import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ScoringTests(unittest.TestCase):
    def setUp(self):
        import ipip_neo
        self.app = ipip_neo

    def test_neutral_answers_have_midpoint_raw_scores(self):
        result = self.app.score([3] * 120)
        self.assertEqual(len(result['scores']), 35)
        for key, value in result['scores'].items():
            self.assertEqual(value['raw'], 72 if len(key) == 1 else 12)

    def test_reverse_keying_uses_original_response_not_corrected_response(self):
        answers = [3] * 120
        answers[8] = 5  # Q9: Use others for my own ends (reverse A2).
        result = self.app.score(answers)
        self.assertEqual(result['scores']['A2']['raw'], 10)
        self.assertEqual(result['scores']['A']['raw'], 70)
        self.assertEqual(result['scores']['N']['raw'], 72)

    def test_midrank_percentile_against_tiny_known_distribution(self):
        self.assertEqual(self.app.midrank_percentile(5, [1, 2, 1], 4), 50)
        self.assertEqual(self.app.midrank_percentile(4, [1, 2, 1], 4), 12.5)
        self.assertEqual(self.app.midrank_percentile(6, [1, 2, 1], 4), 87.5)

    def test_rejects_incomplete_out_of_range_and_noninteger_responses(self):
        for answers in ([3] * 119, [3] * 121, [0] * 120, [6] * 120,
                        [True] * 120, [3.0] * 120, [None] * 120):
            with self.subTest(answers=answers[:2]), self.assertRaises(ValueError):
                self.app.score(answers)

    def test_every_curve_is_monotonic_and_matches_reference_count(self):
        norms = self.app.load_norms()
        for key, scale in norms['scales'].items():
            self.assertEqual(sum(scale['counts']), 410376)
            curve = self.app.percentile_curve(key)
            ps = [p['percentile'] for p in curve]
            self.assertEqual(ps, sorted(ps))
            self.assertEqual(curve[0]['raw'], scale['minimum'])
            self.assertEqual(curve[-1]['raw'], scale['maximum'])

    def test_cli_scores_json_from_another_working_directory(self):
        script = Path(__file__).with_name('ipip_neo.py')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / 'answers.json').write_text(json.dumps([3] * 120))
            run = subprocess.run([sys.executable, str(script), 'score',
                                  'answers.json', '--output', 'report.json'],
                                 cwd=p, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads((p / 'report.json').read_text())
            self.assertEqual(report['scores']['N']['raw'], 72)
            self.assertIn('410,376', run.stdout)

    def test_interactive_quit_resume_and_finish(self):
        script = Path(__file__).with_name('ipip_neo.py')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            cmd = [sys.executable, str(script), 'take', '--answers', str(p/'answers.json'),
                   '--output', str(p/'report.json')]
            first = subprocess.run(cmd, input='invalid\n5\nback\n2\nq\n', text=True,
                                   capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            saved = json.loads((p/'answers.json').read_text())
            self.assertEqual(saved['responses'][0], 2)
            self.assertEqual(saved['responses'][1:], [None] * 119)
            self.assertFalse((p/'report.json').exists())
            second = subprocess.run(cmd, input='3\n' * 119, text=True, capture_output=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(json.loads((p/'report.json').read_text())['scores']['N1']['raw'], 11)


if __name__ == '__main__':
    unittest.main()
