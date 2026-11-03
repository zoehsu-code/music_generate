"""Metrics are pointwise distances, including offsets Pearson cannot detect."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('report', Path(__file__).parents[1] / 'scripts/build_ossl_report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class AbsoluteDifferenceTests(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(report.absolute_difference([1, 2], [1, 2]), (0, 0, [0, 0]))

    def test_offset_and_outlier(self):
        mean, median, diff = report.absolute_difference([0, 1, 2], [1, 2, 12])
        self.assertEqual(diff, [1, 1, 10])
        self.assertEqual(mean, 4)
        self.assertEqual(median, 1)

    def test_invalid_inputs(self):
        for video, music in [([], []), ([1], [1, 2]), ([float('nan')], [0]), ([0], [float('inf')])]:
            with self.assertRaises(ValueError):
                report.absolute_difference(video, music)
