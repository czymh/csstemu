"""Exercise snapshot failure detection without rebuilding the baseline."""
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import test_regression_snapshot as snapshot


class FakeArchive(dict):
    @property
    def files(self):
        return list(self)


class TestSnapshotGuards(unittest.TestCase):
    def setUp(self):
        self.values = {'probe': np.array([1.0, 2.0])}
        for tag in ('xihm', 'pkhm', 'galaxy_Xigm', 'galaxy_DeltaSigma'):
            for mode in ('seq', 'fresh'):
                key = 'callorder|callorder|callorder|co_%s_%s' % (tag, mode)
                self.values[key] = np.array([0.0, 1.0])

    def compare(self, actual, baseline=None):
        baseline = self.values if baseline is None else baseline
        with patch.object(snapshot, 'collect_all', return_value=(actual, {}, {}, {})), \
                patch.object(snapshot.np, 'load', return_value=FakeArchive(baseline)), \
                patch.object(snapshot.json, 'load', return_value={'errors': {}}), \
                contextlib.redirect_stdout(io.StringIO()):
            return snapshot.compare(1e-12, [])

    def test_unchanged_and_zero_callorder(self):
        self.assertEqual(self.compare(self.values), 0)

    def test_nonfinite_changes_fail(self):
        for before, after in ((1.0, np.inf), (1.0, -np.inf),
                              (np.inf, 1.0), (np.inf, -np.inf),
                              (1.0, np.nan), (np.nan, 1.0)):
            with self.subTest(before=before, after=after):
                baseline = dict(self.values, probe=np.array([before]))
                actual = dict(self.values, probe=np.array([after]))
                self.assertEqual(self.compare(actual, baseline), 1)

    def test_same_nonfinite_patterns_pass(self):
        values = dict(self.values, probe=np.array([np.nan, np.inf, -np.inf]))
        self.assertEqual(self.compare(values, values), 0)

    def test_callorder_failure_even_if_baseline_matches(self):
        actual = dict(self.values)
        actual['callorder|callorder|callorder|co_xihm_seq'] = np.array([0.0, 2.0])
        self.assertEqual(self.compare(actual, actual), 1)

    def test_missing_callorder_pair_fails(self):
        actual = dict(self.values)
        del actual['callorder|callorder|callorder|co_xihm_fresh']
        self.assertEqual(self.compare(actual, actual), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
