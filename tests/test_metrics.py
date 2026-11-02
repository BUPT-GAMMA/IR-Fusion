import unittest

import numpy as np

from utils.metrics import evaluate_regression, hotspot_f1


class MetricsTest(unittest.TestCase):
    def test_f1_uses_each_maps_own_90_percent_maximum(self):
        target = np.array([10.0, 9.5, 3.0, 0.0])
        pred = np.array([8.0, 7.5, 9.2, 0.0])
        self.assertAlmostEqual(hotspot_f1(target, pred, 0.9), 0.0)

    def test_unmodified_mae_and_hotspot_region_mirde(self):
        target = np.array([10.0, 9.5, 2.0, 0.0])
        pred = np.array([9.0, 8.5, 9.2, 0.0])
        metrics = evaluate_regression(target, pred, hotspot_ratio=0.9)
        self.assertAlmostEqual(metrics["mae"], 2.3)
        self.assertAlmostEqual(metrics["f1"], 0.8)
        self.assertAlmostEqual(metrics["mirde"], 1.0)


if __name__ == "__main__":
    unittest.main()
