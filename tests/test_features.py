import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.features import build_spice_maps, load_named_node_values


SPICE = """\
R0 n1_m1_0_0 n1_m1_10_0 2
R1 n1_m1_10_0 n1_m4_10_0 4
R2 n1_m4_10_0 n1_m4_10_10 5
I0 n1_m1_0_0 0 0.1
V0 n1_m4_10_10 0 1.0
.op
.end
"""

SOLUTION = """\
node voltage_V ir_drop_V
n1_m1_0_0 1.0 0.0
n1_m1_10_0 0.8 0.2
n1_m4_10_0 0.4 0.6
n1_m4_10_10 0.3 0.7
"""


class FeatureMapTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.spice = root / "toy.sp"
        self.solution = root / "toy.txt"
        self.spice.write_text(SPICE)
        self.solution.write_text(SOLUTION)

    def tearDown(self):
        self.tmp.cleanup()

    def test_named_voltage_column(self):
        values = load_named_node_values(self.solution, "voltage_V")
        self.assertAlmostEqual(values["n1_m4_10_0"], 0.4)

    def test_branch_current_populates_upper_layer(self):
        maps, names, _ = build_spice_maps(
            self.spice,
            image_shape=(3, 3),
            layers=[1, 4],
            include_shortest_path=False,
            node_voltage_path=self.solution,
        )
        current_m1 = maps[names.index("current_m1")]
        current_m4 = maps[names.index("current_m4")]

        np.testing.assert_allclose(current_m1[:, 0], [0.1, 0.1, 0.2], rtol=1e-6)
        np.testing.assert_allclose(current_m4[2, :], [0.12, 0.02, 0.02], rtol=1e-6)

    def test_bottom_csv_replaces_branch_map_only_on_bottom_layer(self):
        bottom = np.full((3, 3), 7.0, dtype=np.float32)
        maps, names, _ = build_spice_maps(
            self.spice,
            image_shape=(3, 3),
            layers=[1, 4],
            include_shortest_path=False,
            bottom_current_map=bottom,
            bottom_current_layer=1,
            node_voltage_path=self.solution,
        )
        np.testing.assert_array_equal(maps[names.index("current_m1")], bottom)
        self.assertGreater(np.count_nonzero(maps[names.index("current_m4")]), 0)


if __name__ == "__main__":
    unittest.main()
