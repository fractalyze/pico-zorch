# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""KoalaBear^7 arithmetic and the septic curve, against Pico's own values.

`golden/src/septic.rs` links `pico-vm` and emits these vectors with the
reference's arithmetic, so a match here is a match against Pico rather than
against a second reading of its source.
"""

from __future__ import annotations

import json
import pathlib

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import lax
from zk_dtypes import koalabear_mont as F

from pico_zorch.septic import curve
from pico_zorch.septic import extension as ext

_GOLDEN = pathlib.Path(__file__).parent / "testdata" / "golden" / "septic.json"


def _u32(x) -> np.ndarray:
    return np.asarray(lax.convert_element_type(x, fnp.uint32))


def _elem(coeffs) -> "np.ndarray":
    return fnp.array(coeffs, dtype=F)


class SepticExtensionTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.golden = json.loads(_GOLDEN.read_text())

    def test_multiplication_matches_reference(self) -> None:
        for i, case in enumerate(self.golden["products"]):
            a, b = _elem(case["a"]), _elem(case["b"])
            np.testing.assert_array_equal(
                _u32(ext.mul(a, b)), np.array(case["a_times_b"]), err_msg=f"case {i}"
            )
            np.testing.assert_array_equal(
                _u32(ext.square(a)), np.array(case["a_squared"]), err_msg=f"case {i}"
            )
            np.testing.assert_array_equal(
                _u32(ext.cube(a)), np.array(case["a_cubed"]), err_msg=f"case {i}"
            )

    def test_multiplication_batches(self) -> None:
        """The leading axes are free, so a whole chip's points multiply at
        once — and the batched result must equal the elementwise one."""
        cases = self.golden["products"]
        a = fnp.stack([_elem(c["a"]) for c in cases])
        b = fnp.stack([_elem(c["b"]) for c in cases])
        np.testing.assert_array_equal(
            _u32(ext.mul(a, b)),
            np.array([c["a_times_b"] for c in cases]),
        )

    def test_inverse_round_trips(self) -> None:
        a = _elem(self.golden["products"][0]["a"])
        np.testing.assert_array_equal(
            _u32(ext.mul(a, curve.inverse(a))), _u32(ext.one())
        )


class SepticCurveTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.golden = json.loads(_GOLDEN.read_text())

    def _point(self, name) -> tuple:
        p = self.golden[name]
        return _elem(p["x"]), _elem(p["y"])

    def test_curve_formula_matches_reference(self) -> None:
        for i, case in enumerate(self.golden["curve_formulas"]):
            np.testing.assert_array_equal(
                _u32(curve.curve_formula(_elem(case["x"]))),
                np.array(case["curve_formula"]),
                err_msg=f"case {i}",
            )

    def test_reference_points_are_on_the_curve(self) -> None:
        """An independent check of the field arithmetic: if `mul` were wrong,
        Pico's own published points would not satisfy the equation."""
        for name in ("starting_digest", "double_starting", "starting_plus_double"):
            x, y = self._point(name)
            self.assertTrue(bool(curve.is_on_curve(x, y)), msg=name)

    def test_double_matches_reference(self) -> None:
        got = curve.double(self._point("starting_digest"))
        want = self._point("double_starting")
        np.testing.assert_array_equal(_u32(got[0]), _u32(want[0]))
        np.testing.assert_array_equal(_u32(got[1]), _u32(want[1]))

    def test_add_matches_reference(self) -> None:
        got = curve.add_incomplete(
            self._point("starting_digest"), self._point("double_starting")
        )
        want = self._point("starting_plus_double")
        np.testing.assert_array_equal(_u32(got[0]), _u32(want[0]))
        np.testing.assert_array_equal(_u32(got[1]), _u32(want[1]))

    def test_negate_matches_reference(self) -> None:
        got = curve.neg(*self._point("starting_digest"))
        want = self._point("negated_starting")
        np.testing.assert_array_equal(_u32(got[0]), _u32(want[0]))
        np.testing.assert_array_equal(_u32(got[1]), _u32(want[1]))


if __name__ == "__main__":
    absltest.main()
