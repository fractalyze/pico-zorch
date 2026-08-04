# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The LogUp permutation trace byte-matches Pico's generator.

`golden/src/septic.rs` runs Pico's own `generate_permutation_trace` over
interactions that select main columns directly, so the fixture pins the LogUp
arithmetic — RLC denominator, the sign flip on receives, batching into columns,
the running sum — without dragging in a chip's expression language.
"""

from __future__ import annotations

import json
import pathlib

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import lax
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from pico_zorch.permutation.logup import Lookup, permutation_trace

_GOLDEN = pathlib.Path(__file__).parent / "testdata" / "golden" / "logup.json"


def _u32(x) -> np.ndarray:
    """Extension elements as their four canonical base coefficients."""
    return np.asarray(lax.convert_element_type(lax.bitcast_convert_type(x, F), fnp.uint32))


def _ext(coeffs) -> "np.ndarray":
    return lax.bitcast_convert_type(fnp.array(coeffs, dtype=F), EF)


class LogUpTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.golden = json.loads(_GOLDEN.read_text())
        cls.main = fnp.array(cls.golden["main"], dtype=F)
        cls.alpha = _ext(cls.golden["alpha"])
        cls.beta = _ext(cls.golden["beta"])

        # Sends first, then receives — the reference's order, which the column
        # batching depends on.
        cls.lookups = [
            Lookup(kind=s["kind"], is_send=True) for s in cls.golden["sends"]
        ] + [Lookup(kind=r["kind"], is_send=False) for r in cls.golden["receives"]]
        specs = cls.golden["sends"] + cls.golden["receives"]
        cls.values = [cls.main[:, spec["value_cols"]] for spec in specs]
        cls.mults = [cls.main[:, spec["mult_col"]] for spec in specs]

    def _run(self):
        return permutation_trace(
            self.lookups,
            self.values,
            self.mults,
            self.alpha,
            self.beta,
            self.golden["batch_size"],
        )

    def test_trace_matches_reference(self) -> None:
        trace, _ = self._run()
        np.testing.assert_array_equal(
            _u32(trace), np.array(self.golden["permutation"])
        )

    def test_regional_cumulative_sum_matches_reference(self) -> None:
        _, regional = self._run()
        np.testing.assert_array_equal(
            _u32(regional), np.array(self.golden["regional_cumulative_sum"])
        )

    def test_width_follows_the_batch_size(self) -> None:
        """Interactions batch `batch_size` to a column, plus the running sum."""
        trace, _ = self._run()
        batch = self.golden["batch_size"]
        expected = -(-len(self.lookups) // batch) + 1
        self.assertEqual(trace.shape[1], expected)

    def test_running_column_is_the_prefix_sum(self) -> None:
        """The last column accumulates the others down the rows, and its final
        entry is the regional sum the transcript observes."""
        trace, regional = self._run()
        got = _u32(trace[:, -1])
        want = _u32(fnp.cumsum(trace[:, :-1].sum(axis=-1), axis=0))
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(got[-1], _u32(regional))

    def test_receives_flip_the_sign(self) -> None:
        """A receive enters negative — that cancellation is what makes a
        matched send/receive pair sum to zero. Flipping every lookup to a send
        must therefore change the trace."""
        all_sends = [Lookup(kind=lk.kind, is_send=True) for lk in self.lookups]
        other, _ = permutation_trace(
            all_sends,
            self.values,
            self.mults,
            self.alpha,
            self.beta,
            self.golden["batch_size"],
        )
        trace, _ = self._run()
        self.assertFalse(np.array_equal(_u32(trace), _u32(other)))


if __name__ == "__main__":
    absltest.main()
