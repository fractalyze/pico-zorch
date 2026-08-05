# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The batched opening's public half, against a multi-round reference open.

`golden`'s `emit_pcs_open` runs two rounds through the reference `pcs.open`
with matrices of differing heights inside a round and differing point counts
per matrix — the axes the uni-stark fixture holds constant.
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

from pico_zorch.pcs.open import Opening, opened_values, reduced_openings
from pico_zorch.pcs.two_adic_fri import TwoAdicFriPcs
from pico_zorch.commit.mmcs import MerkleTreeMmcs
from pico_zorch.poseidon2.koalabear import koalabear16_merkle

_GOLDEN = pathlib.Path(__file__).parent / "testdata" / "golden" / "pcs_open.json"


def _u32(x) -> np.ndarray:
    return np.asarray(lax.convert_element_type(lax.bitcast_convert_type(x, F), fnp.uint32))


def _ext(coeffs) -> "np.ndarray":
    return lax.bitcast_convert_type(fnp.array(coeffs, dtype=F), EF)


class OpenedValuesTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.golden = json.loads(_GOLDEN.read_text())
        cls.log_blowup = cls.golden["log_blowup"]
        _, compressor, tree = koalabear16_merkle()
        cls.pcs = TwoAdicFriPcs(
            MerkleTreeMmcs(tree, compressor), log_blowup=cls.log_blowup
        )

    def _rounds(self) -> list[list[Opening]]:
        out = []
        for rnd in self.golden["rounds"]:
            openings = []
            for mat, pts in zip(rnd["matrices"], rnd["points"]):
                trace = fnp.array(mat["values"], dtype=F)
                openings.append(
                    Opening(
                        lde=self.pcs.lde(trace),
                        trace=trace,
                        points=[_ext(z) for z in pts],
                    )
                )
            out.append(openings)
        return out

    def test_opened_values_match_reference(self) -> None:
        for r, (rnd, openings) in enumerate(
            zip(self.golden["rounds"], self._rounds())
        ):
            want_round = self.golden["opened_values"][r]
            for m, opening in enumerate(openings):
                got = opened_values(opening.trace, opening.points)
                for pt, (g, w) in enumerate(zip(got, want_round[m])):
                    np.testing.assert_array_equal(
                        _u32(g),
                        np.array(w),
                        err_msg=f"round {r} matrix {m} point {pt}",
                    )

    def test_commit_matches_the_reference_per_round(self) -> None:
        """Each round's commitment is pinned too, so a reduction mismatch
        cannot be blamed on having committed the wrong matrices."""
        for rnd in self.golden["rounds"]:
            traces = [fnp.array(m["values"], dtype=F) for m in rnd["matrices"]]
            root, _ = self.pcs.commit(traces)
            np.testing.assert_array_equal(
                np.asarray(lax.convert_element_type(root, fnp.uint32)),
                np.array(rnd["commit"]),
            )


class ReducedOpeningsTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.golden = json.loads(_GOLDEN.read_text())
        cls.log_blowup = cls.golden["log_blowup"]
        _, compressor, tree = koalabear16_merkle()
        cls.pcs = TwoAdicFriPcs(
            MerkleTreeMmcs(tree, compressor), log_blowup=cls.log_blowup
        )
        cls.alpha = _ext([7, 11, 13, 17])

    def _rounds(self) -> list[list[Opening]]:
        out = []
        for rnd in self.golden["rounds"]:
            openings = []
            for mat, pts in zip(rnd["matrices"], rnd["points"]):
                trace = fnp.array(mat["values"], dtype=F)
                openings.append(
                    Opening(
                        lde=self.pcs.lde(trace),
                        trace=trace,
                        points=[_ext(z) for z in pts],
                    )
                )
            out.append(openings)
        return out

    def test_one_accumulator_per_distinct_height(self) -> None:
        acc = reduced_openings(self._rounds(), self.alpha, self.log_blowup)
        committed = {
            m["height"] << self.log_blowup
            for rnd in self.golden["rounds"]
            for m in rnd["matrices"]
        }
        self.assertEqual(set(acc), committed)
        for height, column in acc.items():
            self.assertEqual(column.shape, (height,))

    def test_reduction_is_a_low_degree_codeword(self) -> None:
        """The property FRI is about to assert, checked independently of the
        reference.

        `(p(z) - p(X)) / (z - X)` is a polynomial in X of degree `deg(p) - 1`,
        so the reduction is a codeword of rate `1/blowup` — its coefficients
        above the trace height must vanish. This catches a wrong numerator
        sign, a wrong denominator, or a mis-ordered alpha run, none of which
        a shape check would notice, and it holds for reasons the reference
        cannot be wrong about."""
        opening = self._rounds()[0][0]
        acc = reduced_openings([[opening]], self.alpha, self.log_blowup)
        (height, column), = acc.items()

        # Undo the committed row order, then interpolate over the plain
        # subgroup: on the coset `g*H` this recovers the coefficients of
        # p~(y) = p(g*y), which has the same degree as p.
        natural = lax.bit_reverse(column, dimensions=(0,))
        coeffs = lax.ntt(natural, ntt_type="INTT", ntt_length=height)
        trace_height = opening.trace.shape[0]

        tail = _u32(coeffs[trace_height:])
        # A blowup of 1 would leave nothing to check; fail loudly rather than
        # let the assertion below pass over an empty array.
        self.assertGreater(tail.size, 0, "no tail to check — is log_blowup 0?")
        np.testing.assert_array_equal(
            tail,
            np.zeros_like(tail),
            err_msg=(
                f"reduction is not degree < {trace_height}; the quotient "
                "(p(z)-p(X))/(z-X) must be a codeword or FRI would reject it"
            ),
        )

    def test_offset_is_per_height_not_global(self) -> None:
        """Reordering matrices across *different* heights must change the
        result, because each height carries its own alpha counter. If the
        counter were global this reordering would be a no-op."""
        rounds = self._rounds()
        flat = [o for rnd in rounds for o in rnd]
        heights = [o.lde.shape[0] for o in flat]
        self.assertGreater(len(set(heights)), 1, "fixture must mix heights")

        forward = reduced_openings([flat], self.alpha, self.log_blowup)
        reversed_ = reduced_openings([flat[::-1]], self.alpha, self.log_blowup)
        differs = any(
            not np.array_equal(_u32(forward[h]), _u32(reversed_[h])) for h in forward
        )
        self.assertTrue(differs, "matrix order must affect the reduction")


if __name__ == "__main__":
    absltest.main()
