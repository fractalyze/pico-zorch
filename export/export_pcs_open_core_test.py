# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The exported opening core reproduces the reference proof.

`//pico_zorch/pcs:open_test` pins the *eager* composite, where each stage
dispatches its own jit zone. The core traces the whole opening as one program,
so the zones nest instead — a different path through the same code, and the one
a `Pcs::open` swap would actually run.

Tracing the opening as one program is also what surfaces constants that were
only accidentally concrete: a coset shift or a round-constant table built with
`fnp.array` inside an enclosing trace becomes a tracer, and the failure appears
far from its cause. This test is what keeps those forced to compile time.
"""

from __future__ import annotations

import json
import pathlib

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import lax
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from export.export_pcs_open_core import core_fn, parse_rounds
from pico_zorch.challenger.challenger import PicoTranscript
from pico_zorch.commit.mmcs import MerkleTreeMmcs
from pico_zorch.pcs.two_adic_fri import TwoAdicFriPcs
from pico_zorch.poseidon2.koalabear import koalabear16_merkle

_GOLDEN = (
    pathlib.Path(__file__).parent.parent
    / "pico_zorch"
    / "pcs"
    / "testdata"
    / "golden"
    / "pcs_open.json"
)

# The fixture's layout: two rounds, mixed heights, uneven point counts.
_ROUNDS = "16x2:2,8x1:1;16x4:2,4x3:1"
NUM_QUERIES = 84
PROOF_OF_WORK_BITS = 16


def _u32(x) -> np.ndarray:
    return np.asarray(lax.convert_element_type(lax.bitcast_convert_type(x, F), fnp.uint32))


def _ext(coeffs):
    return lax.bitcast_convert_type(fnp.array(coeffs, dtype=F), EF)


class ExportedOpenCoreTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.golden = json.loads(_GOLDEN.read_text())
        cls.spec = parse_rounds(_ROUNDS)

    def _run(self):
        fn = core_fn(
            self.spec,
            self.golden["log_blowup"],
            NUM_QUERIES,
            PROOF_OF_WORK_BITS,
        )
        _, compressor, tree = koalabear16_merkle()
        pcs = TwoAdicFriPcs(
            MerkleTreeMmcs(tree, compressor), log_blowup=self.golden["log_blowup"]
        )
        ldes, points = [], []
        for rnd in self.golden["rounds"]:
            for mat, pts in zip(rnd["matrices"], rnd["points"]):
                ldes.append(pcs.lde(fnp.array(mat["values"], dtype=F)))
                points.append([_ext(z) for z in pts])
        return frx.jit(fn)(PicoTranscript.new().state, ldes, points)

    def test_traced_core_matches_reference(self) -> None:
        _, roots, final_poly, witness, _, _, _ = self._run()

        want_roots = self.golden["proof"]["commit_phase_commits"]
        self.assertEqual(len(roots), len(want_roots))
        for i, (got, w) in enumerate(zip(roots, want_roots)):
            np.testing.assert_array_equal(
                np.asarray(lax.convert_element_type(got, fnp.uint32)),
                np.array(w["value"]),
                err_msg=f"commit phase layer {i}",
            )
        np.testing.assert_array_equal(
            _u32(final_poly), np.array(self.golden["proof"]["final_poly"]["value"])
        )
        self.assertEqual(
            int(np.asarray(lax.convert_element_type(witness, fnp.uint32))),
            self.golden["proof"]["pow_witness"],
        )

    def test_transcript_state_comes_back_out(self) -> None:
        """The machine prover keeps opening after the PCS returns, so the core
        has to hand the sponge back rather than swallow it."""
        *_, state = self._run()
        fresh = PicoTranscript.new().state
        self.assertEqual(state.sponge_state.shape, fresh.sponge_state.shape)
        self.assertFalse(
            np.array_equal(
                np.asarray(lax.convert_element_type(state.sponge_state, fnp.uint32)),
                np.asarray(lax.convert_element_type(fresh.sponge_state, fnp.uint32)),
            ),
            "the sponge must have advanced",
        )


if __name__ == "__main__":
    absltest.main()
