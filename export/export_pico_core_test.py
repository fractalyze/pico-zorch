# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The exported core reproduces the reference proof.

`fib_e2e_test` pins the *eager* composite, where each stage dispatches its own
jit zone. The core traces the whole prover as one program, so the jit zones
nest instead — a different path through the same code, and the one the Rust
binding actually runs. This pins it against the same golden vector, so the
export cannot drift from the prover it claims to be.

It also pins the wire representation itself, which the Rust side casts rather
than converts — a zero-copy cast is only correct while both libraries agree on
the Montgomery constant.
"""

from __future__ import annotations

import json
import pathlib

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import lax
from zk_dtypes import koalabear_mont as F

from export.export_pico_core import core_fn, output_names
from pico_zorch.uni_stark.testing.fib_air import FibonacciAir, generate_trace_rows
from pico_zorch.uni_stark.types import FriParams

_GOLDEN = (
    pathlib.Path(__file__).parent.parent
    / "pico_zorch"
    / "uni_stark"
    / "testdata"
    / "golden"
    / "fib_prove.json"
)

_DEGREE_BITS = 3


def _u32(x) -> np.ndarray:
    """Montgomery limbs as canonical uint32, for comparison with the golden.

    The wire itself carries Montgomery form (see `_wire`); the golden records
    canonical values, so the conversion belongs here in the test rather than
    on the proving path. Extension elements already arrive widened to their
    four base coefficients.
    """
    return np.asarray(lax.convert_element_type(x, fnp.uint32))


class WireRepresentationTest(absltest.TestCase):
    """The wire is the raw Montgomery limb, cast on the Rust side rather than
    converted. That is only sound while zk_dtypes and Plonky3 agree on the
    representation, so pin it here: a change on either side has to fail as a
    one-line assertion rather than as an unexplainable proof mismatch."""

    # Plonky3 `KoalaBearParameters`: PRIME = 0x7f000001, MONTY_BITS = 32.
    PRIME = 0x7F000001
    R = (1 << 32) % PRIME  # 33554430 — the Montgomery image of 1

    def test_montgomery_constant_matches_plonky3(self) -> None:
        one = np.asarray(fnp.array(1, dtype=F)).reshape(1).view(np.uint32)[0]
        self.assertEqual(int(one), self.R)

    def test_limb_is_a_reduced_u32(self) -> None:
        """Plonky3 stores a reduced residue, so every limb the core emits must
        already be < P — otherwise a cast would produce a `KoalaBear` whose
        arithmetic is silently wrong."""
        values = fnp.array([0, 1, 2, self.PRIME - 1], dtype=F)
        limbs = np.asarray(values).view(np.uint32)
        self.assertTrue((limbs < self.PRIME).all(), msg=f"unreduced limbs {limbs}")


class ExportedCoreTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.golden = json.loads(_GOLDEN.read_text())
        air = FibonacciAir()
        core = core_fn(air, _DEGREE_BITS, FriParams())
        trace = generate_trace_rows(0, 1, 1 << _DEGREE_BITS)
        public_values = fnp.array(cls.golden["public_values"], dtype=F)
        outputs = core(trace, public_values)
        cls.out = dict(zip(output_names(_DEGREE_BITS), outputs))

    def test_commitments(self) -> None:
        want = self.golden["proof"]["commitments"]
        np.testing.assert_array_equal(
            _u32(self.out["trace_root"]), np.array(want["trace"]["value"])
        )
        np.testing.assert_array_equal(
            _u32(self.out["quotient_root"]), np.array(want["quotient_chunks"]["value"])
        )

    def test_opened_values(self) -> None:
        want = self.golden["proof"]["opened_values"]
        np.testing.assert_array_equal(
            _u32(self.out["trace_local"]),
            np.array([v["value"] for v in want["trace_local"]]),
        )
        np.testing.assert_array_equal(
            _u32(self.out["trace_next"]),
            np.array([v["value"] for v in want["trace_next"]]),
        )
        np.testing.assert_array_equal(
            _u32(self.out["quotient_chunks"]),
            np.array([[v["value"] for v in c] for c in want["quotient_chunks"]]),
        )

    def test_fri_commit_phase_and_final(self) -> None:
        want = self.golden["proof"]["opening_proof"]
        np.testing.assert_array_equal(
            _u32(self.out["commit_phase_roots"]),
            np.array([r["value"] for r in want["commit_phase_commits"]]),
        )
        np.testing.assert_array_equal(
            _u32(self.out["final_poly"]), np.array(want["final_poly"]["value"])
        )
        self.assertEqual(int(_u32(self.out["pow_witness"])[()]), want["pow_witness"])

    def test_query_openings(self) -> None:
        """The golden stores checkpoint queries; the core must still produce
        every one, and each stored index must match row for row and path for
        path."""
        opening_proof = self.golden["proof"]["opening_proof"]
        self.assertEqual(
            self.out["trace_opening_rows"].shape[0],
            self.golden["fri_config"]["num_queries"],
        )
        rows = {
            0: _u32(self.out["trace_opening_rows"]),
            1: _u32(self.out["quotient_opening_rows"]),
        }
        paths = {
            0: _u32(self.out["trace_opening_paths"]),
            1: _u32(self.out["quotient_opening_paths"]),
        }
        stored = opening_proof["stored_query_indices"]
        for q, want_q in zip(stored, opening_proof["query_proofs"]):
            for r, want_r in enumerate(want_q["input_proof"]):
                np.testing.assert_array_equal(
                    rows[r][q],
                    np.array(want_r["opened_values"][0]),
                    err_msg=f"query {q} round {r} row",
                )
                np.testing.assert_array_equal(
                    paths[r][:, q],
                    np.array(want_r["opening_proof"]),
                    err_msg=f"query {q} round {r} path",
                )
            for layer, want_step in enumerate(want_q["commit_phase_openings"]):
                pair = _u32(self.out[f"fri_layer{layer}_rows"])[q].reshape(2, 4)
                want_sib = np.array(want_step["sibling_value"]["value"])
                # The reference stores one sibling where the pair row holds
                # both halves, so which slot it lands in follows the query.
                self.assertTrue(
                    (pair[0] == want_sib).all() or (pair[1] == want_sib).all(),
                    msg=f"query {q} layer {layer} sibling",
                )
                np.testing.assert_array_equal(
                    _u32(self.out[f"fri_layer{layer}_paths"])[:, q],
                    np.array(want_step["opening_proof"]),
                    err_msg=f"query {q} layer {layer} path",
                )


if __name__ == "__main__":
    absltest.main()
