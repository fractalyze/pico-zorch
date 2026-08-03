# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""End-to-end byte-match of the uni-stark prover against the reference proof.

golden/ proves the same Fibonacci instance through the fork's own
`p3_uni_stark::prove` under Pico's KoalaBearPoseidon2 config; every wire
field is compared here, so this test is what licenses any claim that the
pipeline reproduces Pico's prover.
"""

from __future__ import annotations

import json
import pathlib

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import lax
from zk_dtypes import koalabear_mont as F

from pico_zorch.challenger.challenger import fresh_challenger
from pico_zorch.poseidon2.koalabear import koalabear16_merkle
from pico_zorch.uni_stark.fri import FriOpener, query_opening
from pico_zorch.uni_stark.prover import StarkProver
from pico_zorch.uni_stark.testing.fib_air import FibonacciAir, generate_trace_rows
from pico_zorch.uni_stark.types import StarkClaim, StarkWitness

_GOLDEN = pathlib.Path(__file__).parent / "testdata" / "golden" / "fib_prove.json"


def _canonical(x) -> np.ndarray:
    return np.asarray(lax.convert_element_type(x, fnp.uint32))


def _ext(value) -> list[int]:
    """A golden extension element {"value": [c0..c3]} as canonical limbs."""
    return value["value"]


class FibE2eTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.golden = json.loads(_GOLDEN.read_text())
        air = FibonacciAir()
        trace = generate_trace_rows(0, 1, 8)
        pv = fnp.array(cls.golden["public_values"], dtype=F)
        claim = StarkClaim(air=air, public_values=pv, degree_bits=3)
        _, _, tree = koalabear16_merkle()
        prover = StarkProver(FriOpener(tree))
        cls.result = prover.prove(claim, StarkWitness(trace), fresh_challenger())

    def test_commitments(self) -> None:
        proof = self.result.reduction_proof
        want = self.golden["proof"]["commitments"]
        np.testing.assert_array_equal(
            _canonical(proof.trace_root), np.array(want["trace"]["value"])
        )
        np.testing.assert_array_equal(
            _canonical(proof.quotient_root),
            np.array(want["quotient_chunks"]["value"]),
        )

    def test_opened_values(self) -> None:
        proof = self.result.reduction_proof
        want = self.golden["proof"]["opened_values"]
        got_local = _canonical(lax.bitcast_convert_type(proof.opening.trace_local, F))
        got_next = _canonical(lax.bitcast_convert_type(proof.opening.trace_next, F))
        np.testing.assert_array_equal(
            got_local, np.array([_ext(v) for v in want["trace_local"]])
        )
        np.testing.assert_array_equal(
            got_next, np.array([_ext(v) for v in want["trace_next"]])
        )
        got_chunks = _canonical(lax.bitcast_convert_type(proof.opening.quotient_chunks, F))
        want_chunks = np.array(
            [[_ext(v) for v in chunk] for chunk in want["quotient_chunks"]]
        )
        np.testing.assert_array_equal(got_chunks, want_chunks)

    def test_fri_commit_phase_and_final(self) -> None:
        proof = self.result.reduction_proof
        want = self.golden["proof"]["opening_proof"]
        self.assertEqual(
            len(proof.opening.fri.commit_phase_roots), len(want["commit_phase_commits"])
        )
        for got_root, want_root in zip(
            proof.opening.fri.commit_phase_roots, want["commit_phase_commits"]
        ):
            np.testing.assert_array_equal(
                _canonical(got_root), np.array(want_root["value"])
            )
        np.testing.assert_array_equal(
            _canonical(lax.bitcast_convert_type(proof.opening.fri.final_poly, F)),
            np.array(_ext(want["final_poly"])),
        )
        self.assertEqual(
            int(_canonical(proof.opening.fri.pow_witness)[()]), want["pow_witness"]
        )

    def test_query_openings(self) -> None:
        proof = self.result.reduction_proof
        # The golden stores checkpoint query proofs (first few + the last,
        # which pins the tail of the index-sampling stream); the prover must
        # still have produced all of them.
        opening_proof = self.golden["proof"]["opening_proof"]
        want_queries = opening_proof["query_proofs"]
        stored_indices = opening_proof["stored_query_indices"]
        self.assertEqual(
            proof.opening.fri.trace_openings.row.shape[0],
            self.golden["fri_config"]["num_queries"],
        )
        for q, want_q in zip(stored_indices, want_queries):
            got_rounds = [
                query_opening(proof.opening.fri.trace_openings, q),
                query_opening(proof.opening.fri.quotient_openings, q),
            ]
            want_rounds = want_q["input_proof"]
            for r, (got, want_r) in enumerate(zip(got_rounds, want_rounds)):
                # opened_values: one matrix per round -> [row].
                want_row = np.array(want_r["opened_values"][0])
                np.testing.assert_array_equal(
                    _canonical(got.row), want_row, err_msg=f"query {q} round {r} row"
                )
                want_path = np.array(want_r["opening_proof"])
                got_path = np.stack([_canonical(p) for p in got.path])
                np.testing.assert_array_equal(
                    got_path, want_path, err_msg=f"query {q} round {r} path"
                )
            for layer, (batched, want_step) in enumerate(
                zip(proof.opening.fri.commit_phase_openings, want_q["commit_phase_openings"])
            ):
                got_open = query_opening(batched, q)
                row = _canonical(got_open.row).reshape(2, 4)
                want_sib = np.array(_ext(want_step["sibling_value"]))
                # The reference stores one sibling where the pair row holds
                # both halves, so which slot it lands in follows the query.
                self.assertTrue(
                    (row[0] == want_sib).all() or (row[1] == want_sib).all(),
                    msg=f"query {q} layer {layer} sibling",
                )
                want_path = np.array(want_step["opening_proof"])
                got_path = np.stack([_canonical(p) for p in got_open.path])
                np.testing.assert_array_equal(
                    got_path, want_path, err_msg=f"query {q} layer {layer} path"
                )


if __name__ == "__main__":
    absltest.main()
