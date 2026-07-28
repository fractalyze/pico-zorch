# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Prover/verifier duality: an honest proof verifies with the verifier ending
on the prover's exact transcript state; tampering flips `ok` and nothing
else."""

from __future__ import annotations

import dataclasses

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import lax
from zk_dtypes import koalabear_mont as F

from pico_zorch.challenger.challenger import fresh_challenger
from pico_zorch.poseidon2.koalabear import koalabear16_merkle
from pico_zorch.uni_stark.prover import StarkProver
from pico_zorch.uni_stark.testing.fib_air import FibonacciAir, generate_trace_rows
from pico_zorch.uni_stark.types import FriParams, StarkClaim, StarkWitness
from pico_zorch.uni_stark.verifier import StarkVerifier

# 4 queries keep the eager query loop test-sized; the byte-match against the
# reference config's 84 queries lives in fib_e2e_test.
_PARAMS = FriParams(log_blowup=1, num_queries=4, proof_of_work_bits=8)


def _setup():
    air = FibonacciAir()
    trace = generate_trace_rows(0, 1, 8)
    pv = fnp.array([0, 1, 21], dtype=F)
    claim = StarkClaim(air=air, public_values=pv, degree_bits=3)
    _, _, tree = koalabear16_merkle()
    return claim, trace, tree


class VerifierTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.claim, trace, cls.tree = _setup()
        prover = StarkProver(tree=cls.tree, params=_PARAMS)
        cls.result = prover.prove(cls.claim, StarkWitness(trace), fresh_challenger())

    def test_honest_proof_verifies(self) -> None:
        verifier = StarkVerifier(tree=self.tree, params=_PARAMS)
        out = verifier.verify(
            self.claim, self.result.reduction_proof, fresh_challenger()
        )
        self.assertTrue(bool(out.ok))
        # The dual replayed the identical byte stream: same sponge state.
        np.testing.assert_array_equal(
            np.asarray(
                lax.convert_element_type(out.transcript.state.sponge_state, fnp.uint32)
            ),
            np.asarray(
                lax.convert_element_type(
                    self.result.transcript.state.sponge_state, fnp.uint32
                )
            ),
        )

    def test_tampered_opened_value_fails(self) -> None:
        proof = self.result.reduction_proof
        bumped = proof.opening.trace_local + fnp.ones(
            (), proof.opening.trace_local.dtype
        )
        tampered = dataclasses.replace(
            proof, opening=dataclasses.replace(proof.opening, trace_local=bumped)
        )
        verifier = StarkVerifier(tree=self.tree, params=_PARAMS)
        out = verifier.verify(self.claim, tampered, fresh_challenger())
        self.assertFalse(bool(out.ok))

    def test_wrong_public_values_fail(self) -> None:
        wrong = dataclasses.replace(
            self.claim, public_values=fnp.array([0, 1, 22], dtype=F)
        )
        verifier = StarkVerifier(tree=self.tree, params=_PARAMS)
        out = verifier.verify(wrong, self.result.reduction_proof, fresh_challenger())
        self.assertFalse(bool(out.ok))


if __name__ == "__main__":
    absltest.main()
