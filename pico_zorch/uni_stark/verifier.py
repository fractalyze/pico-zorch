# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The uni-stark composite verifier — the prover's explicit dual.

Replays the same chain (quotient stage, then FRI opening), ANDing each
stage's verdict, and closes with the out-of-domain identity
folded_constraints(ζ)/Z_H(ζ) == quotient(ζ) — checked here because the
composite is the first place the opened values and the AIR meet. Algebraic
failure lands in `VerifyResult.ok`; structurally impossible proofs raise.
"""

from __future__ import annotations

from dataclasses import dataclass

import frx.numpy as fnp
import numpy as np
from frx import Array, lax
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.commit.merkle import MerkleTree
from zorch.stage import TrivialClaim, VerifierStage, VerifyResult
from zorch.transcript import DuplexTranscript

from pico_zorch.commit.pcs_commit import GENERATOR
from pico_zorch.uni_stark.domain import Coset
from pico_zorch.uni_stark.fri_stage import FriOpeningVerifier
from pico_zorch.uni_stark.prover import bind_instance
from pico_zorch.uni_stark.quotient import log_quotient_degree
from pico_zorch.uni_stark.quotient_stage import QuotientVerifier
from pico_zorch.uni_stark.types import (
    FriParams,
    QuotientClaim,
    QuotientProof,
    StarkClaim,
    StarkProof,
    TraceOpeningClaim,
)


def _ext_monomial(e: int) -> Array:
    limbs = np.zeros(4, dtype=np.uint32)
    limbs[e] = 1
    return lax.bitcast_convert_type(fnp.array(limbs, dtype=F), EF).reshape(())


def _ood_identity(
    claim: StarkClaim, opening: TraceOpeningClaim, proof: StarkProof
) -> Array:
    """The reference verifier's final check: the α-folded constraints at ζ,
    divided by Z_H, equal the chunk recombination (each chunk weighted by the
    other chunk domains' vanishing polynomials, normalized at its first
    point)."""
    air = claim.air
    log_qd = log_quotient_degree(air.constraint_degree)
    quotient_degree = 1 << log_qd
    generator = fnp.array(GENERATOR, dtype=F)
    trace_domain = Coset(claim.degree_bits, fnp.ones((), F))
    quotient_domain = Coset(claim.degree_bits + log_qd, generator)
    qc_domains = quotient_domain.split_domains(quotient_degree)
    zeta = opening.zeta

    zps = []
    for i, dom in enumerate(qc_domains):
        acc = fnp.ones((), EF)
        for j, other in enumerate(qc_domains):
            if j != i:
                acc = acc * other.zp_at_point(zeta)
                acc = acc / other.zp_at_point(dom.shift.astype(EF))
        zps.append(acc)
    quotient = fnp.zeros((), EF)
    for ci in range(quotient_degree):
        for e in range(4):
            quotient = quotient + (
                zps[ci] * _ext_monomial(e) * proof.opening.quotient_chunks[ci, e]
            )

    sels = trace_domain.selectors_at_point(zeta)
    constraints = air.eval(
        proof.opening.trace_local,
        proof.opening.trace_next,
        sels["is_first_row"],
        sels["is_last_row"],
        sels["is_transition"],
        claim.public_values.astype(EF),
    )
    folded = fnp.zeros((), EF)
    for c in constraints:
        folded = folded * opening.alpha + c
    return fnp.array_equal(folded * sels["inv_zeroifier"], quotient)


@dataclass(frozen=True)
class StarkVerifier(
    VerifierStage[StarkClaim, TrivialClaim, StarkProof, DuplexTranscript]
):
    tree: MerkleTree
    params: FriParams = FriParams()

    def verify(
        self,
        claim: StarkClaim,
        reduction_proof: StarkProof,
        transcript: DuplexTranscript,
    ) -> VerifyResult[TrivialClaim, DuplexTranscript]:
        proof = reduction_proof
        t = bind_instance(
            transcript, proof.degree_bits, proof.trace_root, claim.public_values
        )

        quotient = QuotientVerifier(self.params).verify(
            QuotientClaim(
                claim.air, claim.public_values, proof.degree_bits, proof.trace_root
            ),
            QuotientProof(proof.quotient_root),
            t,
        )

        opening = FriOpeningVerifier(
            tree=self.tree,
            width=claim.air.width,
            quotient_degree=1 << log_quotient_degree(claim.air.constraint_degree),
            params=self.params,
        ).verify(quotient.reduced_claim, proof.opening, quotient.transcript)

        ok = quotient.ok & opening.ok
        ok = ok & _ood_identity(claim, quotient.reduced_claim, proof)
        return VerifyResult(TrivialClaim(), opening.transcript, ok)
