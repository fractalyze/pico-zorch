# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The uni-stark composite verifier — the prover's explicit dual.

Replays the same chain, ANDing each stage's verdict, and closes the argument
with the STARK identity at ζ: the α-folded constraints, evaluated on the
opened trace values, must equal Q(ζ)·Z_H(ζ). The opening stage has already
established that those values are the honest evaluations of committed
low-degree polynomials, so this last equation is what turns them back into
a statement about the AIR.

The check lives here rather than in a stage because it is the first point
where the opened values and the AIR meet. Algebraic failure lands in
`VerifyResult.ok`; a structurally impossible proof raises instead, since no
challenge can rescue it.
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
    """The α-folded constraints at ζ, divided by Z_H, must equal Q(ζ).

    Q was committed as `quotient_degree` chunks living on disjoint cosets,
    so recovering Q(ζ) is a Lagrange interpolation across those cosets:
    each chunk is weighted by the other cosets' vanishing polynomials at ζ,
    normalized at its own first point so the weights are 1 on its coset."""
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
