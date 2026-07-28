# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The quotient stage: `QuotientClaim` → `TraceOpeningClaim`.

The reference's mid-protocol slice (Plonky3 uni-stark prover.rs/verifier.rs
at brevis-network/Plonky3@7fbe1908): sample α, evaluate the α-folded
constraints over the disjoint coset, divide by Z_H, commit the chunked
quotient, sample ζ. The verifier half is a pure transcript replay: the
values its algebraic check would need do not exist until the opening stage
has run, so that check belongs to the composite.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import frx
import frx.numpy as fnp
from frx import Array
from zk_dtypes import koalabear_mont as F

from zorch.commit.merkle import MerkleTree
from zorch.stage import ProveResult, ProverStage, VerifierStage, VerifyResult
from zorch.transcript import DuplexTranscript

from pico_zorch.challenger.challenger import sample_ext
from pico_zorch.commit.pcs_commit import GENERATOR, bit_reverse_rows, commit_pcs
from pico_zorch.uni_stark.domain import Coset
from pico_zorch.uni_stark.quotient import (
    flatten_to_base,
    log_quotient_degree,
    quotient_values,
)
from pico_zorch.uni_stark.types import (
    FriParams,
    QuotientClaim,
    QuotientData,
    QuotientProof,
    QuotientWitness,
    TraceOpeningClaim,
)


@partial(frx.jit, static_argnames=("air", "log_n", "log_qd"))
def _quotient_flat(air, log_n, log_qd, lde, public_values, alpha):
    """Quotient evaluations as base columns.

    The trace arrives as the committed (bit-reversed) LDE, whose first
    `quotient_domain.size` rows re-reverse into exactly the natural-order
    sub-coset — the reference's `get_evaluations_on_domain`."""
    trace_domain = Coset(log_n, fnp.ones((), F))
    quotient_domain = Coset(log_n + log_qd, fnp.array(GENERATOR, dtype=F))
    trace_on_qd = bit_reverse_rows(lde[: quotient_domain.size])
    return flatten_to_base(
        quotient_values(
            air, public_values, trace_domain, quotient_domain, trace_on_qd, alpha
        )
    )


@dataclass(frozen=True)
class QuotientProver(
    ProverStage[
        QuotientClaim, QuotientWitness, TraceOpeningClaim, QuotientProof, DuplexTranscript
    ]
):
    tree: MerkleTree
    params: FriParams = FriParams()

    def prove(
        self,
        claim: QuotientClaim,
        witness: QuotientWitness,
        transcript: DuplexTranscript,
    ) -> ProveResult[TraceOpeningClaim, QuotientProof, DuplexTranscript]:
        log_n = claim.degree_bits
        generator = fnp.array(GENERATOR, dtype=F)
        trace_domain = Coset(log_n, fnp.ones((), F))
        log_qd = log_quotient_degree(claim.air.constraint_degree)
        quotient_degree = 1 << log_qd

        t, alpha = sample_ext(transcript)

        quotient_domain = Coset(log_n + log_qd, generator)
        q_flat = _quotient_flat(
            claim.air,
            log_n,
            log_qd,
            witness.trace_data.matrices[0],
            claim.public_values,
            alpha,
        )
        chunks = quotient_domain.split_evals(quotient_degree, q_flat)
        qc_domains = quotient_domain.split_domains(quotient_degree)
        quotient_root, quotient_data = commit_pcs(
            self.tree,
            chunks,
            self.params.log_blowup,
            shifts=[generator / d.shift for d in qc_domains],
        )

        t = t.observe(quotient_root)
        t, zeta = sample_ext(t)

        reduced = TraceOpeningClaim(
            trace_root=claim.trace_root,
            quotient_root=quotient_root,
            alpha=alpha,
            zeta=zeta,
            zeta_next=trace_domain.next_point(zeta),
            degree_bits=log_n,
        )
        proof = QuotientProof(
            quotient_root, QuotientData(chunks, qc_domains, quotient_data)
        )
        return ProveResult(reduced, proof, t)


@dataclass(frozen=True)
class QuotientVerifier(
    VerifierStage[QuotientClaim, TraceOpeningClaim, QuotientProof, DuplexTranscript]
):
    params: FriParams = FriParams()

    def verify(
        self,
        claim: QuotientClaim,
        reduction_proof: QuotientProof,
        transcript: DuplexTranscript,
    ) -> VerifyResult[TraceOpeningClaim, DuplexTranscript]:
        log_n = claim.degree_bits
        trace_domain = Coset(log_n, fnp.ones((), F))

        t, alpha = sample_ext(transcript)
        t = t.observe(reduction_proof.quotient_root)
        t, zeta = sample_ext(t)

        reduced = TraceOpeningClaim(
            trace_root=claim.trace_root,
            quotient_root=reduction_proof.quotient_root,
            alpha=alpha,
            zeta=zeta,
            zeta_next=trace_domain.next_point(zeta),
            degree_bits=log_n,
        )
        return VerifyResult(reduced, t, fnp.array(True))
