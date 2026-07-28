# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The quotient stage: `QuotientClaim` → `TraceOpeningClaim`.

The reference's mid-protocol slice (Plonky3 uni-stark prover.rs/verifier.rs
at brevis-network/Plonky3@7fbe1908): sample α, evaluate the α-folded
constraints over the disjoint coset, divide by Z_H, commit the chunked
quotient, sample ζ. The verifier is a pure transcript replay — the algebraic
check on the opened values (the out-of-domain identity) belongs to the
composite, which is the first place those values exist.
"""

from __future__ import annotations

from dataclasses import dataclass

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
        # The reference's get_evaluations_on_domain: the natural-order
        # sub-coset is the re-bit-reversed prefix of the committed LDE.
        trace_on_qd = bit_reverse_rows(
            witness.trace_data.matrices[0][: quotient_domain.size]
        )
        q_vals = quotient_values(
            claim.air,
            claim.public_values,
            trace_domain,
            quotient_domain,
            trace_on_qd,
            alpha,
        )
        chunks = quotient_domain.split_evals(quotient_degree, flatten_to_base(q_vals))
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
