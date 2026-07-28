# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The uni-stark composite: commit, then chain the claim reductions.

  StarkClaim ──(quotient stage)──▶ TraceOpeningClaim ──(FRI opening)──▶ TrivialClaim

Byte-mirrors Plonky3's uni-stark prove at brevis-network/Plonky3@7fbe1908.
Committing the trace precedes the chain because it is not a transcript
operation; `bind_instance` is single-sourced so the roles cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import frx.numpy as fnp
from frx import Array
from zk_dtypes import koalabear_mont as F

from zorch.commit.merkle import MerkleTree
from zorch.stage import ProveResult, ProverStage, TrivialClaim
from zorch.transcript import DuplexTranscript

from pico_zorch.commit.pcs_commit import commit_pcs
from pico_zorch.uni_stark.fri_stage import FriOpener
from pico_zorch.uni_stark.quotient_stage import QuotientProver
from pico_zorch.uni_stark.types import (
    FriOpeningWitness,
    FriParams,
    QuotientClaim,
    QuotientWitness,
    StarkClaim,
    StarkProof,
    StarkWitness,
)


def bind_instance(
    transcript: DuplexTranscript,
    degree_bits: int,
    trace_root: Array,
    public_values: Array,
) -> DuplexTranscript:
    """The reference's instance observation, which must land before any
    challenge is drawn: log_degree, trace commitment, public values."""
    t = transcript.observe(fnp.array([degree_bits], dtype=F))
    t = t.observe(trace_root)
    return t.observe(public_values)


@dataclass(frozen=True)
class StarkProver(
    ProverStage[StarkClaim, StarkWitness, TrivialClaim, StarkProof, DuplexTranscript]
):
    tree: MerkleTree
    params: FriParams = FriParams()

    def prove(
        self,
        claim: StarkClaim,
        witness: StarkWitness,
        transcript: DuplexTranscript,
    ) -> ProveResult[TrivialClaim, StarkProof, DuplexTranscript]:
        trace_root, trace_data = commit_pcs(
            self.tree, [witness.trace], self.params.log_blowup
        )
        t = bind_instance(transcript, claim.degree_bits, trace_root, claim.public_values)

        quotient = QuotientProver(self.tree, self.params).prove(
            QuotientClaim(claim.air, claim.public_values, claim.degree_bits, trace_root),
            QuotientWitness(witness.trace, trace_data),
            t,
        )

        opening = FriOpener(self.tree, self.params).prove(
            quotient.reduced_claim,
            FriOpeningWitness(
                witness.trace, trace_data, quotient.reduction_proof.data
            ),
            quotient.transcript,
        )

        proof = StarkProof(
            trace_root=trace_root,
            quotient_root=quotient.reduction_proof.quotient_root,
            opening=opening.reduction_proof,
            degree_bits=claim.degree_bits,
        )
        return ProveResult(TrivialClaim(), proof, opening.transcript)
