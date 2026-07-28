# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Claims, witnesses and proof wire types for the uni-stark stages.

The claim chain mirrors the reference protocol's two reductions:
`QuotientClaim` (the AIR holds on the committed trace) reduces to
`TraceOpeningClaim` (the committed polynomials open to consistent values at
ζ and ζ·g), which the FRI opening reduces to `TrivialClaim`. Reduced claims
are execution values, never serialized; field arrays carry raw Montgomery
u32, canonical form appears only on comparison surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from frx import Array

from zorch.commit.merkle import Opening

from pico_zorch.commit.pcs_commit import CommitData
from pico_zorch.uni_stark.air import Air
from pico_zorch.uni_stark.domain import Coset


@dataclass(frozen=True)
class FriParams:
    """Pico's FRI knobs (KoalaBearPoseidon2::new(): 1 / 84 / 16)."""

    log_blowup: int = 1
    num_queries: int = 84
    proof_of_work_bits: int = 16


@dataclass(frozen=True)
class StarkClaim:
    """The statement: this AIR holds on some height-2^degree_bits trace with
    these public values."""

    air: Air
    public_values: Array
    degree_bits: int


@dataclass(frozen=True)
class StarkWitness:
    """The `[2^degree_bits, width]` main trace, natural row order."""

    trace: Array


@dataclass(frozen=True)
class QuotientClaim:
    """`StarkClaim` with its trace commitment bound — the quotient stage's
    input, formed by the composite after `bind_instance`."""

    air: Air
    public_values: Array
    degree_bits: int
    trace_root: Array


@dataclass(frozen=True)
class QuotientWitness:
    """The trace plus its commit data (the LDE the quotient is evaluated
    on, and later the FRI opening's oracle)."""

    trace: Array
    trace_data: CommitData


@dataclass(frozen=True)
class TraceOpeningClaim:
    """The quotient stage's reduced claim: the committed trace and quotient
    polynomials open at ζ (trace also at ζ·g) to values that satisfy the
    out-of-domain identity under α."""

    trace_root: Array
    quotient_root: Array
    alpha: Array
    zeta: Array
    zeta_next: Array
    degree_bits: int


@dataclass(frozen=True)
class QuotientData:
    """Prover-only quotient stage output, the FRI opening's witness half:
    chunk evaluations, their domains, and the commit data."""

    chunks: Sequence[Array]
    qc_domains: Sequence[Coset]
    quotient_data: CommitData


@dataclass(frozen=True)
class QuotientProof:
    """The quotient stage's reduction proof. Only `quotient_root` is wire;
    `data` is prover-only (the FRI opening's witness half, zisk-zorch's
    `TraceCommitment` convention), absent on the verifier path and never
    serialized."""

    quotient_root: Array
    data: QuotientData | None = None


@dataclass(frozen=True)
class FriOpeningWitness:
    """Everything the FRI opening interpolates and opens: the natural-order
    evaluations and both commitments' prover data."""

    trace: Array
    trace_data: CommitData
    quotient: QuotientData


@dataclass(frozen=True)
class FriProof:
    """The reference `FriProof`: one commit per fold layer, the constant
    final polynomial, the PoW witness, and per-query openings of every
    committed matrix along the fold chain."""

    commit_phase_roots: Sequence[Array]
    final_poly: Array
    pow_witness: Array
    # Batched over queries (leading axis num_queries): the trace and quotient
    # leaf-row openings, then one batched pair-row opening per fold layer —
    # the reference's `CommitPhaseProofStep` carries only the sibling, row
    # column `(index >> layer) ^ 1 & 1` here.
    trace_openings: Opening
    quotient_openings: Opening
    commit_phase_openings: Sequence[Opening]


@dataclass(frozen=True)
class FriOpeningProof:
    """The opening stage's wire message: the out-of-domain opened values and
    the FRI low-degree proof (the reference's `OpenedValues` +
    `opening_proof`)."""

    trace_local: Array  # [width] extension
    trace_next: Array  # [width] extension
    quotient_chunks: Array  # [quotient_degree, 4] extension per base limb
    fri: FriProof


@dataclass(frozen=True)
class StarkProof:
    """The composite wire proof, the reference `Proof` field for field."""

    trace_root: Array
    quotient_root: Array
    opening: FriOpeningProof
    degree_bits: int
