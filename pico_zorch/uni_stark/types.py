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
    """`StarkClaim` once its trace commitment is bound — what the quotient
    stage can state, which `StarkClaim` alone cannot."""

    air: Air
    public_values: Array
    degree_bits: int
    trace_root: Array


@dataclass(frozen=True)
class QuotientWitness:
    """The trace plus its commit data — the LDE the quotient evaluates on,
    and later the opening's oracle."""

    trace: Array
    trace_data: CommitData


@dataclass(frozen=True)
class TraceOpeningClaim:
    """What survives the quotient stage: the committed trace and quotient
    polynomials open at ζ (trace also at ζ·g) to values satisfying the
    out-of-domain identity under α."""

    trace_root: Array
    quotient_root: Array
    alpha: Array
    zeta: Array
    zeta_next: Array
    degree_bits: int


@dataclass(frozen=True)
class QuotientData:
    """The opening's witness half. Prover-only: nothing here is a claim, so
    it travels by witness rather than widening the reduced claim."""

    chunks: Sequence[Array]
    qc_domains: Sequence[Coset]
    quotient_data: CommitData


@dataclass(frozen=True)
class QuotientProof:
    """Only `quotient_root` is wire. `data` rides along for the composite to
    hand to the opening stage, is absent on the verifier path, and is never
    serialized."""

    quotient_root: Array
    data: QuotientData | None = None


@dataclass(frozen=True)
class FriOpeningWitness:
    """Everything the opening interpolates and opens: the natural-order
    evaluations, and both commitments' prover data."""

    trace: Array
    trace_data: CommitData
    quotient: QuotientData


@dataclass(frozen=True)
class FriProof:
    """The reference `FriProof`: one commit per fold layer, the constant
    final polynomial, the PoW witness, and the per-query openings."""

    commit_phase_roots: Sequence[Array]
    final_poly: Array
    pow_witness: Array
    # Batched over queries on the leading axis. The reference's
    # `CommitPhaseProofStep` stores only a sibling; the full pair row is kept
    # instead, its sibling at column `(index >> layer) ^ 1 & 1`.
    trace_openings: Opening
    quotient_openings: Opening
    commit_phase_openings: Sequence[Opening]


@dataclass(frozen=True)
class FriOpeningProof:
    """The reference's `OpenedValues` + `opening_proof`."""

    trace_local: Array  # [width] extension
    trace_next: Array  # [width] extension
    quotient_chunks: Array  # [quotient_degree, 4] extension per base limb
    fri: FriProof


@dataclass(frozen=True)
class StarkProof:
    """The reference `Proof`, field for field."""

    trace_root: Array
    quotient_root: Array
    opening: FriOpeningProof
    degree_bits: int
