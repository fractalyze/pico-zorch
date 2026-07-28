# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Claims, witnesses and proof wire types for the uni-stark stages.

The chain is the standard STARK reduction, each link trading a statement
about *all* of a domain for one about a single random point:

  an execution satisfies the AIR
    → a quotient polynomial exists (constraints vanish on the trace domain)
    → committed polynomials evaluate consistently at a random ζ
    → those evaluations are backed by genuinely low-degree functions

Only the last link needs cryptography; the first two are Schwartz-Zippel,
which is why a cheating prover has to be committed to its trace *before* the
challenges are drawn. Reduced claims are execution values the verifier
re-derives, never serialized; field arrays carry raw Montgomery u32, with
canonical form appearing only on comparison and wire surfaces.
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
    """Pico's FRI knobs (KoalaBearPoseidon2::new(): 1 / 84 / 16).

    These are the security budget: each query catches a codeword that is
    far from low-degree with probability ~1 − 2^-log_blowup, and grinding
    buys bits outright, so conjectured soundness is
    log_blowup·num_queries + proof_of_work_bits — 100 bits as configured."""

    log_blowup: int = 1
    num_queries: int = 84
    proof_of_work_bits: int = 16


@dataclass(frozen=True)
class StarkClaim:
    """The NP statement: some height-2^degree_bits execution trace satisfies
    every AIR constraint and is consistent with these public values. The
    trace itself is the witness, and stays private."""

    air: Air
    public_values: Array
    degree_bits: int


@dataclass(frozen=True)
class StarkWitness:
    """The witness: the execution trace, one row per cycle, natural order."""

    trace: Array


@dataclass(frozen=True)
class QuotientClaim:
    """`StarkClaim` once the trace is committed.

    The commitment is what makes the following challenges sound: the prover
    is bound to one specific trace before α exists, so it cannot choose a
    trace that happens to satisfy the constraint combination it is given."""

    air: Air
    public_values: Array
    degree_bits: int
    trace_root: Array


@dataclass(frozen=True)
class QuotientWitness:
    """The trace plus its commit data — the low-degree extension the
    quotient evaluates on, and later the oracle the queries open."""

    trace: Array
    trace_data: CommitData


@dataclass(frozen=True)
class TraceOpeningClaim:
    """The evaluation claim the AIR has been reduced to.

    Once Σ_j α^j·c_j = Q·Z_H is checked at a single ζ drawn from the
    extension, the AIR is discharged: the identity is a polynomial identity
    of bounded degree, so failing it anywhere means failing at a random
    point except with probability deg/|EF|. ζ·g appears because transition
    constraints read the next row, which on a multiplicative domain is the
    same polynomial evaluated one generator step along."""

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
    """The reference `FriProof`.

    One commitment per fold layer, the constant the folding terminates in,
    and per-query openings tying the layers together — the transcript of a
    proximity test, where each query independently checks that consecutive
    layers agree on the folding relation."""

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
