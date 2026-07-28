# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Claims, witnesses and proof wire types for the uni-stark stage.

Mirrors the reference `Proof` (Plonky3 uni-stark/src/proof.rs at
brevis-network/Plonky3@7fbe1908): commitments, out-of-domain opened values,
the FRI opening proof, and the trace's log height. Field arrays carry raw
Montgomery u32 (`koalabear_mont` views); the canonical form appears only on
comparison and serialization surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from frx import Array

from zorch.commit.merkle import Opening

from pico_zorch.uni_stark.air import Air


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
class CommitPhaseOpening:
    """One query's step through one fold layer: the opened pair row (2
    extension values as 8 base columns) and its Merkle path — the reference's
    `CommitPhaseProofStep` carries the sibling value only, which is row
    column `(index >> layer) ^ 1 & 1` here."""

    opening: Opening


@dataclass(frozen=True)
class FriProof:
    """The reference `FriProof`: one commit per fold layer, the constant
    final polynomial, the PoW witness, and per-query openings of every
    committed matrix along the fold chain."""

    commit_phase_roots: Sequence[Array]
    final_poly: Array
    pow_witness: Array
    # Per query, per input round (trace, quotient): the leaf-row opening.
    input_openings: Sequence[Sequence[Opening]]
    # Per query, per fold layer: the pair-row opening.
    commit_phase_openings: Sequence[Sequence[CommitPhaseOpening]]


@dataclass(frozen=True)
class StarkProof:
    trace_root: Array
    quotient_root: Array
    trace_local: Array  # [width] extension
    trace_next: Array  # [width] extension
    quotient_chunks: Array  # [quotient_degree, 4] extension per base limb
    fri: FriProof
    degree_bits: int
