# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The LogUp permutation trace Pico's machine prover commits per chip.

Chips talk to each other by *lookups*: one chip sends a tuple, another receives
it, and the machine is sound only if every send is matched. LogUp turns that
multiset equality into one field identity. Each interaction contributes a term

    ±mult / (alpha + kind + beta*v0 + beta^2*v1 + ...)

with the sign negative for a receive, so a matched send/receive pair cancels
and the whole trace sums to zero exactly when the multisets agree. `alpha` and
`beta` are drawn after the main commitment, so a prover cannot fit its trace to
the random linear combination that fingerprints each tuple.

Interactions are batched `batch_size` to a column to trade width for degree —
a column holds the sum of its chunk's terms — and the final column carries the
running sum down the rows. Its last entry is the chip's regional cumulative
sum, which the machine transcript observes and the verifier checks sums to zero
across chips.

This takes interactions already evaluated to values: a chip's expression
language is its own business, and by the time LogUp runs, every interaction is
just a multiplicity and a tuple per row.

Mirrors `vm/src/machine/permutation.rs` at pico v2.0.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import frx.numpy as fnp
from frx import Array
from zk_dtypes import koalabearx4_mont as EF

from zorch.poly.univariate import powers


@dataclass(frozen=True)
class Lookup:
    """One interaction's static description. `kind` separates the tables a
    chip talks to, so a memory tuple cannot be matched against an ALU tuple
    that happens to hold the same values."""

    kind: int
    is_send: bool


def denominators(
    lookups: Sequence[Lookup],
    values: Sequence[Array],
    alpha: Array,
    beta: Array,
) -> list[Array]:
    """`alpha + kind + Σ_j beta^(j+1)·v_j`, one `[height]` column per lookup.

    The `beta` powers start at 1 for `kind` and continue from `beta` for the
    values, which is why a tuple's first value is weighted `beta` and not
    `beta^0` — getting that offset wrong still produces a consistent-looking
    trace, just not the reference's.
    """
    widest = max(v.shape[-1] for v in values)
    beta_powers = powers(beta, widest + 1)
    out = []
    for lookup, value in zip(lookups, values):
        acc = alpha + beta_powers[0] * fnp.array(lookup.kind, dtype=EF)
        for j in range(value.shape[-1]):
            acc = acc + beta_powers[j + 1] * value[..., j].astype(EF)
        out.append(acc)
    return out


def permutation_trace(
    lookups: Sequence[Lookup],
    values: Sequence[Array],
    mults: Sequence[Array],
    alpha: Array,
    beta: Array,
    batch_size: int,
) -> tuple[Array, Array]:
    """`(trace, regional_cumulative_sum)` for one chip.

    `values[i]` is `[height, k_i]` and `mults[i]` is `[height]`, both base
    field, in the reference's order: every send, then every receive. The trace
    is `[height, ceil(n / batch_size) + 1]` extension columns.
    """
    if len(lookups) != len(values) or len(lookups) != len(mults):
        raise ValueError(
            f"got {len(lookups)} lookups, {len(values)} value blocks and "
            f"{len(mults)} multiplicity columns"
        )
    if not lookups:
        raise ValueError("a permutation trace needs at least one lookup")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    denom = denominators(lookups, values, alpha, beta)
    # A receive's multiplicity enters negative; that sign is the whole reason
    # a matched pair cancels.
    terms = [
        (m.astype(EF) if lk.is_send else -m.astype(EF)) / d
        for lk, m, d in zip(lookups, mults, denom)
    ]

    chunks = [
        sum(terms[i : i + batch_size][1:], terms[i])
        for i in range(0, len(terms), batch_size)
    ]
    columns = fnp.stack(chunks, axis=-1)

    # The running sum is a prefix scan down the rows of each row's total, and
    # its last entry is what the transcript observes.
    row_totals = columns.sum(axis=-1)
    running = fnp.cumsum(row_totals, axis=0)
    trace = fnp.concatenate([columns, running[:, None]], axis=-1)
    return trace, running[-1]
