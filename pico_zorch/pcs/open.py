# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The batched opening argument of `TwoAdicFriPcs`, prover side.

`machine::prove` opens four commitments at once — preprocessed, main,
permutation, quotient — with each matrix carrying its own list of points and
its own height. All of it reduces to a single FRI instance:

    reduced[h][X] += alpha^offset * (sum_i alpha^i*y_i - sum_i alpha^i*p_i[X]) / (z - X)

one accumulator per distinct height `h`, and FRI folds them together as it
descends.

Two details differ from the uni-stark case and both fail as a wrong proof
rather than an error:

* **The alpha offset is per height, not global.** The reference keeps
  `num_reduced[log_height]` and advances it by each matrix's width, so two
  matrices of *different* heights both start their alpha runs from wherever
  their own height had got to. A single global counter gives a consistent but
  different proof.
* **Order is round-major, then matrix, then point.** The offset advances in
  exactly that order, so a consumer that groups by height first — the natural
  way to write it — assigns different alpha powers.

Mirrors `fri/src/two_adic_pcs.rs` at brevis-network/Plonky3@7fbe1908.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import frx
import frx.numpy as fnp
from frx import Array, lax
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.poly.univariate import powers

from pico_zorch.uni_stark.fri import _eval_columns, _lde_code


@dataclass(frozen=True)
class Opening:
    """One matrix's place in the batched argument."""

    #: The committed (bit-reversed) coset LDE, `[height << log_blowup, width]`.
    lde: Array
    #: The trace it extends, `[height, width]` — the opened values come from
    #: this rather than from `lde`, which is field-identical and cheaper: both
    #: describe the same polynomial, and the trace is the smaller one.
    trace: Array
    #: Extension points this matrix is opened at.
    points: Sequence[Array]


def opened_values(matrix: Array, points: Sequence[Array]) -> list[Array]:
    """`p_i(z)` for every column i and point z — `open`'s public half.

    The reference interpolates the LDE's low coset; interpolating the trace
    gives the same polynomial and so the same values, over a domain
    `log_blowup` times smaller.
    """
    coeffs = lax.ntt(matrix.T, ntt_type="INTT", ntt_length=matrix.shape[0])
    return [_eval_columns(coeffs, z) for z in points]


def _inv_denominators(point: Array, height: int, log_blowup: int) -> Array:
    """`1/(z - X)` over the committed domain of an LDE `height` rows tall.

    The reference computes this once for the tallest matrix opened at `z` and
    lets shorter ones truncate — valid because a bit-reversed prefix of a coset
    *is* the smaller coset. Computing it per height is the same values without
    depending on that identity holding.
    """
    domain = _lde_code(height >> log_blowup, log_blowup).domain()
    return (point - domain.astype(EF)) ** -1


def reduced_openings(
    rounds: Sequence[Sequence[Opening]],
    alpha: Array,
    log_blowup: int,
) -> dict[int, Array]:
    """The FRI inputs, one accumulator per distinct committed height.

    Returns `{height: [height] extension column}`. FRI consumes these tallest
    first, mixing each in as its fold reaches that height.
    """
    if not rounds or not any(rounds):
        raise ValueError("an opening argument needs at least one matrix")

    accumulators: dict[int, Array] = {}
    # Per-height column counters — see the module docstring on why this is not
    # one global counter.
    consumed: dict[int, int] = {}

    for opening in (o for round_ in rounds for o in round_):
        height = opening.lde.shape[0]
        width = opening.lde.shape[1]
        if height not in accumulators:
            accumulators[height] = fnp.zeros((height,), dtype=EF)
            consumed[height] = 0

        # sum_i alpha^i * p_i[X], the only term depending on both row and
        # column; everything else factors out of the row loop.
        alpha_pows = powers(alpha, width)
        rows = (opening.lde.astype(EF) * alpha_pows[None, :]).sum(axis=-1)

        for point, values in zip(
            opening.points, opened_values(opening.trace, opening.points)
        ):
            offset = alpha ** consumed[height]
            reduced_y = (alpha_pows * values).sum()
            accumulators[height] = accumulators[height] + offset * (
                reduced_y - rows
            ) * _inv_denominators(point, height, log_blowup)
            consumed[height] += width

    return accumulators


def fri_input(accumulators: dict[int, Array]) -> list[Array]:
    """The accumulators tallest-first, which is the order FRI folds in."""
    return [accumulators[h] for h in sorted(accumulators, reverse=True)]
