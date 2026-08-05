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
from functools import lru_cache
from typing import Sequence

import frx
import frx.numpy as fnp
from frx import Array, lax
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.poly.univariate import powers

from zorch.coding.reed_solomon import BitReversedReedSolomon

from zorch.pcs.fold import from_base_field, to_base_field

from pico_zorch.uni_stark.fri import _eval_columns, _lde_code, sample_query_indices


@dataclass(frozen=True)
class MatrixOpening:
    """One matrix's place in the batched argument.

    Not to be confused with zorch's `Opening`, which is a Merkle path; this is
    an input to the argument, that is an output of it."""

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
    rounds: Sequence[Sequence[MatrixOpening]],
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


@lru_cache(maxsize=None)
def _fold_code(n: int, log_blowup: int) -> BitReversedReedSolomon:
    """The code the fold chain steps down, cached and forced to compile time.

    Constructing a coset code eagerly builds a block-length powers table; built
    under a trace that table would be traced, and exporting the whole opening
    is exactly the case that happens. Mirrors `_lde_code`, minus the coset
    shift — folding works on the plain subgroup.
    """
    with frx.ensure_compile_time_eval():
        return BitReversedReedSolomon(n, 1 << log_blowup, F)


def commit_phase(tree, log_blowup: int, accumulators: dict[int, Array], transcript):
    """FRI's commit phase over inputs of *several* heights.

    Folding halves the codeword each round, so an accumulator shorter than the
    tallest has no rows to contribute until the fold has come down to its
    height — at which point it is added in elementwise. Structurally the same
    idea as the MMCS's injection, and for the same reason: one argument has to
    bind polynomials that live on different domains.

    The layer count follows the codeword shape rather than any value, so the
    loop unrolls at trace time and the whole phase is one dispatch.
    """
    heights = sorted(accumulators, reverse=True)
    folded = accumulators[heights[0]]
    pending = {h: accumulators[h] for h in heights[1:]}
    code = _fold_code(heights[0] >> log_blowup, log_blowup)

    t = transcript
    roots, layers = [], []
    while folded.shape[0] > (1 << log_blowup):
        leaves = to_base_field(code.pair_leaves(folded))
        root, digest_layers = tree.commit(leaves)
        t = t.observe(root)
        t, beta = t.sample_ext()
        folded = code.fold(folded, beta)
        mix = pending.pop(folded.shape[0], None)
        if mix is not None:
            folded = folded + mix
        roots.append(root)
        layers.append((leaves, tuple(digest_layers)))

    if pending:
        # Only reachable if an accumulator is taller than the tallest input,
        # which the sort rules out, or if a height is not a power of two.
        raise AssertionError(f"accumulators never mixed in: {sorted(pending)}")

    final_poly = folded[0]
    t = t.observe(final_poly)
    return final_poly, tuple(roots), tuple(layers), t


def query_indices(transcript, log_max_height: int, num_queries: int):
    """Query positions, sampled against the *tallest* committed height.

    Every round is queried at the same position folded down to its own height,
    so the sampling width is the global maximum rather than any one round's.
    """
    return sample_query_indices(transcript, log_max_height, num_queries)


def round_index(index: Array, log_max_height: int, log_round_height: int) -> Array:
    """A global query position mapped into one round's shorter codeword.

    `index >> (log_global_max - log_round_max)`: the reference's
    `bits_reduced`. Correct only because the committed order is bit-reversed,
    which puts a codeword's positions in the same prefix relationship its
    domain has.
    """
    if log_round_height > log_max_height:
        raise ValueError(
            f"round height 2^{log_round_height} exceeds the global maximum "
            f"2^{log_max_height}"
        )
    return index >> (log_max_height - log_round_height)


def observe_openings(transcript, all_opened: Sequence[Sequence[Sequence[Array]]]):
    """Observe every opened value, in round -> matrix -> point -> column order.

    The reference observes each `y` individually before sampling alpha, so
    alpha depends on all of them; a consumer that observed a different order
    would derive a different alpha and diverge from there on.
    """
    flat = [v for round_ in all_opened for mat in round_ for v in mat]
    return transcript.observe(fnp.concatenate(flat))


def commit_phase_over_rounds(
    tree,
    rounds: Sequence[Sequence[MatrixOpening]],
    transcript,
    log_blowup: int,
    proof_of_work_bits: int,
    num_queries: int,
):
    """Everything in `open` from the opened values through the query indices.

    Returns `(all_opened, final_poly, roots, layers, pow_witness, indices, t)`.
    The per-round input openings are not here: those read the MMCS's
    mixed-height prover data, which opens differently from a single-matrix
    tree.
    """
    all_opened = [
        [opened_values(o.trace, o.points) for o in round_] for round_ in rounds
    ]
    t = observe_openings(transcript, all_opened)
    t, alpha = t.sample_ext()

    accumulators = reduced_openings(rounds, alpha, log_blowup)
    final_poly, roots, layers, t = commit_phase(tree, log_blowup, accumulators, t)

    t, pow_witness = t.grind(proof_of_work_bits)
    log_max_height = max(accumulators).bit_length() - 1
    t, indices = query_indices(t, log_max_height, num_queries)
    return all_opened, final_poly, roots, layers, pow_witness, indices, t


def commit_phase_openings(tree, layers, index: Array, log_blowup: int):
    """One query's path through the fold chain.

    Layer `i` is queried at `index >> i`; since the layer commits *pairs*, the
    committed row is `index >> (i + 1)` and the value the verifier is missing
    is the other half of that pair — `(index >> i) ^ 1` picks which half.

    Returns `[(sibling_value, path)]`, one per layer.
    """
    steps = []
    for i, (leaves, digest_layers) in enumerate(layers):
        pair_index = index >> (i + 1)
        opened = tree.open(leaves, digest_layers, pair_index)
        # The row is a base-field pair of extension elements; take the half the
        # verifier cannot recompute.
        pair = from_base_field(opened.row[None, :], EF, 2)[0]
        sibling = pair[((index >> i) ^ 1) % 2]
        # Stacked, matching `MerkleTreeMmcs.open_batch`: a path is one array
        # per level, and every consumer wants them together.
        steps.append((sibling, fnp.stack(opened.path)))
    return steps
