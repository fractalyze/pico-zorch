# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""TwoAdicFriPcs.commit on zorch blocks.

The reference (Plonky3 fri/src/two_adic_pcs.rs at brevis-network/Plonky3@
7fbe1908, the fork Pico v2.0.0 vendors) commits evaluations on a size-n
subgroup as: coset LDE with shift `GENERATOR / domain.shift` onto the
blown-up domain, rows bit-reversed, one Merkle tree over the row leaves.
That shift choice puts every committed LDE on the same coset `GENERATOR·<g>`
whatever domain it came from. A batch of equal-height matrices shares one
tree, leaf i hashing every matrix's row i in commit order.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Sequence

import frx
import frx.numpy as fnp
from frx import Array, lax

from zorch.coding.reed_solomon import ReedSolomon
from zorch.commit.merkle import MerkleTree

# KoalaBear's multiplicative-group generator: Val::GENERATOR in the reference,
# the coset shift for a natural (shift-1) evaluation domain.
GENERATOR = 3


def coset_lde(evals: Array, log_blowup: int, shift: Array) -> Array:
    """LDE the `[n, w]` column matrix of subgroup evaluations onto the
    shift·<g> coset of size `n << log_blowup`, natural domain order."""
    n, _ = evals.shape
    code = ReedSolomon(n, 1 << log_blowup, evals.dtype, coset_shift=shift)
    return code.extend(evals.T).T


def bit_reverse_rows(matrix: Array) -> Array:
    """Row i holds the evaluation at exponent reverse_bits(i) — the layout
    that makes a fold's conjugate pair adjacent."""
    return lax.bit_reverse(matrix, dimensions=(0,))


@dataclass(frozen=True)
class CommitData:
    """Prover-side result of one pcs commit: the bit-reversed LDE matrices
    (reference row order) and the shared tree's digest layers."""

    matrices: tuple[Array, ...]
    leaves: Array
    digest_layers: list[Array]


def commit_matrices(
    tree: MerkleTree, matrices: Sequence[Array]
) -> tuple[Array, CommitData]:
    """One Merkle tree over equal-height matrices. The reference's
    MerkleTreeMmcs also injects shorter matrices at matching layers; no
    caller here commits mixed heights, so that path is unimplemented."""
    heights = {m.shape[0] for m in matrices}
    if len(heights) != 1:
        raise ValueError(f"matrices must share a height, got {sorted(heights)}")
    leaves = (
        fnp.concatenate(list(matrices), axis=1) if len(matrices) > 1 else matrices[0]
    )
    raw_root, digest_layers = tree.commit(leaves)
    return raw_root, CommitData(tuple(matrices), leaves, digest_layers)


@partial(frx.jit, static_argnames=("tree", "log_blowup"))
def _commit_pcs(tree, evals, shifts, log_blowup):
    ldes = [
        bit_reverse_rows(coset_lde(e, log_blowup, s)) for e, s in zip(evals, shifts)
    ]
    leaves = fnp.concatenate(ldes, axis=1) if len(ldes) > 1 else ldes[0]
    raw_root, digest_layers = tree.commit(leaves)
    return tuple(ldes), leaves, raw_root, tuple(digest_layers)


def commit_pcs(
    tree: MerkleTree,
    evals: Sequence[Array],
    log_blowup: int,
    *,
    shifts: Sequence[Any] | None = None,
) -> tuple[Array, CommitData]:
    """The reference `pcs.commit`: per-matrix coset LDE, rows bit-reversed,
    one shared tree. `shifts` defaults to GENERATOR, which is correct for
    natural (shift-1) domains; a quotient chunk passes GENERATOR/its own
    shift."""
    dtype = evals[0].dtype
    if shifts is None:
        shifts = [fnp.array(GENERATOR, dtype=dtype)] * len(evals)
    ldes, leaves, raw_root, digest_layers = _commit_pcs(
        tree, tuple(evals), tuple(shifts), log_blowup
    )
    return raw_root, CommitData(ldes, leaves, list(digest_layers))
