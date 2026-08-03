# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Plonky3's `MerkleTreeMmcs` — a Merkle commitment over a *batch* of matrices.

This sits one layer below the PCS. A two-adic FRI PCS coset-extends each
polynomial to its evaluation domain, bit-reverses, and hands the resulting
matrices here; binding them to a single root is this scheme's job, and it knows
nothing about domains, blowup or FRI.

A uni-stark commits one matrix at a time, so a plain Merkle tree suffices.
Pico's machine prover does not: `commit_main` hands every chip's main trace to
a single `pcs.commit`, each at whatever height that chip ran to. Committing
them separately would give a different root, so the batching is part of the
protocol rather than an optimization.

The construction folds the batch into one tree by *height*. Sort the matrices
tallest-first and hash the tallest into the leaf layer; then, at every layer,
compress the previous layer pairwise and mix in any matrix whose height equals
that layer's length:

    no matrix at this height:  digest[i] = compress(prev[2i], prev[2i+1])
    a matrix at this height:   digest[i] = compress(
                                   compress(prev[2i], prev[2i+1]),
                                   hash(row i of each such matrix, joined))

So a short matrix enters the tree at the layer where the tree has become as
narrow as the matrix is tall, and every matrix is bound by the one root.

Two details are easy to get subtly wrong, and both fail as a wrong root rather
than an error:

* Rows of matrices sharing a height are **concatenated before hashing**, not
  hashed separately and combined — for injected layers and for the leaf layer
  alike.
* The mixed-in digest is compressed *with* the pairwise result, not in place of
  it, so an injected layer costs two compressions per node.

Mirrors `merkle-tree/src/merkle_tree.rs` at brevis-network/Plonky3@7fbe1908,
the fork Pico v2.0.0 vendors.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import frx
import frx.numpy as fnp
from frx import Array

from zorch.commit.merkle import MerkleTree


def _is_power_of_two(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def _grouped_by_height(matrices: Sequence[Array]) -> dict[int, list[Array]]:
    """Matrices keyed by height, each group in the caller's original order.

    Order within a height is load-bearing: the rows are concatenated before
    hashing, so a permutation of the group is a different commitment. The
    reference sorts only *between* heights (a stable sort by descending
    height), which leaves the caller's order intact within one.
    """
    groups: dict[int, list[Array]] = defaultdict(list)
    for matrix in matrices:
        groups[matrix.shape[0]].append(matrix)
    return groups


@dataclass(frozen=True)
class MerkleTreeMmcs:
    """Plonky3's `MerkleTreeMmcs` over Pico's Poseidon2-KoalaBear parameters.

    Frozen so it can key a jit zone, matching how `MerkleTree` is passed
    around: the scheme is configuration, not data.
    """

    tree: MerkleTree
    compressor: Any

    @property
    def digest_elems(self) -> int:
        return self.tree.digest_elems

    def commit(self, matrices: Sequence[Array]) -> tuple[Array, list[Array]]:
        """Commit matrices of differing heights to one root.

        `matrices` are row-major `(height, width)` in the layout they are
        committed in — for a FRI PCS that is the bit-reversed coset LDE.
        Returns `(root, digest_layers)` in the shape a single-matrix
        `MerkleTree.commit` returns, so the query machinery reads it unchanged.

        The caller's order is preserved within each height; the batch is sorted
        only across heights.
        """
        if not matrices:
            raise ValueError("commit requires at least one matrix")

        heights = [m.shape[0] for m in matrices]
        # Plonky3 permits non-power-of-two heights, but then a layer can be odd
        # and pad, and an injected matrix can run out before the layer does.
        # Neither arises for the power-of-two heights a two-adic PCS commits,
        # and an unreachable branch in a commitment is one no test would ever
        # cover — so this refuses the case rather than guessing at it.
        bad = [h for h in heights if not _is_power_of_two(h)]
        if bad:
            raise ValueError(
                f"commit needs power-of-two heights, got {sorted(bad)}; the "
                "reference's odd-layer padding and short-matrix tail are "
                "deliberately not implemented"
            )

        groups = _grouped_by_height(matrices)
        descending = sorted(groups, key=lambda h: -h)

        # The tallest matrices form the leaf layer; the rest wait for the layer
        # whose length equals their height.
        layer = self.tree.hash_leaves(_join(groups[descending[0]]))
        digest_layers = [layer]
        pending = {h: groups[h] for h in descending[1:]}

        while layer.shape[0] > 1:
            pairs = layer.reshape(-1, 2, self.digest_elems)
            folded = frx.vmap(self.compressor.compress)(pairs)
            inject = pending.pop(folded.shape[0], None)
            if inject is not None:
                rows = self.tree.hash_leaves(_join(inject))
                folded = frx.vmap(self.compressor.compress)(
                    fnp.stack([folded, rows], axis=1)
                )
            layer = folded
            digest_layers.append(layer)

        if pending:
            # Only reachable if a matrix is taller than the leaf layer, which
            # the sort rules out — a guard against a future edit, not a
            # user-facing error.
            raise AssertionError(
                f"matrices never injected: heights {sorted(pending)}"
            )

        return layer[0], digest_layers


def _join(matrices: Sequence[Array]) -> Array:
    """Concatenate same-height matrices along their columns.

    This is the whole reason a group is a list: the reference hashes
    `matrices.flat_map(|m| m.row(i))`, one digest over the joined row, not one
    digest per matrix.
    """
    return matrices[0] if len(matrices) == 1 else fnp.concatenate(matrices, axis=1)
