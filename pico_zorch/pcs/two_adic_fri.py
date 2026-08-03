# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Plonky3's `TwoAdicFriPcs`, commit side.

The polynomial commitment scheme Pico's machine prover reaches every heavy
kernel through. Its job on the commit side is small and entirely about
domains: extend each polynomial from its trace domain onto a blown-up coset,
put the rows in the committed order, and hand the resulting matrices to the
MMCS. Binding them to one root is the MMCS's business, and it knows nothing
about domains, blowup or FRI.

    commit(matrices) = mmcs.commit([bit_reverse(coset_lde(m)) for m in matrices])

Mixed heights matter here because Pico's `commit_main` commits every chip's
main trace in one call, each at whatever height that chip ran to. Each matrix
is extended against *its own* domain, so the committed heights differ by the
same ratios the trace heights did, and the MMCS folds them into one tree by
height.

Mirrors `fri/src/two_adic_pcs.rs` at brevis-network/Plonky3@7fbe1908, the fork
Pico v2.0.0 vendors.

Not the uni-stark path. `uni_stark/fri.py` carries its own single-height commit
plus the whole opening argument; this is the batched commit the machine prover
needs, and the two are worth consolidating once the opening side lands (see
issue #8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import frx
import frx.numpy as fnp
from frx import Array, lax
from zk_dtypes import koalabear_mont as F

from zorch.coding.reed_solomon import ReedSolomon

from pico_zorch.commit.mmcs import MerkleTreeMmcs

# KoalaBear's multiplicative-group generator, `Val::GENERATOR` in the
# reference: the coset the LDE lands on, disjoint from the trace domain.
GENERATOR = 3


@dataclass(frozen=True)
class TwoAdicFriPcs:
    """The commit side of Pico's `TwoAdicFriPcs`.

    Frozen so it can key a jit zone — the scheme is configuration, not data.
    """

    mmcs: MerkleTreeMmcs
    log_blowup: int = 1

    def lde(self, matrix: Array) -> Array:
        """One matrix extended onto the blown-up coset, in committed order.

        Bit-reversed because that is the layout the MMCS is committed over and
        the layout FRI's folding indexes; the natural-order codeword is only an
        intermediate.
        """
        code = ReedSolomon(
            matrix.shape[0],
            1 << self.log_blowup,
            matrix.dtype,
            coset_shift=fnp.array(GENERATOR, dtype=matrix.dtype),
        )
        natural = code.extend(matrix.T)
        return lax.bit_reverse(natural, dimensions=(1,)).T

    def commit(self, matrices: Sequence[Array]) -> tuple[Array, tuple[Array, ...]]:
        """Commit a batch of trace matrices, each against its own domain.

        Returns `(root, ldes)`. The extensions come back because the quotient
        stage reads its trace off them (`get_evaluations_on_domain` in the
        reference) rather than re-extending.
        """
        if not matrices:
            raise ValueError("commit requires at least one matrix")
        # Compared per matrix rather than as a set: a numpy dtype and the
        # scalar type compare equal but do not hash alike, so set equality
        # would reject the right dtype.
        foreign = sorted({str(m.dtype) for m in matrices if m.dtype != F})
        if foreign:
            raise ValueError(f"commit expects KoalaBear matrices, got {foreign}")
        ldes = tuple(self.lde(m) for m in matrices)
        root, _ = self.mmcs.commit(ldes)
        return root, ldes
