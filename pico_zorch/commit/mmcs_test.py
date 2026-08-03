# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`MerkleTreeMmcs` byte-matches Plonky3's.

`golden/`'s `emit_batch_commit` runs the same matrices through the reference
`pcs.commit`, so the root here is the reference semantics by construction. The
fixture is shaped to reach both paths a single-matrix commit cannot: two
matrices share the tallest height (several matrices' rows hashing into one
leaf) and two enter lower down by injection.
"""

from __future__ import annotations

import json
import pathlib

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import lax
from zk_dtypes import koalabear_mont as F

from zorch.coding.reed_solomon import ReedSolomon

from pico_zorch.commit.mmcs import MerkleTreeMmcs
from pico_zorch.poseidon2.koalabear import koalabear16_merkle
from pico_zorch.uni_stark.fri import GENERATOR

_GOLDEN = pathlib.Path(__file__).parent / "testdata" / "golden" / "batch_commit.json"


def _canonical(x) -> np.ndarray:
    return np.asarray(lax.convert_element_type(x, fnp.uint32))


class MixedHeightCommitTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.golden = json.loads(_GOLDEN.read_text())
        cls.log_blowup = cls.golden["log_blowup"]

    def _committed_matrices(self) -> list:
        """Each matrix's LDE in the committed (bit-reversed) layout, which is
        what `TwoAdicFriPcs::commit` feeds the MMCS."""
        committed = []
        for m in self.golden["matrices"]:
            values = fnp.array(m["values"], dtype=F)
            code = ReedSolomon(
                m["height"],
                1 << self.log_blowup,
                F,
                coset_shift=fnp.array(GENERATOR, dtype=F),
            )
            natural = code.extend(values.T)
            committed.append(lax.bit_reverse(natural, dimensions=(1,)).T)
        return committed

    def test_lde_matches_reference(self) -> None:
        """Pin the extension before the tree, so a root mismatch cannot be
        blamed on the wrong thing."""
        for m, lde in zip(self.golden["matrices"], self._committed_matrices()):
            natural = lax.bit_reverse(lde.T, dimensions=(1,)).T
            np.testing.assert_array_equal(
                _canonical(natural),
                np.array(m["lde_natural_order"]),
                err_msg=f"LDE of the height-{m['height']} matrix",
            )

    def test_root_matches_reference(self) -> None:
        _, compressor, tree = koalabear16_merkle()
        mmcs = MerkleTreeMmcs(tree, compressor)
        root, layers = mmcs.commit(self._committed_matrices())
        np.testing.assert_array_equal(
            _canonical(root), np.array(self.golden["root"])
        )
        # The tallest committed matrix is 2^4 rows after the blowup, so the
        # tree is 5 layers deep including the leaves.
        tallest = max(m.shape[0] for m in self._committed_matrices())
        self.assertEqual(len(layers), tallest.bit_length())

    def test_order_within_a_height_is_load_bearing(self) -> None:
        """Rows of same-height matrices are concatenated before hashing, so
        swapping two of them is a different commitment — not a no-op."""
        _, compressor, tree = koalabear16_merkle()
        mmcs = MerkleTreeMmcs(tree, compressor)
        committed = self._committed_matrices()
        tallest = max(m.shape[0] for m in committed)
        same = [i for i, m in enumerate(committed) if m.shape[0] == tallest]
        self.assertGreaterEqual(len(same), 2, "fixture must share a tallest height")

        swapped = list(committed)
        swapped[same[0]], swapped[same[1]] = swapped[same[1]], swapped[same[0]]
        other, _ = mmcs.commit(swapped)
        root, _ = mmcs.commit(committed)
        self.assertFalse(
            np.array_equal(_canonical(root), _canonical(other)),
            "swapping two matrices of equal height must change the root",
        )

    def test_rejects_a_non_power_of_two_height(self) -> None:
        _, compressor, tree = koalabear16_merkle()
        ragged = [fnp.zeros((6, 2), dtype=F)]
        with self.assertRaisesRegex(ValueError, "power-of-two"):
            MerkleTreeMmcs(tree, compressor).commit(ragged)


if __name__ == "__main__":
    absltest.main()
