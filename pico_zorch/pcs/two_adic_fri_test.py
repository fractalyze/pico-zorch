# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""`TwoAdicFriPcs.commit` byte-matches Plonky3's on a mixed-height batch.

`mmcs_test` pins the tree given already-extended matrices; this pins the layer
above it, so a root mismatch localizes to one of the two rather than "somewhere
in the commit".
"""

from __future__ import annotations

import json
import pathlib

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import lax
from zk_dtypes import koalabear_mont as F

from pico_zorch.commit.mmcs import MerkleTreeMmcs
from pico_zorch.pcs.two_adic_fri import TwoAdicFriPcs
from pico_zorch.poseidon2.koalabear import koalabear16_merkle

_GOLDEN = (
    pathlib.Path(__file__).parent.parent
    / "commit"
    / "testdata"
    / "golden"
    / "batch_commit.json"
)


def _canonical(x) -> np.ndarray:
    return np.asarray(lax.convert_element_type(x, fnp.uint32))


class TwoAdicFriPcsTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.golden = json.loads(_GOLDEN.read_text())
        _, compressor, tree = koalabear16_merkle()
        cls.pcs = TwoAdicFriPcs(
            MerkleTreeMmcs(tree, compressor), log_blowup=cls.golden["log_blowup"]
        )
        cls.matrices = [
            fnp.array(m["values"], dtype=F) for m in cls.golden["matrices"]
        ]

    def test_root_matches_reference(self) -> None:
        root, _ = self.pcs.commit(self.matrices)
        np.testing.assert_array_equal(
            _canonical(root), np.array(self.golden["root"])
        )

    def test_lde_matches_reference(self) -> None:
        """The extension is pinned separately from the tree, so a failure says
        which layer moved."""
        for want, matrix in zip(self.golden["matrices"], self.matrices):
            natural = lax.bit_reverse(self.pcs.lde(matrix).T, dimensions=(1,)).T
            np.testing.assert_array_equal(
                _canonical(natural),
                np.array(want["lde_natural_order"]),
                err_msg=f"LDE of the height-{want['height']} matrix",
            )

    def test_each_matrix_extends_against_its_own_domain(self) -> None:
        """A batch is not extended to a common height: each matrix keeps its
        own blowup ratio, which is what lets the MMCS place it by height."""
        _, ldes = self.pcs.commit(self.matrices)
        for want, lde in zip(self.golden["matrices"], ldes):
            self.assertEqual(lde.shape[0], want["height"] << self.golden["log_blowup"])

    def test_rejects_a_foreign_dtype(self) -> None:
        with self.assertRaisesRegex(ValueError, "KoalaBear"):
            self.pcs.commit([fnp.zeros((4, 2), dtype=fnp.uint32)])


if __name__ == "__main__":
    absltest.main()
