# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Byte-match of the pcs trace commit against the reference TwoAdicFriPcs
(golden/ commits the Fibonacci trace through the fork's own pcs.commit)."""

from __future__ import annotations

import json
import pathlib

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import lax
from zk_dtypes import koalabear_mont as F

from pico_zorch.commit.pcs_commit import (
    GENERATOR,
    bit_reverse_rows,
    commit_pcs,
    coset_lde,
)
from pico_zorch.poseidon2.koalabear import koalabear16_merkle

_GOLDEN = pathlib.Path(__file__).parent / "testdata" / "golden" / "trace_commit.json"


def _canonical(x) -> np.ndarray:
    return np.asarray(lax.convert_element_type(x, fnp.uint32))


class PcsCommitTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.golden = json.loads(_GOLDEN.read_text())
        self.trace = fnp.array(self.golden["trace"], dtype=F)

    def test_coset_lde_matches_reference_natural_order(self) -> None:
        lde = coset_lde(self.trace, 1, fnp.array(GENERATOR, dtype=F))
        want = np.array(self.golden["lde_natural_order"], dtype=np.uint32)
        np.testing.assert_array_equal(_canonical(lde), want)

    def test_commit_root_matches_reference(self) -> None:
        _, _, tree = koalabear16_merkle()
        root, data = commit_pcs(tree, [self.trace], 1)
        want = np.array(self.golden["root"], dtype=np.uint32)
        np.testing.assert_array_equal(_canonical(root), want)
        self.assertEqual(data.matrices[0].shape, (16, 2))
        # The committed matrix is the bit-reversed LDE.
        lde = coset_lde(self.trace, 1, fnp.array(GENERATOR, dtype=F))
        np.testing.assert_array_equal(
            _canonical(data.matrices[0]), _canonical(bit_reverse_rows(lde))
        )


if __name__ == "__main__":
    absltest.main()
