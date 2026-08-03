# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Byte-match of FriOpener.commit against the reference TwoAdicFriPcs
(golden/ commits the Fibonacci trace through the fork's own pcs.commit)."""

from __future__ import annotations

import json
import pathlib

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import lax
from zk_dtypes import koalabear_mont as F

from pico_zorch.poseidon2.koalabear import koalabear16_merkle
from pico_zorch.uni_stark.fri import GENERATOR, FriOpener

_GOLDEN = pathlib.Path(__file__).parent / "testdata" / "golden" / "trace_commit.json"


def _canonical(x) -> np.ndarray:
    return np.asarray(lax.convert_element_type(x, fnp.uint32))


class PcsCommitTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.golden = json.loads(_GOLDEN.read_text())
        self.trace = fnp.array(self.golden["trace"], dtype=F)

    def test_commit_root_matches_reference(self) -> None:
        _, _, tree = koalabear16_merkle()
        root, data = FriOpener(tree).commit([self.trace], shifts=[GENERATOR])
        want = np.array(self.golden["root"], dtype=np.uint32)
        np.testing.assert_array_equal(_canonical(root), want)
        self.assertEqual(data.matrices[0].shape, (16, 2))
        natural = fnp.array(self.golden["lde_natural_order"], dtype=F)
        np.testing.assert_array_equal(
            _canonical(data.matrices[0]),
            _canonical(lax.bit_reverse(natural, dimensions=(0,))),
        )


if __name__ == "__main__":
    absltest.main()
