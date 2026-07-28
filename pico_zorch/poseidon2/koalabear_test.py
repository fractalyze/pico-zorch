# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Byte-match of the Poseidon2-KoalaBear-16 Merkle stack against Plonky3.

Also the repo's wiring smoke test: a pass proves the zorch pin, the frx
runtime, and the koalabear dtypes all resolve end to end.
"""

from __future__ import annotations

import frx.numpy as fnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from pico_zorch.poseidon2.koalabear import koalabear16_merkle, koalabear16_perm

# Plonky3 golden (p3_commit=4318eba, default_koalabear_poseidon2_16):
# PaddingFreeSponge<_,16,8,8> leaves + TruncatedPermutation<_,2,8,16> over
# arange(32) reshaped to a 4x8 matrix (hash rows, fold pairs).
_PLONKY3_MERKLE_ROOT_4X8 = fnp.array(
    [
        1670701318,
        437280557,
        23464423,
        637192971,
        1642004034,
        359231982,
        157670030,
        587973557,
    ],
    dtype=F,
)


class KoalabearStackTest(absltest.TestCase):
    def test_perm_shape_and_dtype(self) -> None:
        out = koalabear16_perm().permute(fnp.zeros((16,), dtype=F))
        self.assertEqual(out.shape, (16,))
        self.assertEqual(out.dtype, F)

    def test_merkle_root_matches_plonky3_golden(self) -> None:
        _, _, tree = koalabear16_merkle()
        raw_root, _ = tree.commit(fnp.arange(32, dtype=F).reshape(4, 8))
        self.assertTrue(bool(fnp.array_equal(raw_root, _PLONKY3_MERKLE_ROOT_4X8)))


if __name__ == "__main__":
    absltest.main()
