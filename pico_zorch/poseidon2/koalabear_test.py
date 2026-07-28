# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Byte-match of the Poseidon2-KoalaBear-16 permutation against the Pico
reference (golden/ links the Plonky3 fork Pico v2.0.0 vendors and builds the
permutation from Pico's own RC_16_30 table)."""

from __future__ import annotations

import json
import pathlib

import frx.numpy as fnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from pico_zorch.poseidon2.koalabear import koalabear16_params, koalabear16_perm

_GOLDEN = pathlib.Path(__file__).parent / "testdata" / "golden" / "poseidon2.json"


class KoalabearPoseidon2Test(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.golden = json.loads(_GOLDEN.read_text())

    def test_constants_match_reference(self) -> None:
        params = koalabear16_params()
        ext = self.golden["external_constants"]
        want_initial = fnp.array(ext[:4], dtype=F)
        want_terminal = fnp.array(ext[4:], dtype=F)
        self.assertTrue(
            bool(fnp.array_equal(params.external_constants_initial, want_initial))
        )
        self.assertTrue(
            bool(fnp.array_equal(params.external_constants_terminal, want_terminal))
        )
        want_internal = fnp.array(self.golden["internal_constants"], dtype=F)
        self.assertTrue(
            bool(fnp.array_equal(params.internal_constants[:, 0], want_internal))
        )

    def test_permute_byte_matches_reference(self) -> None:
        perm = koalabear16_perm()
        for vec in self.golden["vectors"]:
            state = fnp.array(vec["input"], dtype=F)
            want = fnp.array(vec["output"], dtype=F)
            got = perm.permute(state)
            self.assertTrue(
                bool(fnp.array_equal(got, want)), msg=f"vector {vec['name']}"
            )


if __name__ == "__main__":
    absltest.main()
