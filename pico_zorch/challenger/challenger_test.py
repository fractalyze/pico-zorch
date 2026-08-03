# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Byte-match of the challenger against the reference DuplexChallenger script
(golden/ drives Pico's DuplexChallenger<KoalaBear, Poseidon2KoalaBear<16>,
16, 8> through the same observe/sample/sample_bits/grind sequence)."""

from __future__ import annotations

import json
import pathlib

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from frx import lax
from zk_dtypes import koalabear_mont as F

from pico_zorch.challenger.challenger import fresh_challenger

_GOLDEN = pathlib.Path(__file__).parent / "testdata" / "golden" / "challenger.json"


def _canonical(x) -> np.ndarray:
    return np.asarray(lax.convert_element_type(x, fnp.uint32))


class ChallengerTest(absltest.TestCase):
    def test_script_byte_matches_reference(self) -> None:
        steps = json.loads(_GOLDEN.read_text())["steps"]
        t = fresh_challenger()

        t = t.observe(fnp.array([1, 2, 3], dtype=F))
        t, s1 = t.sample(1)
        t, s2 = t.sample(1)
        got = np.concatenate([_canonical(s1), _canonical(s2)])
        self.assertEqual(list(got), steps[0]["out"], msg=steps[0]["op"])

        t = t.observe(fnp.array(list(range(100, 111)), dtype=F))
        t, e1 = t.sample_ext()
        got_ext = _canonical(lax.bitcast_convert_type(e1, fnp.uint32)).reshape(-1)
        # bitcast gives Montgomery limbs; convert each limb via a field view.
        limbs = lax.bitcast_convert_type(e1, F).reshape(-1)
        self.assertEqual(list(_canonical(limbs)), steps[1]["out"], msg=steps[1]["op"])
        del got_ext

        got_bits = []
        for bits in (4, 16, 24):
            t, b = t.sample_bits(bits)
            got_bits.append(int(b))
        self.assertEqual(got_bits, steps[2]["out"], msg=steps[2]["op"])

        # grind() returns the transcript already advanced through
        # check_witness, mirroring the reference's grind-side assert.
        t, witness = t.grind(8)
        self.assertEqual(int(_canonical(witness)[()]), steps[3]["out"], msg=steps[3]["op"])

        t, tail = t.sample(1)
        self.assertEqual(int(_canonical(tail)[0]), steps[4]["out"], msg=steps[4]["op"])


if __name__ == "__main__":
    absltest.main()
