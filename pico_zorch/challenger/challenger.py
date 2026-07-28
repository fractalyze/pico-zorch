# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Pico's challenger as a zorch transcript flavour.

Pico's `SC_Challenger = DuplexChallenger<KoalaBear, Poseidon2KoalaBear<16>,
16, 8>` is byte-for-byte zorch's overwrite-mode `DuplexTranscript` over the
Pico permutation: overwrite absorb with a permute per full rate block, squeeze
pops the rate block back-to-front, and PoW checks the low canonical bits of
one squeeze. The helpers here add the two Plonky3 conventions the base seam
does not name: extension sampling as 4 base pops packed c0..c3, and
`sample_bits` as the low bits of one squeeze's canonical value.
"""

from __future__ import annotations

from typing import Any

import frx
import frx.numpy as fnp
from frx import Array, lax
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.transcript import DuplexTranscript, reinterpret_challenge

from pico_zorch.poseidon2.koalabear import koalabear16_params

RATE = 8


class JitPermutation:
    """`Permutation` wrapper with a jitted `permute`.

    The transcript drives the permutation from eager host loops where each
    un-jitted permute re-dispatches its few hundred field ops per call; one
    compile here collapses that to a single dispatch."""

    def __init__(self, inner: Poseidon2) -> None:
        self._inner = inner
        self.width: int = inner.width
        self.dtype: Any = inner.dtype
        self.has_dedicated_fusion: bool = inner.has_dedicated_fusion
        self._permute = frx.jit(inner.permute)

    def permute(self, state: Array) -> Array:
        return self._permute(state)

    # Value identity from the wrapped permutation: JitPermutation rides as a
    # static meta_field in DuplexTranscript, so without it every fresh
    # challenger would be a new jit cache key (zorch#214 convention).
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, JitPermutation):
            return NotImplemented
        return self._inner == other._inner

    def __hash__(self) -> int:
        return hash(self._inner)


def fresh_challenger() -> DuplexTranscript:
    """Pico's challenger at its initial (all-zero) state."""
    return DuplexTranscript.new(
        JitPermutation(Poseidon2(koalabear16_params())), RATE
    )


def sample_ext(transcript: DuplexTranscript) -> tuple[DuplexTranscript, Array]:
    """`sample_ext_element`: four base squeezes packed as c0 + c1·X + c2·X² +
    c3·X³ of the quartic extension (X⁴ = 3)."""
    t, raw = transcript.sample(4)
    return t, reinterpret_challenge(raw, EF)


def sample_bits(
    transcript: DuplexTranscript, bits: int
) -> tuple[DuplexTranscript, Array]:
    """`CanSampleBits::sample_bits`: the low `bits` of one squeeze's canonical
    value. Canonical, not Montgomery — the device array carries raw Montgomery
    u32, so the mask must follow a form conversion."""
    t, raw = transcript.sample(1)
    canonical = lax.convert_element_type(raw, fnp.uint32)
    return t, (canonical[0] & fnp.uint32((1 << bits) - 1)).astype(fnp.int32)


def observe_ext(transcript: DuplexTranscript, values: Array) -> DuplexTranscript:
    """`observe_ext_element`: absorb the base limbs c0..c3 in order. zorch's
    `observe` bitcast-flattens extension arrays limb-first, which is the same
    order — named here so call sites read as the reference does."""
    return transcript.observe(values)
