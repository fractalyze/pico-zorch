# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Pico's challenger as a zorch transcript flavour.

Pico's `SC_Challenger = DuplexChallenger<KoalaBear, Poseidon2KoalaBear<16>,
16, 8>` is byte-for-byte zorch's overwrite-mode `DuplexTranscript` over the
Pico permutation — same absorb mode, same back-to-front squeeze, same
low-canonical-bits PoW predicate — so binding the flavour is all it takes.
The helpers add the two Plonky3 conventions the base seam does not name:
extension sampling, and `sample_bits`.
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

    The transcript drives the permutation from eager host loops, where an
    un-jitted permute re-dispatches its few hundred field ops every call."""

    def __init__(self, inner: Poseidon2) -> None:
        self._inner = inner
        self.width: int = inner.width
        self.dtype: Any = inner.dtype
        self.has_dedicated_fusion: bool = inner.has_dedicated_fusion
        self._permute = frx.jit(inner.permute)

    def permute(self, state: Array) -> Array:
        return self._permute(state)

    # Forwarded from the wrapped permutation, which carries value equality
    # for this reason: the wrapper is a static meta_field of the transcript,
    # so identity semantics would make every fresh challenger a cache miss.
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
    c3·X³ over the quartic extension X⁴ = 3."""
    t, raw = transcript.sample(4)
    return t, reinterpret_challenge(raw, EF)


def sample_bits(
    transcript: DuplexTranscript, bits: int
) -> tuple[DuplexTranscript, Array]:
    """`CanSampleBits::sample_bits`: the low `bits` of one squeeze's
    canonical value. Masking the device array directly would read Montgomery
    bits and draw different indices."""
    t, raw = transcript.sample(1)
    canonical = lax.convert_element_type(raw, fnp.uint32)
    return t, (canonical[0] & fnp.uint32((1 << bits) - 1)).astype(fnp.int32)
