# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Pico's Fiat-Shamir transcript.

Pico's `SC_Challenger = DuplexChallenger<KoalaBear, Poseidon2KoalaBear<16>,
16, 8>` uses the same overwrite-mode duplex state machine as zorch, but the
protocol owns its concrete transcript type and Fiat-Shamir cadence. Challenges
are drawn through zorch's `ChallengePolicy`, which reads the limb count off the
degree ratio; `sample_bits` and the proof-of-work absorption path retain Pico's
Plonky3 semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import frx.numpy as fnp
from frx import Array, lax
from frx.tree_util import register_dataclass
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.challenge import ChallengePolicy
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.transcript import DuplexState, DuplexTranscript

from pico_zorch.poseidon2.koalabear import koalabear16_params

RATE = 8

# Pico draws every challenge in the quartic extension; naming the field here
# names the soundness floor the whole scheme rests on.
CHALLENGE = ChallengePolicy(EF)


@register_dataclass
@dataclass(frozen=True)
class PicoTranscript:
    """Pico's transcript over its Poseidon2-KoalaBear permutation.

    The protocol owns this public type and delegates the generic overwrite-mode
    state machine to zorch. Composition keeps Pico independent of protected
    ``DuplexTranscript`` methods and makes protocol-specific cadence explicit.
    """

    _duplex: DuplexTranscript

    @classmethod
    def new(cls) -> PicoTranscript:
        permutation = Poseidon2(koalabear16_params())
        return cls(DuplexTranscript.new(permutation, RATE))

    @property
    def state(self) -> DuplexState:
        return self._duplex.state

    @property
    def field(self) -> Any:
        return self._duplex.field

    @property
    def has_dedicated_fusion(self) -> bool:
        return self._duplex.has_dedicated_fusion

    def observe(self, values: Array) -> PicoTranscript:
        return PicoTranscript(self._duplex.observe(values))

    def sample(self, n: int = 1) -> tuple[PicoTranscript, Array]:
        duplex, samples = self._duplex.sample(n)
        return PicoTranscript(duplex), samples

    def observe_and_sample(
        self, values: Array, n: int = 1
    ) -> tuple[PicoTranscript, Array]:
        duplex, samples = self._duplex.observe_and_sample(values, n)
        return PicoTranscript(duplex), samples

    def sample_ext(self) -> tuple[PicoTranscript, Array]:
        """Draw one quartic-extension challenge using Pico's cadence."""
        transcript, challenge = CHALLENGE.sample(self)
        return transcript, challenge

    def sample_bits(self, bits: int) -> tuple[PicoTranscript, Array]:
        """Draw the canonical low bits of one base-field squeeze."""
        transcript, samples = self.sample_bits_many(bits, 1)
        return transcript, samples[0]

    def sample_bits_many(
        self, bits: int, count: int
    ) -> tuple[PicoTranscript, Array]:
        """Draw canonical low bits from ``count`` consecutive squeezes.

        Field arrays contain Montgomery representations, so masking them
        before canonical conversion would silently select different queries.
        """
        transcript, raw = self.sample(count)
        canonical = lax.convert_element_type(raw, fnp.uint32)
        mask = fnp.uint32((1 << bits) - 1)
        return transcript, (canonical & mask).astype(fnp.int32)

    def check_witness(
        self, witness: Array, *, pow_bits: int
    ) -> tuple[PicoTranscript, Array]:
        duplex, ok = self._duplex.check_witness(witness, pow_bits=pow_bits)
        return PicoTranscript(duplex), ok

    def grind(self, pow_bits: int) -> tuple[PicoTranscript, Array]:
        duplex, witness = self._duplex.grind(pow_bits)
        return PicoTranscript(duplex), witness


def fresh_challenger() -> PicoTranscript:
    """Pico's challenger at its initial (all-zero) state."""
    return PicoTranscript.new()
