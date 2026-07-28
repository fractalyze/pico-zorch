# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Two-adic multiplicative cosets and their Lagrange selectors.

The trace lives on a multiplicative subgroup H = <g> of order 2^k, which is
what makes a STARK cheap: the vanishing polynomial is X^n − 1, the next row
is one generator step, and every domain the protocol needs (the blown-up
LDE, the disjoint quotient coset, each chunk's sub-coset) is a shift of a
subgroup the NTT already knows how to evaluate on.

Mirrors the reference `TwoAdicMultiplicativeCoset` (Plonky3
commit/src/domain.rs at brevis-network/Plonky3@7fbe1908). Natural domain
order throughout — the bit-reversed committed layout is
`pico_zorch.commit`'s concern."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import frx.numpy as fnp
from frx import Array

from zorch.coding.reed_solomon import eval_domain
from zorch.poly.univariate import powers


def subgroup_gen(dtype: Any, log_n: int) -> Array:
    """two_adic_generator(log_n), read off the same NTT the encoder uses so
    the two cannot disagree about the root."""
    if log_n == 0:
        return fnp.ones((), dtype)
    return eval_domain(dtype, 1 << log_n)[1]


@dataclass(frozen=True)
class Coset:
    """shift·<g_{2^log_n}>, the reference's `TwoAdicMultiplicativeCoset`."""

    log_n: int
    shift: Array

    @property
    def size(self) -> int:
        return 1 << self.log_n

    def gen(self) -> Array:
        return subgroup_gen(self.shift.dtype, self.log_n)

    def points(self) -> Array:
        """Natural order: shift·g^i."""
        return eval_domain(self.shift.dtype, self.size, shift=self.shift)

    def next_point(self, x: Array) -> Array:
        return x * self.gen().astype(x.dtype)

    def create_disjoint_domain(self, min_size: int, generator: Array) -> Coset:
        return Coset(max(min_size - 1, 0).bit_length(), self.shift * generator)

    def split_domains(self, num_chunks: int) -> list[Coset]:
        log_chunks = num_chunks.bit_length() - 1
        g = self.gen()
        shifts, acc = [], fnp.ones((), self.shift.dtype)
        for _ in range(num_chunks):
            shifts.append(self.shift * acc)
            acc = acc * g
        return [Coset(self.log_n - log_chunks, s) for s in shifts]

    def split_evals(self, num_chunks: int, evals: Array) -> list[Array]:
        """Chunk i takes rows i, i+k, i+2k, … — strided, not contiguous, so
        each chunk lands on one of `split_domains`' shifted cosets."""
        return [evals[i::num_chunks] for i in range(num_chunks)]

    def zp_at_point(self, point: Array) -> Array:
        """Z_H(point) = (point/shift)^n − 1, unnormalized as the reference
        leaves it."""
        shift_inv = (fnp.ones((), self.shift.dtype) / self.shift).astype(point.dtype)
        x = point * shift_inv
        for _ in range(self.log_n):
            x = x * x
        return x - fnp.ones((), point.dtype)

    def selectors_at_point(self, point: Array) -> dict[str, Array]:
        one = fnp.ones((), point.dtype)
        shift_inv = (fnp.ones((), self.shift.dtype) / self.shift).astype(point.dtype)
        unshifted = point * shift_inv
        z_h = unshifted
        for _ in range(self.log_n):
            z_h = z_h * z_h
        z_h = z_h - one
        g_last = (fnp.ones((), self.shift.dtype) / self.gen()).astype(point.dtype)
        return {
            "is_first_row": z_h / (unshifted - one),
            "is_last_row": z_h / (unshifted - g_last),
            "is_transition": unshifted - g_last,
            "inv_zeroifier": one / z_h,
        }

    def selectors_on_coset(self, coset: Coset) -> dict[str, Array]:
        """Boundary and transition selectors over `coset`, unnormalized as
        the reference leaves them.

        These are what let one polynomial identity carry row-dependent
        constraints: `is_first_row`/`is_last_row` are Z_H divided by the
        single root being isolated, so they vanish on all of H but that row.

        Z_H cycles with period 2^rate_bits over the coset, since
        Z_H(s·g_N^i) = s^n·w_r^(i mod 2^r) − 1 with w_r the rate-bit
        generator."""
        dtype = self.shift.dtype
        one = fnp.ones((), dtype)
        rate_bits = coset.log_n - self.log_n

        s_pow_n = coset.shift
        for _ in range(self.log_n):
            s_pow_n = s_pow_n * s_pow_n
        w_r = subgroup_gen(dtype, rate_bits)
        z_evals = s_pow_n * powers(w_r, 1 << rate_bits) - one

        xs = coset.points()
        reps = coset.size >> rate_bits
        z_cycled = fnp.tile(z_evals, reps)
        # Invert before tiling: `one / z_cycled` reads the same and costs a
        # full coset-length inversion instead of 2^rate_bits of them.
        inv_z_cycled = fnp.tile(one / z_evals, reps)

        g_trace = self.gen()
        first_point = one
        last_point = one / g_trace  # g^{n-1} = g^{-1}

        # Both point selectors off one inversion: 1/d₀ = d₁/(d₀d₁).
        d_first = xs - first_point
        d_last = xs - last_point
        inv_prod = one / (d_first * d_last)
        return {
            "is_first_row": z_cycled * inv_prod * d_last,
            "is_last_row": z_cycled * inv_prod * d_first,
            "is_transition": d_last,
            "inv_zeroifier": inv_z_cycled,
        }
