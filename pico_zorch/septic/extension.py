# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The degree-7 extension of KoalaBear, `SepticExtension` in Pico.

Pico's global interaction argument accumulates a curve point per chip and
observes it in the transcript, so cross-chip sums bind without an extra
commitment round. That curve lives over a *septic* extension because the
argument needs a group whose order resists the birthday attack a degree-4
extension would allow — the security of the whole cross-chip argument rests on
the extension degree, which is why it is 7 and not the 4 the FRI challenges use.

Elements are `[..., 7]` arrays of base field coefficients, low degree first:
`a[0] + a[1]·z + ... + a[6]·z^6`. Batching lives on the leading axes, so a
whole chip's worth of points multiplies in one call.

The modulus is `z^7 + 2z^6 + (p-2) = 0`, i.e.

    z^7 = 2 - 2·z^6

from Pico's `EXT_COEFFS = [2, 0, 0, 0, 0, 0, p-2]` (`p-2 = 2130706431 ≡ -2`).
Mirrors `vm/src/machine/septic/extension.rs` and `.../fields/koalabear.rs` at
pico v2.0.0.
"""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array
from zk_dtypes import koalabear_mont as F

# Degree of the extension.
DEGREE = 7

# `EXT_COEFFS`: z^7 reduces to Σ EXT_COEFFS[i]·z^i. The trailing entry is p-2,
# which is -2 — kept in Pico's canonical form so the constant is greppable
# against the reference rather than silently "simplified".
EXT_COEFFS = (2, 0, 0, 0, 0, 0, 2130706431)


def from_base(coeffs: Array) -> Array:
    """A `[..., 7]` coefficient array as a septic element (an identity, named
    so call sites read as field operations rather than array plumbing)."""
    if coeffs.shape[-1] != DEGREE:
        raise ValueError(
            f"a septic element needs {DEGREE} coefficients, got {coeffs.shape[-1]}"
        )
    return coeffs


def zero(shape: tuple[int, ...] = ()) -> Array:
    return fnp.zeros((*shape, DEGREE), dtype=F)


def one(shape: tuple[int, ...] = ()) -> Array:
    return fnp.zeros((*shape, DEGREE), dtype=F).at[..., 0].set(fnp.ones((), F))


def add(a: Array, b: Array) -> Array:
    return a + b


def sub(a: Array, b: Array) -> Array:
    return a - b


def neg(a: Array) -> Array:
    return -a


def scalar_mul(a: Array, k: Array) -> Array:
    """Multiply by a base-field scalar, broadcast over the coefficients."""
    return a * k[..., None]


def mul(a: Array, b: Array) -> Array:
    """Schoolbook multiply, then fold z^7..z^12 back down.

    The fold runs high-to-low because reducing z^k can raise the coefficient of
    z^(k-1) — going the other way would leave already-visited powers dirty.
    Unrolled rather than scanned: 13 static positions, so the whole reduction
    is straight-line and fuses into the surrounding program.
    """
    a = from_base(a)
    b = from_base(b)
    # Degree <= 12 product, as a list of [...]-shaped coefficient arrays.
    prod = [None] * (2 * DEGREE - 1)
    for i in range(DEGREE):
        for j in range(DEGREE):
            term = a[..., i] * b[..., j]
            prod[i + j] = term if prod[i + j] is None else prod[i + j] + term

    for power in range(2 * DEGREE - 2, DEGREE - 1, -1):
        coeff = prod[power]
        for offset, red in enumerate(EXT_COEFFS):
            if red == 0:
                continue
            idx = (power - DEGREE) + offset
            prod[idx] = prod[idx] + coeff * fnp.array(red, dtype=F)
        prod[power] = None

    return fnp.stack(prod[:DEGREE], axis=-1)


def square(a: Array) -> Array:
    return mul(a, a)


def cube(a: Array) -> Array:
    return mul(square(a), a)
