# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The septic curve Pico's global interaction argument accumulates over.

    y^2 = x^3 + 2x + 611*z^5      over KoalaBear^7

A chip's global cumulative sum is a point on this curve, and the machine
transcript observes its coordinates directly — so the point *is* the digest,
and adding two chips' contributions is a curve addition.

Points are `([..., 7], [..., 7])` coordinate pairs, batching on the leading
axes. The addition is `add_incomplete`, matching the reference: it is the plain
chord formula with no special case for `P == Q` or `P == -Q`. That is sound in
Pico's use because the accumulated points are pseudorandom (each derives from a
hashed interaction), so a collision is a soundness event rather than a case to
branch on — and a branch here would not be constant-shape anyway.

Mirrors `vm/src/machine/septic/curve.rs` at pico v2.0.0.
"""

from __future__ import annotations

import frx.numpy as fnp
from frx import Array
from zk_dtypes import koalabear_mont as F

from pico_zorch.septic import extension as ext

# The `611*z^5` term of the curve equation.
_B_COEFF_INDEX = 5
_B_COEFF_VALUE = 611


def curve_formula(x: Array) -> Array:
    """The right-hand side `x^3 + 2x + 611*z^5`.

    A function of x alone, so it is well defined off the curve too — which is
    what makes it usable as an on-curve predicate.
    """
    b = fnp.zeros((*x.shape[:-1], ext.DEGREE), dtype=F)
    b = b.at[..., _B_COEFF_INDEX].set(fnp.array(_B_COEFF_VALUE, dtype=F))
    return ext.cube(x) + x * fnp.array(2, dtype=F) + b


def is_on_curve(x: Array, y: Array) -> Array:
    """`y^2 == curve_formula(x)`, elementwise over the leading axes."""
    return fnp.all(ext.square(y) == curve_formula(x), axis=-1)


def neg(x: Array, y: Array) -> tuple[Array, Array]:
    return x, -y


def add_incomplete(
    p: tuple[Array, Array], q: tuple[Array, Array]
) -> tuple[Array, Array]:
    """The chord addition, with no doubling or inverse-point case.

    slope = (y2 - y1)/(x2 - x1);  x3 = slope^2 - x1 - x2;  y3 = slope*(x1 - x3) - y1

    Division is by the septic inverse, so this needs `x1 != x2` — see the
    module docstring for why the reference does not guard that.
    """
    x1, y1 = p
    x2, y2 = q
    slope = ext.mul(ext.sub(y2, y1), inverse(ext.sub(x2, x1)))
    x3 = ext.sub(ext.sub(ext.square(slope), x1), x2)
    y3 = ext.sub(ext.mul(slope, ext.sub(x1, x3)), y1)
    return x3, y3


def double(p: tuple[Array, Array]) -> tuple[Array, Array]:
    """The tangent formula: slope = (3x^2 + 2)/(2y), the derivative of the
    curve equation (the `611*z^5` term is constant and drops out)."""
    x, y = p
    three = fnp.array(3, dtype=F)
    two = fnp.array(2, dtype=F)
    numer = ext.square(x) * three + fnp.zeros_like(x).at[..., 0].set(two)
    slope = ext.mul(numer, inverse(y * two))
    x3 = ext.sub(ext.square(slope), x * two)
    y3 = ext.sub(ext.mul(slope, ext.sub(x, x3)), y)
    return x3, y3


def inverse(a: Array) -> Array:
    """The multiplicative inverse in KoalaBear^7, by Fermat: a^(q-2) where
    q = p^7.

    Computed as a square-and-multiply over the exponent's bits. The exponent is
    a compile-time constant, so the chain unrolls and no data-dependent control
    flow enters the program.
    """
    exponent = _ORDER - 2
    result = ext.one(a.shape[:-1])
    base = a
    while exponent:
        if exponent & 1:
            result = ext.mul(result, base)
        base = ext.square(base)
        exponent >>= 1
    return result


# |F_{p^7}| with p = KoalaBear's modulus.
_P = 2130706433
_ORDER = _P**ext.DEGREE
