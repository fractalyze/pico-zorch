# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Quotient polynomial evaluation, mirroring the reference `quotient_values`
(Plonky3 uni-stark/src/prover.rs at brevis-network/Plonky3@7fbe1908).

quotient(X) = Σ_j α^{C-1-j}·c_j(X) / Z_H(X), evaluated pointwise over the
quotient coset in natural order. Working in evaluations rather than
coefficients is what makes this linear-time: Z_H is a fixed cheap function
on a coset disjoint from the trace domain, so the division is one field
multiply per point by a precomputed inverse.

The α powers run high-to-low because the reference reverses them while its
folder walks constraints first-to-last, which is why an AIR's emission
order is part of the byte contract (`air.Air`).
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from frx import Array
from zk_dtypes import koalabearx4_mont as EF

from zorch.pcs.fold import to_base_field
from zorch.poly.univariate import powers

from pico_zorch.uni_stark.air import Air
from pico_zorch.uni_stark.domain import Coset


def log_quotient_degree(constraint_degree: int) -> int:
    """ceil(log2(constraint_degree − 1)) — the reference's quotient blowup.

    A degree-d constraint over a degree-(n−1) trace has degree ~d·n, so
    dividing by Z_H (degree n) leaves Q of degree ~(d−1)·n: it needs d−1
    chunks of trace degree, rounded up to a power of two."""
    return (max(constraint_degree - 1, 1) - 1).bit_length()


def quotient_values(
    air: Air,
    public_values: Array,
    trace_domain: Coset,
    quotient_domain: Coset,
    trace_on_quotient_domain: Array,
    alpha: Array,
) -> Array:
    """`[quotient_size]` extension evaluations of the quotient polynomial.

    A trace row's successor sits `2^(coset log gap)` rows away on the
    quotient coset, and wraps — the coset is `2^log_qd` interleaved copies
    of the trace domain."""
    sels = trace_domain.selectors_on_coset(quotient_domain)
    n = quotient_domain.size
    next_step = 1 << (quotient_domain.log_n - trace_domain.log_n)

    local = trace_on_quotient_domain
    next_rows = fnp.take(
        trace_on_quotient_domain,
        fnp.asarray((np.arange(n) + next_step) % n),
        axis=0,
    )

    constraints = air.eval(
        local,
        next_rows,
        sels["is_first_row"],
        sels["is_last_row"],
        sels["is_transition"],
        public_values,
    )
    count = len(constraints)

    alpha_powers = powers(alpha, count)  # [alpha^0 .. alpha^{count-1}]
    acc = fnp.zeros((n,), dtype=EF)
    for j, c in enumerate(constraints):
        acc = acc + alpha_powers[count - 1 - j] * c.astype(EF)
    return acc * sels["inv_zeroifier"].astype(EF)


def flatten_to_base(quotient: Array) -> Array:
    """`[n]` extension evaluations -> `[n, 4]` base columns — the reference's
    `flatten_to_base`. The column axis is added first because zorch folds
    limbs into an existing trailing axis."""
    return to_base_field(quotient[:, None])
