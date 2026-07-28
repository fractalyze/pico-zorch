# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Quotient polynomial evaluation, mirroring the reference `quotient_values`
(Plonky3 uni-stark/src/prover.rs at brevis-network/Plonky3@7fbe1908).

quotient(X) = Σ_j α^{C-1-j}·c_j(X) / Z_H(X) over the quotient coset, natural
order. The α powers run high-to-low because the reference reverses them while
its folder walks constraints first-to-last — emission order is part of the
byte contract (see `air.Air`).
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from frx import Array, lax
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.poly.univariate import powers

from pico_zorch.uni_stark.air import Air
from pico_zorch.uni_stark.domain import Coset


def quotient_values(
    air: Air,
    public_values: Array,
    trace_domain: Coset,
    quotient_domain: Coset,
    trace_on_quotient_domain: Array,
    alpha: Array,
) -> Array:
    """`[quotient_size]` extension evaluations of the quotient polynomial.

    `trace_on_quotient_domain` is `[quotient_size, width]` in natural coset
    order; the "next" row for point i is row i + 2^(coset log gap), wrapping.
    """
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
    """`[n]` extension evaluations -> `[n, 4]` base columns (limb order
    c0..c3), the reference's `flatten_to_base`."""
    return lax.bitcast_convert_type(quotient, F)
