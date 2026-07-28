# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The AIR seam for the uni-stark stage.

An AIR exposes its constraint evaluations directly instead of a builder DSL:
`eval` returns the constraint values in emission order, and that order is part
of the byte contract — the reference folds `Σ_j α^{C-1-j}·c_j` prover-side and
Horner (`acc·α + c_j`) verifier-side, which only agree when both sides emit
the same order the reference AIR's `eval` does.

`eval` is written against broadcastable arrays so one definition serves both
roles: the prover calls it with `[N]`-shaped rows and selectors over the
quotient coset; the verifier with extension scalars at zeta.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from frx import Array


class Air(Protocol):
    width: int
    # The max degree_multiple of any constraint (selector degree included),
    # the reference's `constraint_degree`; drives the quotient chunk count
    # via log_quotient_degree = ceil(log2(constraint_degree - 1)).
    constraint_degree: int

    def eval(
        self,
        local: Array,
        next: Array,
        is_first_row: Array,
        is_last_row: Array,
        is_transition: Array,
        public_values: Array,
    ) -> Sequence[Array]:
        """Constraint values c_0..c_{C-1} in reference emission order.

        `local`/`next` are `[..., width]`; the selectors broadcast against
        their leading axes."""
        ...
