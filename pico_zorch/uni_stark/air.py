# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The AIR seam for the uni-stark stage.

An AIR exposes its constraint evaluations directly instead of a builder DSL.
Emission order is part of the byte contract: the reference folds
`Σ_j α^{C-1-j}·c_j` prover-side and Horner (`acc·α + c_j`) verifier-side,
which agree only when both walk the constraints in the reference AIR's order.

`eval` takes broadcastable arrays so one definition serves both roles — the
prover passes `[N]`-shaped coset rows, the verifier extension scalars at ζ.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from frx import Array


class Air(Protocol):
    width: int
    # The reference's `constraint_degree`: the max degree_multiple over all
    # constraints, selectors included. Sets the quotient chunk count, so an
    # understated value silently produces an unprovable claim.
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

        `local`/`next` are `[..., width]`; selectors broadcast against their
        leading axes."""
        ...
