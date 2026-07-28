# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The Fibonacci AIR from the reference's uni-stark test suite (fib_air.rs),
constraint emission order preserved — it feeds the α folding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import frx.numpy as fnp
import numpy as np
from frx import Array
from zk_dtypes import koalabear_mont as F


@dataclass(frozen=True)
class FibonacciAir:
    width: int = 2
    # is_first/is_last/is_transition selectors are degree-1, each constraint
    # is degree-1 in the trace: degree_multiple 2 throughout.
    constraint_degree: int = 2

    def eval(
        self,
        local: Array,
        next: Array,
        is_first_row: Array,
        is_last_row: Array,
        is_transition: Array,
        public_values: Array,
    ) -> Sequence[Array]:
        a, b, x = public_values[0], public_values[1], public_values[2]
        left, right = local[..., 0], local[..., 1]
        nleft, nright = next[..., 0], next[..., 1]
        return [
            is_first_row * (left - a),
            is_first_row * (right - b),
            is_transition * (right - nleft),
            is_transition * (left + right - nright),
            is_last_row * (right - x),
        ]


def generate_trace_rows(a: int, b: int, n: int) -> Array:
    rows = np.zeros((n, 2), dtype=np.uint64)
    rows[0] = (a, b)
    p = 2130706433
    for i in range(1, n):
        rows[i, 0] = rows[i - 1, 1]
        rows[i, 1] = (rows[i - 1, 0] + rows[i - 1, 1]) % p
    return fnp.array(rows.astype(np.uint32), dtype=F)
