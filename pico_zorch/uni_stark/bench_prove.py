# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Per-stage wall-clock for the uni-stark prover.

Mirrors the composite's call order so a stage's time covers exactly its
claim reduction, and re-checks the golden commitments when the size matches
the fixture — a timed run is then also a byte-match run. `--trace_dir`
captures an frx profiler trace of the last pass.

How to run and read it, including which pass to trust: docs/development.md.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import pathlib
import time
from typing import Any

import frx
import frx.numpy as fnp
import numpy as np
from frx import lax
from zk_dtypes import koalabear_mont as F

from pico_zorch.challenger.challenger import fresh_challenger
from pico_zorch.commit.pcs_commit import commit_pcs
from pico_zorch.poseidon2.koalabear import koalabear16_merkle
from pico_zorch.uni_stark.fri_stage import FriOpener
from pico_zorch.uni_stark.prover import bind_instance
from pico_zorch.uni_stark.quotient_stage import QuotientProver
from pico_zorch.uni_stark.testing.fib_air import FibonacciAir, generate_trace_rows
from pico_zorch.uni_stark.types import (
    FriOpeningWitness,
    FriParams,
    QuotientClaim,
    QuotientWitness,
)

_GOLDEN = pathlib.Path(__file__).parent / "testdata" / "golden" / "fib_prove.json"


@dataclasses.dataclass(frozen=True)
class _WideFibAir(FibonacciAir):
    """Fibonacci in columns 0-1, zero padding beyond: width sweeps without
    the constraint set (or its degree) changing."""

    width: int = 2


def _block(obj: Any) -> None:
    """`block_until_ready` over a dataclass tree. Proof types are not
    registered pytrees, so blocking on the result itself returns while the
    device is still working and every timing below it reads short."""
    if hasattr(obj, "block_until_ready"):
        obj.block_until_ready()
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            _block(getattr(obj, f.name))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _block(item)


def _timed(name: str, fn):
    start = time.perf_counter()
    out = fn()
    _block(out)
    print(f"  [phase {name}] {(time.perf_counter() - start) * 1e3:.1f}ms")
    return out


def _canonical(x) -> np.ndarray:
    return np.asarray(lax.convert_element_type(x, fnp.uint32))


def _check_golden(trace_root, quotient) -> None:
    golden = json.loads(_GOLDEN.read_text())
    want = golden["proof"]["commitments"]
    ok = (_canonical(trace_root) == np.array(want["trace"]["value"])).all()
    ok &= (
        _canonical(quotient.reduction_proof.quotient_root)
        == np.array(want["quotient_chunks"]["value"])
    ).all()
    print(f"  golden commitments: {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit(1)


def main() -> None:
    _enable_persistent_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree_bits", type=int, action="append")
    parser.add_argument("--n_cols", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--num_queries", type=int, default=84)
    parser.add_argument("--trace_dir", type=str, default="")
    args = parser.parse_args()
    if args.n_cols < 2:
        raise ValueError("n_cols must be >= 2 (Fibonacci occupies columns 0-1)")

    params = FriParams(num_queries=args.num_queries)
    _, _, tree = koalabear16_merkle()

    for degree_bits in args.degree_bits or [10]:
        n = 1 << degree_bits
        fib = np.asarray(_canonical(generate_trace_rows(0, 1, n)))
        trace = fnp.array(
            np.pad(fib, ((0, 0), (0, args.n_cols - 2))), dtype=F
        )
        air = _WideFibAir(width=args.n_cols)
        pv = fnp.array([0, 1, int(fib[-1, 1])], dtype=F)
        is_golden_config = (
            degree_bits == 3 and args.n_cols == 2 and params == FriParams()
        )
        print(
            f"degree_bits={degree_bits} n_cols={args.n_cols} "
            f"queries={params.num_queries} backend={frx.default_backend()}"
        )

        for run in range(args.runs):
            print(f" pass {run + 1}:")
            ctx = (
                frx.profiler.trace(args.trace_dir, create_perfetto_trace=True)
                if args.trace_dir and run == args.runs - 1
                else contextlib.nullcontext()
            )
            start = time.perf_counter()
            with ctx:
                with frx.profiler.TraceAnnotation("TraceCommit"):
                    trace_root, trace_data = _timed(
                        "TraceCommit",
                        lambda: commit_pcs(tree, [trace], params.log_blowup),
                    )
                t = bind_instance(fresh_challenger(), degree_bits, trace_root, pv)
                with frx.profiler.TraceAnnotation("Quotient"):
                    quotient = _timed(
                        "Quotient",
                        lambda: QuotientProver(tree, params).prove(
                            QuotientClaim(air, pv, degree_bits, trace_root),
                            QuotientWitness(trace, trace_data),
                            t,
                        ),
                    )
                with frx.profiler.TraceAnnotation("FriOpen"):
                    opening = _timed(
                        "FriOpen",
                        lambda: FriOpener(tree, params).prove(
                            quotient.reduced_claim,
                            FriOpeningWitness(
                                trace, trace_data, quotient.reduction_proof.data
                            ),
                            quotient.transcript,
                        ),
                    )
                del opening
            print(f"  [total] {(time.perf_counter() - start) * 1e3:.1f}ms")
            if is_golden_config:
                _check_golden(trace_root, quotient)


def _enable_persistent_cache() -> None:
    """A cold pass compiles for minutes; the cache pays that once per
    (program, jaxlib) across invocations."""
    cache = os.environ.get(
        "FRX_COMPILATION_CACHE_DIR",
        os.path.expanduser("~/.cache/pico-zorch-frxcc"),
    )
    os.makedirs(cache, exist_ok=True)
    frx.config.update("jax_compilation_cache_dir", cache)


if __name__ == "__main__":
    main()
