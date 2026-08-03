# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Sub-phase profile of the FRI opening stage.

Replays the stage's step sequence with a device sync after each sub-phase,
so a sub-phase's wall time is its own rather than the next sync's.
`--trace_dir` also captures an frx profiler trace of the warm pass; reading
device-busy against wall inside a sub-phase is what distinguishes kernel
cost from dispatch cost.

Invocation and environment: docs/development.md.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time

import frx
import frx.numpy as fnp
import numpy as np
from frx import lax
from zk_dtypes import koalabear_mont as F

from zorch.coding.reed_solomon import BitReversedReedSolomon

from pico_zorch.challenger.challenger import fresh_challenger
from pico_zorch.uni_stark.fri import (
    GENERATOR,
    FriOpener,
    _lde_code,
    _open_head,
    fold_chain,
    open_batch,
    sample_query_indices,
)
from pico_zorch.poseidon2.koalabear import koalabear16_merkle
from pico_zorch.uni_stark.bench_prove import _block, _WideFibAir
from pico_zorch.uni_stark.prover import bind_instance
from pico_zorch.uni_stark.quotient_stage import QuotientProver
from pico_zorch.uni_stark.testing.fib_air import generate_trace_rows
from pico_zorch.uni_stark.types import (
    CommitData,
    FriParams,
    QuotientClaim,
    QuotientWitness,
)


class _Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def lap(self, label: str, result=None) -> None:
        _block(result)
        now = time.perf_counter()
        print(f"  {label:<28}{(now - self.t0) * 1e3:8.1f}ms")
        self.t0 = now


def _one_pass(pcs, trace, trace_data, claim, quotient, t, n, n_cols):
    tm = _Timer()
    params = pcs.params
    with frx.profiler.TraceAnnotation("open_head"):
        trace_local, trace_next, chunk_values, ro, tt = _open_head(
            trace,
            tuple(quotient.chunks),
            tuple(d.shift for d in quotient.qc_domains),
            trace_data.leaves,
            quotient.quotient_data.leaves,
            claim.zeta,
            claim.zeta_next,
            t,
            n_cols,
            len(quotient.chunks),
            _lde_code(n, params.log_blowup).domain(),
        )
    tm.lap("open_head(ood+obs+ro)", (ro, tt.state.sponge_state))

    code = BitReversedReedSolomon(n, 1 << params.log_blowup, F)
    with frx.profiler.TraceAnnotation("fold_chain"):
        final_poly, roots, layers, tt = fold_chain(
            pcs.tree, code, params.log_blowup, ro, tt
        )
    phase_data = [
        CommitData((leaves,), leaves, list(digest_layers))
        for leaves, digest_layers in layers
    ]
    tm.lap(f"fold_chain[{len(phase_data)} layers]", (final_poly, roots, tt.state.sponge_state))

    with frx.profiler.TraceAnnotation("grind"):
        tt, pow_witness = tt.grind(params.proof_of_work_bits)
    tm.lap("grind(16)", pow_witness)

    tt, idx = sample_query_indices(
        tt, (n - 1).bit_length() + params.log_blowup, params.num_queries
    )
    # Block on the indices themselves: they now stay on device, so without
    # this the sample would be attributed to the following open_batch.
    tm.lap("sample_indices", idx)

    with frx.profiler.TraceAnnotation("open_batch"):
        opens = [
            open_batch(pcs.tree, trace_data, idx),
            open_batch(pcs.tree, quotient.quotient_data, idx),
        ] + [
            open_batch(pcs.tree, data, idx >> (layer + 1))
            for layer, data in enumerate(phase_data)
        ]
    tm.lap(f"open_batch[{len(opens)} trees]", opens)


def profile(degree_bits: int, n_cols: int, params: FriParams, trace_dir: str) -> None:
    n = 1 << degree_bits
    fib = np.asarray(lax.convert_element_type(generate_trace_rows(0, 1, n), fnp.uint32))
    trace = fnp.array(np.pad(fib, ((0, 0), (0, n_cols - 2))), dtype=F)
    air = _WideFibAir(width=n_cols)
    pv = fnp.array([0, 1, int(fib[-1, 1])], dtype=F)
    _, _, tree = koalabear16_merkle()
    pcs = FriOpener(tree, params)

    trace_root, trace_data = pcs.commit([trace], shifts=[GENERATOR])
    t = bind_instance(fresh_challenger(), degree_bits, trace_root, pv)
    q = QuotientProver(pcs).prove(
        QuotientClaim(air, pv, degree_bits, trace_root),
        QuotientWitness(trace, trace_data),
        t,
    )
    claim, quotient = q.reduced_claim, q.reduction_proof.data
    _block(q)

    print(
        f"FriOpen sub-phases, degree_bits={degree_bits} n_cols={n_cols} "
        f"queries={params.num_queries} backend={frx.default_backend()}"
    )
    for run in range(2):
        print(f" pass {run + 1}:")
        ctx = (
            frx.profiler.trace(trace_dir, create_perfetto_trace=True)
            if run == 1 and trace_dir
            else contextlib.nullcontext()
        )
        with ctx:
            _one_pass(
                pcs, trace, trace_data, claim, quotient, q.transcript, n, n_cols
            )
    if trace_dir:
        print(f"profiler trace written to {trace_dir}")


def main() -> None:
    _enable_persistent_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree_bits", type=int, default=16)
    parser.add_argument("--n_cols", type=int, default=32)
    parser.add_argument("--num_queries", type=int, default=84)
    parser.add_argument("--trace_dir", type=str, default="")
    args = parser.parse_args()
    profile(
        args.degree_bits,
        args.n_cols,
        FriParams(num_queries=args.num_queries),
        args.trace_dir,
    )


def _enable_persistent_cache() -> None:
    """Cold passes recompile for minutes; the persistent cache pays that once
    per (program, jaxlib) across invocations."""
    cache = os.environ.get(
        "FRX_COMPILATION_CACHE_DIR",
        os.path.expanduser("~/.cache/pico-zorch-frxcc"),
    )
    os.makedirs(cache, exist_ok=True)
    frx.config.update("jax_compilation_cache_dir", cache)


if __name__ == "__main__":
    main()
