# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Sub-phase profile of FriOpener.prove — where the wall clock actually goes.

Replays the stage's exact step sequence with a device sync after each
sub-phase; `--trace_dir` additionally wraps the warm pass in the frx
profiler (perfetto trace enabled) for a per-kernel device timeline.

    FRX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=<idx> \\
        bazel run //pico_zorch/uni_stark:profile_fri_open -- \\
        --degree_bits=16 --trace_dir=/tmp/friopen-trace
"""

from __future__ import annotations

import argparse
import contextlib
import time

import frx
import frx.numpy as fnp
import numpy as np
from frx import lax
from zk_dtypes import koalabear_mont as F

from zorch.coding.reed_solomon import BitReversedReedSolomon

from pico_zorch.challenger.challenger import fresh_challenger, sample_ext
from pico_zorch.commit.pcs_commit import commit_pcs
from pico_zorch.poseidon2.koalabear import koalabear16_merkle
from pico_zorch.uni_stark.bench_prove import _block, _WideFibAir
from pico_zorch.uni_stark.fri_stage import (
    _bitrev_lde_domain,
    _commit_pairs,
    _fold,
    _opening_pos,
    eval_matrix_at,
    open_batch,
    reduced_openings,
    sample_query_indices,
)
from pico_zorch.uni_stark.prover import bind_instance
from pico_zorch.uni_stark.quotient_stage import QuotientProver
from pico_zorch.uni_stark.testing.fib_air import generate_trace_rows
from pico_zorch.uni_stark.types import FriParams, QuotientClaim, QuotientWitness


class _Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def lap(self, label: str, result=None) -> None:
        _block(result)
        now = time.perf_counter()
        print(f"  {label:<28}{(now - self.t0) * 1e3:8.1f}ms")
        self.t0 = now


def _one_pass(tree, trace, trace_data, claim, quotient, t, n, n_cols, params):
    tm = _Timer()
    one = fnp.ones((), F)

    with frx.profiler.TraceAnnotation("ood_evals"):
        trace_local = eval_matrix_at(trace, one, claim.zeta)
        trace_next = eval_matrix_at(trace, one, claim.zeta_next)
        chunk_values = fnp.stack(
            [
                eval_matrix_at(c, d.shift, claim.zeta)
                for c, d in zip(quotient.chunks, quotient.qc_domains)
            ]
        )
    tm.lap("ood_evals", (trace_local, trace_next, chunk_values))

    opened = fnp.concatenate([trace_local, trace_next, chunk_values.reshape(-1)])
    with frx.profiler.TraceAnnotation("observe_opened"):
        tt = t.observe(opened)
        tt, alpha_fri = sample_ext(tt)
    tm.lap(f"observe_opened[{opened.shape[0]}]", (tt.state.sponge_state, alpha_fri))

    lde_height = n << params.log_blowup
    with frx.profiler.TraceAnnotation("reduced_openings"):
        ro = reduced_openings(
            fnp.concatenate(
                [trace_data.leaves, trace_data.leaves, quotient.quotient_data.leaves],
                axis=1,
            ),
            opened,
            fnp.stack([claim.zeta, claim.zeta_next]),
            _opening_pos(n_cols, len(quotient.chunks)),
            alpha_fri,
            _bitrev_lde_domain(lde_height),
        )
    tm.lap("reduced_openings", ro)

    code = BitReversedReedSolomon(n, 1 << params.log_blowup, F)
    folded = ro
    phase_data = []
    commit_ms = transcript_ms = fold_ms = 0.0
    with frx.profiler.TraceAnnotation("fold_chain"):
        while folded.shape[0] > (1 << params.log_blowup):
            s0 = time.perf_counter()
            pairs_base = lax.bitcast_convert_type(code.pair_leaves(folded), F)
            leaves = pairs_base.reshape(-1, 8)
            root, digest_layers = _commit_pairs(tree, leaves)
            from pico_zorch.commit.pcs_commit import CommitData
            data = CommitData((leaves,), leaves, digest_layers)
            _block((root, data))
            s1 = time.perf_counter()
            tt = tt.observe(root)
            tt, beta = sample_ext(tt)
            _block(beta)
            s2 = time.perf_counter()
            folded = _fold(code, folded, beta)
            folded.block_until_ready()
            s3 = time.perf_counter()
            commit_ms += (s1 - s0) * 1e3
            transcript_ms += (s2 - s1) * 1e3
            fold_ms += (s3 - s2) * 1e3
            phase_data.append(data)
    print(f"  fold_chain[{len(phase_data)} layers]")
    print(f"    commits                 {commit_ms:8.1f}ms")
    print(f"    transcript              {transcript_ms:8.1f}ms")
    print(f"    folds                   {fold_ms:8.1f}ms")
    tm.t0 = time.perf_counter()
    tt = tt.observe(folded[0])
    tm.lap("observe_final", tt.state.sponge_state)

    with frx.profiler.TraceAnnotation("grind"):
        tt, pow_witness = tt.grind(params.proof_of_work_bits)
    tm.lap("grind(16)", pow_witness)

    tt, indices = sample_query_indices(
        tt, (n - 1).bit_length() + params.log_blowup, params.num_queries
    )
    tm.lap("sample_indices", None)

    idx = fnp.asarray(indices.astype(np.int32))
    with frx.profiler.TraceAnnotation("open_batch"):
        opens = [
            open_batch(tree, trace_data, idx),
            open_batch(tree, quotient.quotient_data, idx),
        ] + [
            open_batch(tree, data, idx >> (layer + 1))
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

    trace_root, trace_data = commit_pcs(tree, [trace], params.log_blowup)
    t = bind_instance(fresh_challenger(), degree_bits, trace_root, pv)
    q = QuotientProver(tree, params).prove(
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
                tree, trace, trace_data, claim, quotient, q.transcript, n, n_cols, params
            )
    if trace_dir:
        print(f"profiler trace written to {trace_dir}")


def main() -> None:
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


if __name__ == "__main__":
    main()
