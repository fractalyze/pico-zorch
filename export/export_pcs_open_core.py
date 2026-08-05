# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Export the batched PCS opening as one executable.

The other half of the seam Pico's machine prover swaps at. Unlike the uni-stark
core this is not a whole proof: `machine::prove` drives its own transcript on
the host across four rounds, with AIR work in between, so the PCS is called as
two stages and the transcript has to cross the boundary.

    core(state, traces..., points...) -> (opened, roots, final_poly, witness,
                                          indices, openings..., state')

The sponge state goes in and comes back out because of that. It is five
fixed-size arrays, so it traces like any other buffer and the host never has to
know how the sponge works — it just carries the state between stages.

**Traces, not extensions, are the input.** The opening needs each matrix's LDE,
but re-deriving it on device costs one NTT and moves `1/blowup` as much data as
shipping it would. That matters here specifically: the pinned `xla-pjrt` copies
every output to host, so anything this core were handed from the commit core
would have made a round trip first. Recomputing is strictly cheaper than
transferring until an execution's outputs can stay resident.

Shape specialization: the round layout, matrix shapes and per-matrix point
counts all trace in.

    bazel run //export:export_pcs_open_core -- --rounds=16x2:2,8x1:1;16x4:2,4x3:1
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import frx
import numpy as np
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from pico_zorch.challenger.challenger import PicoTranscript
from pico_zorch.commit.mmcs import MerkleTreeMmcs
from pico_zorch.pcs.open import (
    MatrixOpening,
    commit_phase_openings,
    commit_phase_over_rounds,
)
from pico_zorch.pcs.two_adic_fri import TwoAdicFriPcs
from pico_zorch.poseidon2.koalabear import koalabear16_merkle

ART = Path(
    os.environ.get(
        "PICO_ZORCH_ARTIFACTS",
        str(Path(__file__).resolve().parent.parent / "artifacts"),
    )
)


def parse_rounds(text: str) -> list[list[tuple[int, int, int]]]:
    """`16x2:2,8x1:1;16x4:2` -> [[(16,2,2),(8,1,1)],[(16,4,2)]].

    Height x width : number of opening points, matrices comma-separated,
    rounds semicolon-separated.
    """
    rounds = []
    for chunk in text.split(";"):
        matrices = []
        for part in chunk.split(","):
            shape, _, npoints = part.strip().partition(":")
            height, _, width = shape.partition("x")
            matrices.append((int(height), int(width), int(npoints)))
        rounds.append(matrices)
    return rounds


def core_fn(spec, log_blowup: int, num_queries: int, proof_of_work_bits: int):
    """The opening as one traceable function of state, traces and points."""
    _, compressor, tree = koalabear16_merkle()
    pcs = TwoAdicFriPcs(MerkleTreeMmcs(tree, compressor), log_blowup=log_blowup)

    def core(state, traces, points):
        rounds = []
        i = 0
        for matrices in spec:
            round_ = []
            for _ in matrices:
                trace = traces[i]
                round_.append(
                    MatrixOpening(lde=pcs.lde(trace), trace=trace, points=points[i])
                )
                i += 1
            rounds.append(round_)

        transcript = PicoTranscript.from_state(state)
        opened, final_poly, roots, layers, witness, indices, t = (
            commit_phase_over_rounds(
                tree, rounds, transcript, log_blowup, proof_of_work_bits, num_queries
            )
        )
        steps = [
            commit_phase_openings(tree, layers, indices[q], log_blowup)
            for q in range(num_queries)
        ]
        return opened, roots, final_poly, witness, indices, steps, t.state

    return core


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", required=True, help="HxW:POINTS,... ; ... per round")
    ap.add_argument("--log_blowup", type=int, default=1)
    ap.add_argument("--num_queries", type=int, default=84)
    ap.add_argument("--proof_of_work_bits", type=int, default=16)
    args = ap.parse_args()

    spec = parse_rounds(args.rounds)
    fn = core_fn(spec, args.log_blowup, args.num_queries, args.proof_of_work_bits)

    flat = [m for round_ in spec for m in round_]
    traces = [np.zeros((h, w), dtype=F) for h, w, _ in flat]
    points = [[np.zeros((), dtype=EF)] * n for _, _, n in flat]
    state = PicoTranscript.new().state

    lowered = frx.jit(fn).lower(state, traces, points)
    buf = io.BytesIO()
    lowered.compiler_ir(dialect="stablehlo").operation.write_bytecode(buf)

    ART.mkdir(parents=True, exist_ok=True)
    stem = "pcs_open_core_" + args.rounds.replace(";", "__").replace(",", "_").replace(
        ":", "p"
    )
    (ART / f"{stem}.mlirbc").write_bytes(buf.getvalue())

    manifest = {
        "log_blowup": args.log_blowup,
        "num_queries": args.num_queries,
        "proof_of_work_bits": args.proof_of_work_bits,
        "rounds": [
            [{"height": h, "width": w, "points": n} for h, w, n in round_]
            for round_ in spec
        ],
    }
    (ART / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    size = (ART / f"{stem}.mlirbc").stat().st_size
    print(f"wrote {ART / stem}.mlirbc ({size} B) + .json; rounds={args.rounds}")


if __name__ == "__main__":
    main()
