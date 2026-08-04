# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Export the batched PCS commit as one executable.

This is the core behind Pico's `commit_main`: every chip's main trace goes in
at its own height, and one root comes out. It is a much smaller program than
the uni-stark core — no transcript, no FRI, no queries — because the machine
prover keeps its Fiat-Shamir on the host and calls the PCS per round.

    core(m0, m1, ...) -> (root, lde0, lde1, ..., layer0, layer1, ...)

The extensions and the digest layers come back because the Rust side rebuilds
Plonky3's `MerkleTree` from them: the opening argument still runs on the
reference's CPU path, and that path reads its matrices and siblings straight
out of the prover data. Shipping them back is the cost of swapping only the
commit; it goes away when the opening moves too.

Shape specialization: the batch's heights and widths trace in, so an executable
is fixed to one list of shapes and the trace *values* are runtime inputs.

    bazel run //export:export_pcs_commit_core -- --shapes=4x3,16x2,8x1,16x4
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

from pico_zorch.commit.mmcs import MerkleTreeMmcs
from pico_zorch.pcs.two_adic_fri import TwoAdicFriPcs
from pico_zorch.poseidon2.koalabear import koalabear16_merkle

ART = Path(
    os.environ.get(
        "PICO_ZORCH_ARTIFACTS",
        str(Path(__file__).resolve().parent.parent / "artifacts"),
    )
)


def parse_shapes(text: str) -> list[tuple[int, int]]:
    """`4x3,16x2` -> [(4, 3), (16, 2)] — height x width, in commit order."""
    shapes = []
    for part in text.split(","):
        height, _, width = part.strip().partition("x")
        shapes.append((int(height), int(width)))
    return shapes


def output_names(num_matrices: int, num_layers: int) -> list[str]:
    names = ["root"]
    names += [f"lde{i}" for i in range(num_matrices)]
    names += [f"digest_layer{i}" for i in range(num_layers)]
    return names


def core_fn(log_blowup: int):
    """The batched commit as one traceable function of the trace matrices."""
    _, compressor, tree = koalabear16_merkle()
    # Built outside the trace: the round-constant tables validate themselves
    # with numpy, which a traced `fnp.array` would defeat.
    pcs = TwoAdicFriPcs(MerkleTreeMmcs(tree, compressor), log_blowup=log_blowup)

    def core(*matrices):
        ldes = tuple(pcs.lde(m) for m in matrices)
        root, digest_layers = pcs.mmcs.commit(ldes)
        return (root, *ldes, *digest_layers)

    return core


def _spec(name: str, array) -> dict:
    return {"name": name, "dtype": str(array.dtype), "dims": list(array.shape)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--shapes",
        required=True,
        help="comma-separated HEIGHTxWIDTH per matrix, in commit order",
    )
    ap.add_argument("--log_blowup", type=int, default=1)
    args = ap.parse_args()

    shapes = parse_shapes(args.shapes)
    fn = core_fn(args.log_blowup)
    example = tuple(np.zeros((h, w), dtype=F) for h, w in shapes)

    # Trace once for the manifest, lower the same function for the bytecode.
    out_avals = frx.eval_shape(fn, *example)
    num_layers = len(out_avals) - 1 - len(shapes)
    names = output_names(len(shapes), num_layers)
    if len(names) != len(out_avals):
        raise AssertionError(
            f"manifest names {len(names)} outputs but the core returns {len(out_avals)}"
        )

    lowered = frx.jit(fn).lower(*example)
    buf = io.BytesIO()
    lowered.compiler_ir(dialect="stablehlo").operation.write_bytecode(buf)

    ART.mkdir(parents=True, exist_ok=True)
    stem = "pcs_commit_core_" + "_".join(f"{h}x{w}" for h, w in shapes)
    (ART / f"{stem}.mlirbc").write_bytes(buf.getvalue())

    manifest = {
        "log_blowup": args.log_blowup,
        "shapes": [{"height": h, "width": w} for h, w in shapes],
        "inputs": [
            _spec(f"matrix{i}", a) for i, a in enumerate(example)
        ],
        "outputs": [_spec(n, a) for n, a in zip(names, out_avals)],
    }
    (ART / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    size = (ART / f"{stem}.mlirbc").stat().st_size
    print(f"wrote {ART / stem}.mlirbc ({size} B) + .json; shapes={args.shapes}")


if __name__ == "__main__":
    main()
