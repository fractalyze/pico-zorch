# Copyright 2026 The pico-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Export the fused uni-stark core: one executable per instance shape.

Unlike a Groth16 back-end, a STARK does not split into "cheap CPU head, one
fused kernel, cheap CPU tail" — Fiat-Shamir interleaves the transcript with
every heavy stage, so a host-side transcript would force a round trip per
challenge. `pico_zorch`'s prover already runs the sponge on device, which is
what lets the *whole* proof lower to a single program:

    core(trace, public_values) -> every field of the reference `Proof`

The trace commitment, quotient, FRI commit phase, grind and query openings all
land in that one program, so the only host round trip in a proof is reading the
finished proof back.

Named for what it is: this core reproduces `p3_uni_stark::prove`, which Pico's
machine prover does not call (`vm/src/machine/prover.rs` runs a multi-chip
protocol and reaches these kernels through the `Pcs` trait). Calling it a "pico
core" would claim a scope it does not have.

Shape specialization: the AIR, trace height and width trace in, so an
executable is fixed to one `(air, degree_bits, width)` and the trace *values*
are a runtime input — the same per-shape core `bellman-zorch` exports.

The initial transcript is the fresh (all-zero) challenger, matching
`p3_uni_stark::prove` under `Challenger::new(pico_perm())`. A core cannot serve
a pre-seeded challenger; the Rust binding rejects one rather than proving
against the wrong sponge.

Alongside the bytecode the exporter writes a JSON manifest naming every input
and output in parameter order, so the Rust side binds buffers to proof fields
by name instead of by an index it could silently get wrong. The shapes come
from tracing the same function that is lowered, so the manifest cannot drift
from what the executable returns.

Run under the Bazel-provided interpreter (see docs/development.md):

    bazel run //export:export_uni_stark_core -- --degree_bits=3 --width=2
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import frx
import frx.numpy as fnp
import numpy as np
from frx import lax
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from pico_zorch.challenger.challenger import fresh_challenger
from pico_zorch.poseidon2.koalabear import koalabear16_merkle
from pico_zorch.uni_stark.fri import FriOpener
from pico_zorch.uni_stark.prover import StarkProver
from pico_zorch.uni_stark.quotient import log_quotient_degree
from pico_zorch.uni_stark.testing.fib_air import FibonacciAir
from pico_zorch.uni_stark.types import FriParams, StarkClaim, StarkWitness

# An AIR traces into the core, so the exporter has to name one. Adding an AIR
# is an entry here plus its Python definition — the Rust side is unchanged.
# `FibonacciAir` keeps the recurrence in columns 0-1 and pads beyond, so width
# sweeps without the constraint set changing (the reference's widened
# `FibonacciAir { width }`).
AIRS = {"fib": FibonacciAir}

# Public-value count per AIR: the length of the reference's `pis`.
PUBLIC_VALUES = {"fib": 3}

# The 12 proof fields that do not depend on the FRI layer count, in the order
# `_flatten` emits them.
FIXED_OUTPUTS = (
    "trace_root",
    "quotient_root",
    "trace_local",
    "trace_next",
    "quotient_chunks",
    "commit_phase_roots",
    "final_poly",
    "pow_witness",
    "trace_opening_rows",
    "trace_opening_paths",
    "quotient_opening_rows",
    "quotient_opening_paths",
    "query_indices",
)

ART = Path(
    os.environ.get(
        "PICO_ZORCH_ARTIFACTS",
        str(Path(__file__).resolve().parent.parent / "artifacts"),
    )
)


def output_names(num_layers: int) -> list[str]:
    """Every output in emission order. `_flatten` asserts against this, so the
    manifest and the executable cannot disagree about what output `i` is."""
    names = list(FIXED_OUTPUTS)
    for layer in range(num_layers):
        names += [f"fri_layer{layer}_rows", f"fri_layer{layer}_paths"]
    return names


def _wire(x):
    """Reinterpret to the wire type. No arithmetic, and deliberately so.

    The wire is the raw Montgomery limb, which is exactly Plonky3's in-memory
    `KoalaBear` (`MontyField31` is `#[repr(transparent)]` over the same
    `u32`, same modulus 0x7f000001, same R = 2^32). So the Rust side casts the
    buffer instead of rebuilding elements one at a time.

    Converting to standard form here would look tidier and cost a Montgomery
    reduction per element on both sides — 2^20 x 32 of them for the trace
    alone, every proof. Plonky3's own note on the canonical constructor says
    it "should be avoided in performance critical implementations".

    An extension element bitcasts to its four base coefficients on a new
    trailing axis: that is what `BinomialExtensionField<KoalaBear, 4>` is, and
    it keeps the whole wire one type.
    """
    return lax.bitcast_convert_type(x, F) if x.dtype == EF else x


def _flatten(proof, num_layers: int) -> list:
    """The reference `Proof`, field for field, as a flat list of arrays.

    Merkle paths stack because every entry of an arity-2 path is one 8-element
    digest and the batched open puts the query axis in front: a `depth`-long
    list of `[queries, 8]` becomes `[depth, queries, 8]`.
    """
    opening = proof.opening
    fri = opening.fri
    out = [
        _wire(proof.trace_root),
        _wire(proof.quotient_root),
        _wire(opening.trace_local),
        _wire(opening.trace_next),
        _wire(opening.quotient_chunks),
        _wire(fnp.stack(fri.commit_phase_roots)),
        _wire(fri.final_poly),
        _wire(fri.pow_witness),
        _wire(fri.trace_openings.row),
        _wire(fnp.stack(fri.trace_openings.path)),
        _wire(fri.quotient_openings.row),
        _wire(fnp.stack(fri.quotient_openings.path)),
        # Not a field value — the sampled indices, which the Rust side needs to
        # pick the reference's sibling out of each layer's pair row.
        fri.query_indices,
    ]
    for layer in range(num_layers):
        opened = fri.commit_phase_openings[layer]
        out.append(_wire(opened.row))
        out.append(_wire(fnp.stack(opened.path)))
    assert len(out) == len(output_names(num_layers))
    return out


def core_fn(air, degree_bits: int, params: FriParams):
    """The whole prover as one traceable function of `(trace, public_values)`."""
    _, _, tree = koalabear16_merkle()
    prover = StarkProver(FriOpener(tree, params))
    # Built outside the trace: the round-constant tables validate themselves
    # with numpy, which a traced `fnp.array` would defeat. Concrete here, they
    # bake into the executable as constants — which is what they are.
    challenger = fresh_challenger()

    def core(trace, public_values):
        # Both arrive in the wire type, which is already the prover's type —
        # no conversion on the way in either.
        claim = StarkClaim(air=air, public_values=public_values, degree_bits=degree_bits)
        result = prover.prove(claim, StarkWitness(trace), challenger)
        proof = result.reduction_proof
        # The layer count follows the codeword shape, so it is known at trace
        # time — the same `log_max_height - log_blowup` the verifier checks.
        num_layers = len(proof.opening.fri.commit_phase_roots)
        return tuple(_flatten(proof, num_layers))

    return core


def _spec(name: str, array) -> dict:
    return {"name": name, "dtype": str(array.dtype), "dims": list(array.shape)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--air", default="fib", choices=sorted(AIRS))
    ap.add_argument("--degree_bits", type=int, required=True)
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--log_blowup", type=int, default=FriParams.log_blowup)
    ap.add_argument("--num_queries", type=int, default=FriParams.num_queries)
    ap.add_argument(
        "--proof_of_work_bits", type=int, default=FriParams.proof_of_work_bits
    )
    args = ap.parse_args()

    params = FriParams(
        log_blowup=args.log_blowup,
        num_queries=args.num_queries,
        proof_of_work_bits=args.proof_of_work_bits,
    )
    air = AIRS[args.air](width=args.width)
    n = 1 << args.degree_bits

    fn = core_fn(air, args.degree_bits, params)
    example = (
        np.zeros((n, args.width), dtype=F),
        np.zeros(PUBLIC_VALUES[args.air], dtype=F),
    )

    # Trace once for the manifest, then lower the same function for the
    # bytecode: the shapes reported are the shapes produced.
    out_avals = frx.eval_shape(fn, *example)
    # Two outputs (rows, paths) per FRI layer beyond the fixed fields.
    names = output_names((len(out_avals) - len(FIXED_OUTPUTS)) // 2)
    if len(names) != len(out_avals):
        raise AssertionError(
            f"manifest names {len(names)} outputs but the core returns "
            f"{len(out_avals)}"
        )

    lowered = frx.jit(fn).lower(*example)
    module = lowered.compiler_ir(dialect="stablehlo")
    buf = io.BytesIO()
    module.operation.write_bytecode(buf)

    ART.mkdir(parents=True, exist_ok=True)
    stem = f"uni_stark_core_{args.air}_d{args.degree_bits}_w{args.width}"
    (ART / f"{stem}.mlirbc").write_bytes(buf.getvalue())

    manifest = {
        "air": args.air,
        "degree_bits": args.degree_bits,
        "width": args.width,
        "log_blowup": params.log_blowup,
        "num_queries": params.num_queries,
        "proof_of_work_bits": params.proof_of_work_bits,
        "quotient_degree": 1 << log_quotient_degree(air.constraint_degree),
        "inputs": [_spec("trace", example[0]), _spec("public_values", example[1])],
        "outputs": [_spec(n, a) for n, a in zip(names, out_avals)],
    }
    (ART / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")

    size = (ART / f"{stem}.mlirbc").stat().st_size
    print(
        f"wrote {ART / stem}.mlirbc ({size} B) + .json; "
        f"air={args.air} degree_bits={args.degree_bits} width={args.width}"
    )


if __name__ == "__main__":
    main()
