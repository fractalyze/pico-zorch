# pico-zorch (Rust binding)

A drop-in GPU replacement for `p3_uni_stark::prove` under Pico's
`KoalaBearPoseidon2` config. Swap one call and the proof is **byte-identical**
to the reference:

```rust
// before
let proof = p3_uni_stark::prove(&cfg, &air, &mut challenger, trace, &pis);

// after — p3_uni_stark::verify accepts it unchanged
let proof = pico_zorch::prove(&trace, &pis)?;
```

The AIR, the FRI config and the challenger are baked into the exported core, so
they are not arguments here; the core's manifest records them and `prove`
rejects a trace the core was not exported for.

## Why the whole proof is one executable

[`bellman-zorch`](https://github.com/fractalyze/bellman-zorch) splits Groth16
into a cheap CPU head, one fused kernel and a cheap CPU tail. A STARK does not
split that way — Fiat-Shamir interleaves the transcript with every heavy stage,
so a host-side transcript would cost a round trip per challenge.

`pico_zorch`'s Python prover already runs the sponge on device, so the entire
proof lowers to a single StableHLO program: trace commit, quotient, FRI commit
phase, grind and query openings. One dispatch, one readback, no host round trip
mid-proof.

## Setup

Needs an NVIDIA GPU, a Rust toolchain, and `clang`/`libclang` (the `xla-pjrt`
shim generates its PJRT bindings with `bindgen` at build time). The CUDA 12
userspace and the PJRT plugin both come from the repo's Bazel pip set, so there
is nothing to install by hand — just point at them:

```sh
EXT=$(bazel info output_base)/external
export LD_LIBRARY_PATH=$(ls -d $EXT/*nvidia*/site-packages/nvidia/*/lib | tr '\n' ':')
export XLA_PJRT_PLUGIN=$EXT/*frx_cuda12_pjrt/site-packages/frx_plugins/xla_cuda12/xla_cuda_plugin.so
```

## Exporting a core

`lax.ntt` and the Poseidon2 permutation lower shape-specialized, so an
executable is fixed to one `(air, degree_bits, width)`; the trace *values* are a
runtime input. Export one per instance shape:

```sh
FRX_PLATFORMS=cuda bazel run //export:export_pico_core -- --degree_bits=3 --width=2
# -> artifacts/pico_core_fib_d3_w2.mlirbc  (+ .json manifest)
```

Export against the backend you will run on. The manifest beside the bytecode
names every input and output in parameter order; the Rust side binds buffers by
name, so adding a proof field cannot silently shift a decode by one.

## Running

```sh
# No GPU needed — proof reassembly against a real reference proof:
cargo test

# The byte-match contract (needs a GPU and a matching core):
export PICO_ZORCH_CORE_MLIRBC=$PWD/../artifacts/pico_core_fib_d3_w2.mlirbc
cargo test --test gpu_byte_match -- --ignored --test-threads=1
```

`--test-threads=1` matters: a second PJRT client in one process aborts.

## Benchmark

RTX 5090 vs the reference CPU prover on the identical instance, widened
Fibonacci AIR, 84 queries. Each size asserts byte-identity against
`p3_uni_stark::prove` before it is timed. CPU is built `--features parallel`
(Pico ships rayon); both figures are the minimum over converged passes.

| rows | CPU (parallel) | GPU | speedup |
| ---- | -------------- | --- | ------- |
| 2^16 = 65,536 | 74.6 ms | 5.6 ms | 13.3x |
| 2^20 = 1,048,576 | 1047.9 ms | 23.4 ms | 44.8x |

### Per-proof phase breakdown (ms)

| rows | total | h2d | dispatch | readback | assemble |
| ---- | ----- | --- | -------- | -------- | -------- |
| 2^16 | 5.6 | 0.55 | 2.5 | 1.3 | 1.3 |
| 2^20 | 23.4 | 8.2 | 11.8 | 1.6 | 1.9 |

Two of those columns are host work rather than proving:

- **`h2d`** is uploading the trace — 134 MB at 2^20 x 32, about 16 GB/s, and
  now the largest single host cost. This is the column the Montgomery wire
  protects: a canonical wire would add a full host-side pass over that same
  134 MB before the upload could start.
- **`assemble`** is rebuilding `p3_uni_stark::Proof` through serde. It carries
  no information — it exists purely because `Proof`'s fields are private — so
  it is overhead by construction. It was *half* the wall time at 2^16 until the
  hop moved from JSON to bincode; what is left is close to the floor for
  materializing the proof structure at all.

The repo's Python bench reports a different end-to-end figure at the same sizes
(`docs/development.md`). That is not a contradiction: it generates its trace on
device and never leaves, so it pays neither `h2d` nor `assemble`. Both numbers
are honest; this one is what a Rust caller actually sees.

## The wire

Field elements cross the boundary as **raw Montgomery limbs**, not canonical
values. Plonky3's `MontyField31` is `#[repr(transparent)]` over `u32` with the
same modulus (`0x7f000001`) and the same `R = 2^32` that `zk_dtypes` uses, so a
buffer is reinterpreted rather than converted — a canonical wire would cost a
Montgomery reduction per element on both sides, 33M of them per proof for a
2^20 x 32 trace.

That agreement is load-bearing, so it is pinned from both directions:
`export_pico_core_test.WireRepresentationTest` on the Python side and
`wire::tests::montgomery_constant_matches_the_exporter` here. A change in either
library fails as a one-line assertion instead of an unexplainable proof
mismatch.

Extension elements arrive widened to their four base coefficients, which is
exactly `BinomialExtensionField<KoalaBear, 4>`.

## Two layout differences, undone in `proof::assemble`

- **Query-major batching.** The core opens every query of a tree in one vmapped
  call, so an opening arrives as `[queries, ...]` and a Merkle path as
  `[depth, queries, 8]`. The reference nests the other way, one `QueryProof` per
  query.
- **Pair rows vs siblings.** The core keeps a FRI layer's whole pair row where
  the reference keeps only the sibling. Which half that is follows the parity of
  `index >> layer`, which is why the core also returns the sampled query
  indices.

`Proof`, `Commitments` and `OpenedValues` have private fields, so those three
are rebuilt through serde; everything below them is public and built directly.
Both ends of that hop use identical leaf types, so it re-wraps rather than
re-encodes.

## Scope

`p3_uni_stark::prove` only. Pico's machine-level prover — the multi-chip outer
transcript with permutation traces — is not covered by the Python prover yet, so
it is not covered here either. See the repo [README](../README.md#status).
