# pico-zorch

A lean **Pico prover** built on [`zorch`](https://github.com/fractalyze/zorch)'s
scheme-agnostic SNARK building blocks. `zorch` provides the reusable pieces
(hashing, Merkle commitment, Reed-Solomon LDE, transcript, …); `pico-zorch`
adds only the Pico-specific glue on top — the Poseidon2-KoalaBear
parameterizations, Pico's transcript and commitment conventions, and the
byte-match against the [Pico](https://github.com/brevis-network/pico)
reference prover.

```
frx  ──▶  zorch (scheme-/zkVM-agnostic blocks)  ──▶  pico-zorch (Pico / Plonky3 glue)
```

Why a separate repo: Pico proves with a Plonky3-based univariate STARK — FRI
over the KoalaBear field with Poseidon2 hashing. None of that scheme-specific
knowledge belongs in `zorch` (its hard rule); building directly on `zorch`'s
blocks keeps this prover small and gives a focused target to grow Pico glue
and benchmark against Pico's CUDA reference.

## Status

The full uni-stark proving pipeline runs on zorch's claim-reduction stages
and byte-matches the reference end to end — every wire field of a Fibonacci
proof (commitments, out-of-domain openings, FRI roots, final polynomial, PoW
witness, all query openings and Merkle paths) equals the output of the
vendored fork's own `p3_uni_stark::prove` under Pico's RISCV-phase FRI
config (log_blowup 1, 84 queries, 16 PoW bits):

| Layer | Module | Byte-matched against |
| --- | --- | --- |
| Poseidon2 permutation (Pico's RC_16_30 constants) | [`pico_zorch/poseidon2/`](pico_zorch/poseidon2/) | fork permutation vectors |
| DuplexChallenger flavour | [`pico_zorch/challenger/`](pico_zorch/challenger/) | observe/sample/sample_bits/grind script |
| TwoAdicFriPcs commit (coset LDE, bit-reversed MMCS) | [`pico_zorch/commit/`](pico_zorch/commit/) | trace LDE + Merkle root |
| Quotient, FRI open, prover/verifier stages | [`pico_zorch/uni_stark/`](pico_zorch/uni_stark/) | the complete `p3_uni_stark::prove` proof |

The prover is `StarkProver`, a zorch
`ProverStage[StarkClaim, StarkWitness, TrivialClaim, StarkProof,
DuplexTranscript]`; `StarkVerifier` is its explicit dual and ends on the
prover's exact sponge state. Not yet covered: Pico's machine-level prover
(the multi-chip outer transcript with permutation traces — `p3_uni_stark`
validates every layer beneath it), bincode wire serialization, and
GPU-scale shards.

## Regenerating the golden vectors

The fixtures under `pico_zorch/**/testdata/golden/` are produced by the Rust
harness in [`golden/`](golden/), which links the exact Plonky3 fork rev Pico
v2.0.0 vendors and rebuilds Pico's `KoalaBearPoseidon2` config:

```sh
cd golden && cargo run --release
```

The harness is deterministic — the serial (no-rayon) build makes the
reference's `find_any` grind "lowest witness wins", the same rule zorch's
`grind_search` implements — so a regeneration is a no-op unless the
reference pin changes.

## The scheme (what Pico actually runs)

Pico delegates proving to its Plonky3-based backend. The constants that pin
this repo's glue, all from
[Pico v2.0.0](https://github.com/brevis-network/pico/tree/v2.0.0) (the open
release behind Pico Prism 2.0):

- **Field**: KoalaBear (2^31 − 2^24 + 1) by default; the upstream repo also
  ships STARK-on-BabyBear, and CircleSTARK-on-Mersenne31 for the RISCV phase
  only.
- **Hash**: Poseidon2 over KoalaBear — width 16, x³ S-box, 4+4 external
  rounds, 20 internal rounds. Round constants are Pico's own `RC_16_30`
  table (vm/src/primitives/mod.rs), NOT Plonky3's default instance.
- **Merkle**: Plonky3 `MerkleTreeMmcs` — `PaddingFreeSponge<_, 16, 8, 8>` row
  leaves, `TruncatedPermutation<_, 2, 8, 16>` node compression, 8-element
  roots.
- **PCS**: FRI (two-adic, Reed-Solomon LDE), per Plonky3's `TwoAdicFriPcs`.
- **ISA**: RISC-V 64IM (moved from RV32 in v2.0.0); proof phases are RISCV →
  RECURSION → EVM.
- **Reference rev**: Pico v2.0.0 vendors
  [brevis-network/Plonky3@7fbe1908](https://github.com/brevis-network/Plonky3/tree/7fbe1908820f4d843370096ec517fd7429c9930d).

## Development

`pico-zorch` is pure Python on frx + zorch, built with Bazel (bzlmod). It
consumes `zorch` as a Bazel module, pinned in `MODULE.bazel` via
`git_override` for reproducible builds.

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.in \
    --extra-index-url https://fractalyze.github.io/pypi/simple/
```

**Dev against a local `zorch` checkout** instead of the pinned commit — create
`.bazelrc.user` (gitignored):

```
common --override_module=zorch=/abs/path/to/your/zorch/checkout
```

Run the tests (CPU is the default for determinism):

```sh
bazel test //...
```

## License

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).
