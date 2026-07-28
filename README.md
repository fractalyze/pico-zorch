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

Early bootstrap: the Poseidon2-KoalaBear-16 hash stack
([`pico_zorch/poseidon2/`](pico_zorch/poseidon2/)) on zorch's engine, with the
Merkle root byte-matched against Plonky3's `MerkleTreeMmcs` golden vector.
Next: pin the round constants and goldens against Pico's vendored Plonky3
fork rather than upstream, then grow the RISCV-phase trace commit.

## The scheme (what Pico actually runs)

Pico delegates proving to its Plonky3-based backend. The constants that pin
this repo's glue, all from
[Pico v2.0.0](https://github.com/brevis-network/pico/tree/v2.0.0) (the open
release behind Pico Prism 2.0):

- **Field**: KoalaBear (2^31 − 2^24 + 1) by default; the upstream repo also
  ships STARK-on-BabyBear, and CircleSTARK-on-Mersenne31 for the RISCV phase
  only.
- **Hash**: Poseidon2 over KoalaBear — width 16, x³ S-box, 4+4 external
  rounds, 20 internal rounds (Plonky3's `default_koalabear_poseidon2_16`).
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
