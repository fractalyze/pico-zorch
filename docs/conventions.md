# Coding conventions

`pico-zorch` inherits `zorch`'s conventions (`@jit` usage, type annotations,
`_`-private naming, snake_case); this page only adds the rules specific to a
byte-match consumer repo.

## Comments carry WHY, never WHAT

Names, types, and tests already say what the code does. A comment exists to
state what the code cannot: which Pico / Plonky3 convention a line mirrors,
why a shape or ordering is load-bearing for the byte-match, which reference
decision forced a non-obvious choice.

## Pin external references

Link Pico sources as GitHub permalinks at tag
[`v2.0.0`](https://github.com/brevis-network/pico/tree/v2.0.0) (or a short
commit SHA) with line ranges — never a branch. Plonky3 references point at
Pico's vendored fork rev,
[brevis-network/Plonky3@7fbe1908](https://github.com/brevis-network/Plonky3/tree/7fbe1908820f4d843370096ec517fd7429c9930d),
not upstream Plonky3, because Pico's byte behaviour is the fork's. A branch
link rots silently; a permalink stays true to the constant it pins.

## Golden tests are the spec

Every primitive that mirrors Pico's backend (Poseidon2 permutation, Merkle
tree, transcript, LDE, FRI) is pinned by a golden vector generated from the
reference. Conventions:

- Goldens live in `testdata/golden/*.json` next to the test that consumes
  them, are small (KBs), and are committed.
- A golden test compares with exact equality (`fnp.array_equal`), never a
  tolerance — field elements either match or they don't.
- A golden's provenance (which crate, which rev, which harness produced it)
  is stated where the golden is defined.

## What lives here vs in zorch

Pico / Plonky3 glue lives here: the Poseidon2-KoalaBear round constants, the
MerkleTreeMmcs shape conventions (PaddingFreeSponge leaves,
TruncatedPermutation folds, 8-element roots), Pico's transcript discipline,
and the trace-commit pipeline. Anything reusable by another scheme — the
Poseidon2 permutation core, k-ary Merkle folding, Reed-Solomon/NTT, the
duplex transcript — belongs in `zorch`.
