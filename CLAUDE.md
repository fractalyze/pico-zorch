# Project context for Claude Code

Everything load-bearing lives in repo docs. Treat those as the source of truth;
this file is just the map plus the rules every change must respect.

- **Project overview & quick start:** [`README.md`](README.md)
- **Coding conventions:** [`docs/conventions.md`](docs/conventions.md)
- **Environment, testing, and the per-stage bench:** [`docs/development.md`](docs/development.md)

## Non-negotiables

- **Pico-specific only.** This repo holds the Pico / Plonky3 glue: the
  Poseidon2-KoalaBear parameterizations, Pico's transcript and commitment
  conventions, and the byte-match against the Pico reference prover. Anything
  scheme- or zkVM-agnostic belongs upstream in `zorch`, never here.
- **Byte-match is the contract.** Every primitive that mirrors Pico's backend
  must be pinned by a golden vector generated from the reference. A change
  that breaks a golden test is wrong until the reference says otherwise.
- **Pin external references.** Link Pico sources as GitHub permalinks at tag
  `v2.0.0` (or a short commit SHA) and Plonky3 sources at Pico's vendored
  fork rev `brevis-network/Plonky3@7fbe1908` — never a branch.
