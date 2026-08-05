//! A GPU Pico uni-stark prover: a drop-in for `p3_uni_stark::prove` under
//! Pico's `KoalaBearPoseidon2` config, byte-identical to the reference.
//!
//! ```text
//! // before
//! let proof = p3_uni_stark::prove(&cfg, &air, &mut challenger, trace, &pis);
//! // after
//! let proof = pico_zorch::prove(&trace, &pis)?;   // p3_uni_stark::verify accepts it
//! ```
//!
//! # Why the whole proof is one executable
//!
//! A Groth16 back-end splits cleanly into a cheap head, one fused kernel and a
//! cheap tail, which is how [`bellman-zorch`] is shaped. A STARK does not:
//! Fiat-Shamir interleaves the transcript with every heavy stage, so a
//! host-side transcript would force a round trip per challenge. `pico_zorch`'s
//! Python prover runs the sponge on device, so the entire proof — trace
//! commit, quotient, FRI commit phase, grind and query openings — lowers to a
//! single StableHLO program. One dispatch, one readback.
//!
//! # Shape specialization
//!
//! The AIR, trace height and width trace into the core, so an executable is
//! fixed to one `(air, degree_bits, width)`; the trace *values* are a runtime
//! input. Export one per shape:
//!
//! ```sh
//! bazel run //export:export_uni_stark_core -- --degree_bits=20 --width=32
//! ```
//!
//! # Transcript
//!
//! The core starts from the fresh challenger. [`prove`] therefore takes no
//! challenger — see [`prove::prove_with`] for why accepting one would be worse
//! than refusing it.
//!
//! [`bellman-zorch`]: https://github.com/fractalyze/bellman-zorch

pub mod gpu;
pub mod manifest;
pub mod pcs;
pub mod transcript;
pub mod proof;
pub mod prove;
pub mod wire;

pub use gpu::{Core, Phases};
pub use manifest::Manifest;
pub use prove::{prove, prove_at, prove_with, CORE_ENV};

// The config this binding reproduces. Re-exported so a consumer can name the
// proof type and verify without also depending on `golden` directly.
pub use pico_zorch_golden::{config, pico_perm, Challenge, Challenger, MyConfig, Val};
