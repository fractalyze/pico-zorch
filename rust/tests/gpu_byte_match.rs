//! The contract: our proof is the reference's proof, field for field.
//!
//! This is the test that licenses the claim in the README. It needs a GPU and
//! a core exported for the same instance, so it is `#[ignore]`d by default:
//!
//! ```sh
//! bazel run //export:export_pico_core -- --degree_bits=3 --width=2
//! export PICO_ZORCH_CORE_MLIRBC=$PWD/../artifacts/pico_core_fib_d3_w2.mlirbc
//! export XLA_PJRT_PLUGIN=<...>/frx_plugins/xla_cuda12/xla_cuda_plugin.so
//! cargo test --test gpu_byte_match -- --ignored
//! ```
//!
//! The instance matches `golden/`'s fixture generator, so a failure here and a
//! failure in `//export:export_pico_core_test` point at the same drift from
//! opposite sides.

use std::path::Path;

use p3_field::FieldAlgebra;
use p3_matrix::Matrix;
use pico_zorch::{config, pico_perm, Challenger, Val};
use pico_zorch_golden::{fib_trace, FibonacciAir};

/// The shape `export/export_pico_core.py --degree_bits=3 --width=2` produces.
const DEGREE_BITS: usize = 3;
const WIDTH: usize = 2;

fn core_path() -> String {
    std::env::var(pico_zorch::CORE_ENV).unwrap_or_else(|_| {
        panic!(
            "set {} to a core exported for degree_bits={DEGREE_BITS} width={WIDTH}",
            pico_zorch::CORE_ENV
        )
    })
}

#[test]
#[ignore = "needs a GPU and an exported core (see the module docs)"]
fn proof_is_byte_identical_to_the_reference() {
    let (trace, last) = fib_trace(WIDTH, 1 << DEGREE_BITS);
    let public_values = vec![Val::ZERO, Val::ONE, last];
    let air = FibonacciAir { width: WIDTH };
    let cfg = config();

    let mut challenger = Challenger::new(pico_perm());
    let reference = p3_uni_stark::prove(&cfg, &air, &mut challenger, trace.clone(), &public_values);

    let ours = pico_zorch::prove_at(Path::new(&core_path()), &trace, &public_values)
        .expect("prove on the exported core");

    // `Proof`'s fields are private, so compare through its serialization —
    // which covers every field, nested to the leaves.
    assert_eq!(
        serde_json::to_value(&ours).unwrap(),
        serde_json::to_value(&reference).unwrap(),
        "proof differs from p3_uni_stark::prove"
    );
}

#[test]
#[ignore = "needs a GPU and an exported core (see the module docs)"]
fn reference_verifier_accepts_our_proof() {
    let (trace, last) = fib_trace(WIDTH, 1 << DEGREE_BITS);
    let public_values = vec![Val::ZERO, Val::ONE, last];
    let air = FibonacciAir { width: WIDTH };

    let ours = pico_zorch::prove_at(Path::new(&core_path()), &trace, &public_values)
        .expect("prove on the exported core");

    let mut challenger = Challenger::new(pico_perm());
    p3_uni_stark::verify(&config(), &air, &mut challenger, &ours, &public_values)
        .expect("the reference verifier must accept our proof");
}

#[test]
#[ignore = "needs a GPU and an exported core (see the module docs)"]
fn rejects_a_trace_the_core_was_not_exported_for() {
    // A shape mismatch has to fail here, with a message naming the re-export,
    // rather than as an opaque plugin abort deep inside PJRT.
    let (trace, _) = fib_trace(WIDTH, 1 << (DEGREE_BITS + 1));
    let public_values = vec![Val::ZERO, Val::ONE, Val::ZERO];

    // `Proof` has no `Debug`, so unwrap the error by hand.
    let err = match pico_zorch::prove_at(Path::new(&core_path()), &trace, &public_values) {
        Ok(_) => panic!("a mismatched trace height must be rejected"),
        Err(e) => e,
    };
    assert!(
        err.contains("re-export"),
        "error should tell the caller how to fix it, got: {err}"
    );
    assert_eq!(trace.width(), WIDTH);
}
