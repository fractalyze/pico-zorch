//! Times `pico_zorch::prove` on the instance `golden`'s `reference-bench`
//! times the reference CPU prover on, so the two numbers can be divided.
//!
//! ```sh
//! bazel run //export:export_pico_core -- --degree_bits=16 --width=32
//! cargo run --release --example bench -- 16 32 5
//! ```
//!
//! Every size asserts byte-identity against the reference before reporting a
//! time — a timed number is only worth as much as the byte-match at the same
//! pin, and a prover that is fast because it computes the wrong thing is the
//! failure mode worth guarding against.
//!
//! # Reading the output
//!
//! XLA compiles the core inside `gpu::load`, and the byte-match check above is
//! its first execution, so the timed loop is already warm. Pass 1 is still
//! dropped: autotuning and allocator growth settle on the first execution or
//! two. Quote a converged pass and its min-max spread, never the mean. A phase
//! whose spread is a large fraction of its value is not evidence in either
//! direction from a single run.
//!
//! # What this does *not* compare against
//!
//! Not Pico's machine prover. Its GPU path never calls `p3_uni_stark::prove`
//! (see `vm/src/machine/prover.rs`), so a ratio against Pico's published
//! block-proving numbers would be scope-confounded. The comparable thing is the
//! reference CPU prover on the identical instance:
//!
//! ```sh
//! cd golden && cargo run --release --bin reference-bench --features parallel -- 16 32 5
//! ```
//!
//! That build needs `--features parallel`: Pico's own defaults include rayon,
//! so a serial reference understates it. This binary links the serial build on
//! purpose, because it byte-matches (the reference grind is `find_any` over a
//! rayon iterator, so parallelism makes which witness it finds irreproducible).

use std::path::PathBuf;
use std::time::{Duration, Instant};

use p3_field::FieldAlgebra;
use pico_zorch::{config, pico_perm, Challenger, Phases, Val};
use pico_zorch_golden::{fib_trace, FibonacciAir};

fn core_path(degree_bits: usize, width: usize) -> PathBuf {
    if let Ok(explicit) = std::env::var(pico_zorch::CORE_ENV) {
        return PathBuf::from(explicit);
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate has a parent directory")
        .join("artifacts");
    root.join(format!("pico_core_fib_d{degree_bits}_w{width}.mlirbc"))
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() {
    let mut args = std::env::args().skip(1);
    let degree_bits: usize = args.next().map_or(16, |a| a.parse().unwrap());
    let width: usize = args.next().map_or(32, |a| a.parse().unwrap());
    let runs: usize = args.next().map_or(5, |a| a.parse().unwrap());

    let (trace, last) = fib_trace(width, 1 << degree_bits);
    let public_values = vec![Val::ZERO, Val::ONE, last];
    let air = FibonacciAir { width };
    let path = core_path(degree_bits, width);

    println!(
        "pico-zorch GPU prove   degree_bits={degree_bits} width={width} \
         queries={}\n  core: {}",
        pico_zorch_golden::NUM_QUERIES,
        path.display()
    );

    let core = pico_zorch::gpu::load(&path).unwrap_or_else(|e| {
        panic!("{e}\n  export it: bazel run //export:export_pico_core -- --degree_bits={degree_bits} --width={width}")
    });

    // Correctness gate. The serial reference is the byte-match contract; if
    // this fails the timings below are meaningless.
    let t0 = Instant::now();
    let mut challenger = Challenger::new(pico_perm());
    let reference =
        p3_uni_stark::prove(&config(), &air, &mut challenger, trace.clone(), &public_values);
    let reference_serial_ms = ms(t0.elapsed());

    let (first, _) = pico_zorch::prove_with(core, &trace, &public_values).expect("prove");
    assert_eq!(
        serde_json::to_value(&first).unwrap(),
        serde_json::to_value(&reference).unwrap(),
        "GPU proof differs from p3_uni_stark::prove at degree_bits={degree_bits} width={width}"
    );
    println!("  byte-match vs p3_uni_stark::prove: OK");
    println!(
        "  reference (serial, this process): {reference_serial_ms:.1}ms  \
         — not the comparison number, see the module docs"
    );

    let mut timings: Vec<(f64, Phases)> = Vec::with_capacity(runs);
    for run in 0..runs {
        let t = Instant::now();
        let (proof, phases) = pico_zorch::prove_with(core, &trace, &public_values).expect("prove");
        let total = ms(t.elapsed());
        std::hint::black_box(&proof);
        let tag = if run == 0 { "  <- dropped (not yet settled)" } else { "" };
        println!(
            "  run {}: {total:8.1}ms   h2d {:5.2}  dispatch {:7.1}  readback {:6.1}  assemble {:6.1}{tag}",
            run + 1,
            ms(phases.h2d),
            ms(phases.dispatch),
            ms(phases.readback),
            ms(phases.assemble),
        );
        timings.push((total, phases));
    }
    // The cold pass measured compilation, not proving; drop it before
    // reporting a converged figure rather than averaging it in.
    let warm: Vec<f64> = timings.iter().skip(1).map(|(t, _)| *t).collect();
    if warm.is_empty() {
        println!("\n  pass runs >= 2 for a converged figure");
        return;
    }
    let min = warm.iter().copied().fold(f64::INFINITY, f64::min);
    let max = warm.iter().copied().fold(0.0, f64::max);
    println!("\n  converged min {min:.1}ms (spread {min:.1}-{max:.1}) over {} passes", warm.len());
}
