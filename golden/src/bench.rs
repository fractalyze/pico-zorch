//! Times the reference `p3_uni_stark::prove` on the same instance shape the
//! Python bench proves, so the two numbers are comparable: identical AIR,
//! trace height and width, and the config Pico runs.
//!
//! Build with `--features parallel` — Pico's own defaults include rayon, so
//! a serial build understates the reference.

use std::time::Instant;

use p3_uni_stark::{prove, verify};

use pico_zorch_golden::{config, fib_trace, pico_perm, Challenger, FibonacciAir, NUM_QUERIES};

fn main() {
    let mut args = std::env::args().skip(1);
    let degree_bits: usize = args.next().map_or(20, |a| a.parse().unwrap());
    let width: usize = args.next().map_or(32, |a| a.parse().unwrap());
    let runs: usize = args.next().map_or(5, |a| a.parse().unwrap());

    let (trace, last) = fib_trace(width, 1 << degree_bits);
    let pis = vec![Default::default(), p3_field::FieldAlgebra::ONE, last];
    let air = FibonacciAir { width };
    let cfg = config();

    println!(
        "reference p3_uni_stark::prove  degree_bits={degree_bits} width={width} \
         queries={NUM_QUERIES} parallel={}",
        cfg!(feature = "parallel")
    );
    for run in 0..runs {
        let mut challenger = Challenger::new(pico_perm());
        let t0 = Instant::now();
        let proof = prove(&cfg, &air, &mut challenger, trace.clone(), &pis);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if run == 0 {
            let mut v = Challenger::new(pico_perm());
            verify(&cfg, &air, &mut v, &proof, &pis).expect("reference proof must verify");
        }
        std::hint::black_box(&proof);
        println!("  run {}: {:.1}ms", run + 1, ms);
    }
}
