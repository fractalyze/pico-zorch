//! Septic-curve golden vectors, generated with Pico's own arithmetic.
//!
//! Pico's global interaction argument accumulates one curve point per chip over
//! the degree-7 extension of KoalaBear and observes it in the transcript. That
//! arithmetic is `pico-vm`'s, so this links `pico-vm` rather than reimplementing
//! the reduction from the spec — a fixture derived from a second reading of the
//! source would pin that reading, not the reference.
//!
//! Split out from the main generator because `pico-vm` needs a nightly compiler
//! (`#![feature(const_type_id)]`), which the other fixtures should not require:
//!
//! ```sh
//! cargo +nightly run --features septic --bin septic-golden
//! ```
//!
//! Field elements serialize as canonical u32; a septic element as its seven
//! coefficients, low degree first.

use std::fs;
use std::path::Path;

use p3_field::{FieldAlgebra, PrimeField32};
use serde_json::{json, Value};

use pico_vm::machine::septic::{SepticCurve, SepticDigest, SepticExtension};

use pico_zorch_golden::Val;

fn ser_ext(x: SepticExtension<Val>) -> Value {
    json!(x.0.iter().map(|c| c.as_canonical_u32()).collect::<Vec<_>>())
}

fn ser_point(p: &SepticCurve<Val>) -> Value {
    json!({ "x": ser_ext(p.x), "y": ser_ext(p.y) })
}

/// A reproducible spread of septic elements — not random, so the fixture is
/// stable across runs, and not sparse, so a reduction bug in any coefficient
/// position shows up.
fn sample(seed: u32) -> SepticExtension<Val> {
    SepticExtension(std::array::from_fn(|i| {
        Val::from_canonical_u32(1 + seed * 7 + i as u32 * 31)
    }))
}

fn main() {
    let root = std::env::args().nth(1).unwrap_or_else(|| "..".to_string());

    // Extension arithmetic. `mul` is the operation with a reduction in it, so
    // it carries the fixture; add/sub are coefficient-wise and cannot hide a
    // bug that mul would not also expose.
    let products = (0..6u32)
        .map(|s| {
            let (a, b) = (sample(s), sample(s + 11));
            json!({
                "a": ser_ext(a),
                "b": ser_ext(b),
                "a_times_b": ser_ext(a * b),
                "a_squared": ser_ext(a * a),
                "a_cubed": ser_ext(a * a * a),
            })
        })
        .collect::<Vec<_>>();

    // `y^2 = x^3 + 2x + 611*z^5` evaluated away from the curve too: the
    // formula is a function of x alone, so it is pinned independently of
    // whether any particular x lifts to a point.
    let formulas = (0..4u32)
        .map(|s| {
            let x = sample(s);
            json!({ "x": ser_ext(x), "curve_formula": ser_ext(SepticCurve::curve_formula(x)) })
        })
        .collect::<Vec<_>>();

    // The published constants must satisfy the curve equation. Pinning them
    // gives the Python side points it can check its own arithmetic against
    // without needing to lift an x itself.
    let start = SepticDigest::<Val>::starting_digest().0;
    let zero = SepticDigest::<Val>::zero().0;
    assert!(start.check_on_point(), "reference starting digest must be on the curve");

    // Point addition, the operation the per-chip accumulation is built from.
    let doubled = start.double();
    let sum = start.add_incomplete(doubled);
    assert!(doubled.check_on_point(), "2P must stay on the curve");
    assert!(sum.check_on_point(), "P + 2P must stay on the curve");

    let out = format!("{root}/pico_zorch/septic/testdata/golden/septic.json");
    let value = json!({
        "modulus_note": "z^7 = 2 - 2*z^6  (EXT_COEFFS = [2,0,0,0,0,0,p-2])",
        "products": products,
        "curve_formulas": formulas,
        "starting_digest": ser_point(&start),
        "zero_digest": ser_point(&zero),
        "double_starting": ser_point(&doubled),
        "starting_plus_double": ser_point(&sum),
        "negated_starting": ser_point(&start.neg()),
    });

    let path = Path::new(&out);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, format!("{:#}\n", value)).unwrap();
    println!("wrote {out}");
}
