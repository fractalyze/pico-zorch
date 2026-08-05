//! The GPU `Pcs::commit` is the reference's, and what it hands on is too.
//!
//! Matching the root is necessary but not sufficient: the consumer's quotient
//! stage reads extensions back out through `get_evaluations_on_domain`, so a
//! commitment that roots correctly over the wrong extensions would pass a root
//! check and fail much later. Both are checked here.
//!
//! Needs a GPU and a core exported for the same batch shape:
//!
//! ```sh
//! bazel run //export:export_pcs_commit_core -- --shapes=4x3,16x2,8x1,16x4
//! cargo test --test pcs_commit -- --ignored --test-threads=1
//! ```

use std::path::PathBuf;

use p3_commit::{Pcs, TwoAdicMultiplicativeCoset};
use p3_field::{Field, FieldAlgebra};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use pico_zorch::{pcs::TwoAdicFriPcs, Challenge, Challenger, Val};
use pico_zorch_golden::{pcs, MyPcs};

/// The shapes `//export:export_pcs_commit_core --shapes=4x3,16x2,8x1,16x4`
/// produces — the same batch `golden`'s `emit_batch_commit` pins, deliberately
/// unsorted so a consumer that forgets to sort by height fails.
const SHAPES: [(usize, usize); 4] = [(4, 3), (16, 2), (8, 1), (16, 4)];

fn artifacts() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate has a parent directory")
        .join("artifacts")
}

fn commit_core_path() -> PathBuf {
    let stem: Vec<String> = SHAPES.iter().map(|(h, w)| format!("{h}x{w}")).collect();
    artifacts().join(format!("pcs_commit_core_{}.mlirbc", stem.join("_")))
}

/// Only `commit` is exercised here, but the type takes both cores, so this
/// points at whichever open core is around. `open` is not wired yet.
fn open_core_path() -> PathBuf {
    artifacts().join("pcs_open_core_16x2p2_8x1p1__16x4p2_4x3p1.mlirbc")
}

/// Position-dependent values, matching the golden emitter: a matrix landing in
/// the wrong slot cannot coincidentally still match.
fn matrices() -> Vec<RowMajorMatrix<Val>> {
    SHAPES
        .iter()
        .enumerate()
        .map(|(m, &(height, width))| {
            let values = (0..height * width)
                .map(|i| Val::from_canonical_usize(1 + i + 100 * m))
                .collect();
            RowMajorMatrix::new(values, width)
        })
        .collect()
}

fn domains_and_matrices(
    p: &MyPcs,
) -> Vec<(
    <MyPcs as Pcs<Challenge, Challenger>>::Domain,
    RowMajorMatrix<Val>,
)> {
    matrices()
        .into_iter()
        .map(|m| {
            let domain =
                <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(p, m.height());
            (domain, m)
        })
        .collect()
}

fn gpu_pcs() -> TwoAdicFriPcs {
    TwoAdicFriPcs::new(pcs(), &commit_core_path(), &open_core_path()).unwrap_or_else(|e| {
        panic!(
            "{e}\n  export it: bazel run //export:export_pcs_commit_core -- --shapes={}",
            SHAPES
                .iter()
                .map(|(h, w)| format!("{h}x{w}"))
                .collect::<Vec<_>>()
                .join(",")
        )
    })
}

#[test]
#[ignore = "needs a GPU and an exported commit core (see the module docs)"]
fn commitment_matches_the_reference() {
    let reference = pcs();
    let (want, _) =
        <MyPcs as Pcs<Challenge, Challenger>>::commit(&reference, domains_and_matrices(&reference));
    let (got, _) = Pcs::<Challenge, Challenger>::commit(&gpu_pcs(), domains_and_matrices(&reference));

    assert_eq!(
        serde_json::to_value(got).unwrap(),
        serde_json::to_value(want).unwrap(),
        "GPU commitment differs from TwoAdicFriPcs::commit"
    );
}

#[test]
#[ignore = "needs a GPU and an exported commit core (see the module docs)"]
fn evaluations_on_domain_match_the_reference() {
    // What the consumer's quotient stage reads. It runs inside a rayon
    // par_iter, so this path is CPU-side by construction — but it still has to
    // produce the reference's extensions.
    let reference = pcs();
    let gpu = gpu_pcs();

    let (_, want_data) =
        <MyPcs as Pcs<Challenge, Challenger>>::commit(&reference, domains_and_matrices(&reference));
    let (_, got_data) =
        Pcs::<Challenge, Challenger>::commit(&gpu, domains_and_matrices(&reference));

    for (idx, &(height, _)) in SHAPES.iter().enumerate() {
        // The reference only serves evaluations on the generator coset — the
        // shift a quotient domain carries — so a natural domain would trip its
        // own assertion.
        let domain = TwoAdicMultiplicativeCoset {
            log_n: height.trailing_zeros() as usize,
            shift: Val::GENERATOR,
        };
        let want = <MyPcs as Pcs<Challenge, Challenger>>::get_evaluations_on_domain(
            &reference, &want_data, idx, domain,
        )
        .to_row_major_matrix();
        let got =
            Pcs::<Challenge, Challenger>::get_evaluations_on_domain(&gpu, &got_data, idx, domain)
                .to_row_major_matrix();
        assert_eq!(got.values, want.values, "matrix {idx}");
    }
}
