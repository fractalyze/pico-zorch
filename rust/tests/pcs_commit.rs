//! The GPU `Pcs::commit` is the reference's, commitment and prover data alike.
//!
//! Matching the root is necessary but not sufficient: the opening argument runs
//! on the reference's untouched CPU path and reads its matrices and siblings out
//! of our `ProverData`, so a tree that roots correctly but stores the wrong
//! leaves would pass a root check and then fail — or worse, silently open — much
//! later. This compares the whole prover data and then actually opens.
//!
//! Needs a GPU and a core exported for the same batch shape:
//!
//! ```sh
//! bazel run //export:export_pcs_commit_core -- --shapes=4x3,16x2,8x1,16x4
//! cargo test --test pcs_commit -- --ignored --test-threads=1
//! ```

use std::path::PathBuf;

use p3_commit::Pcs;
use p3_field::FieldAlgebra;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use pico_zorch::{pcs::TwoAdicFriPcs, Challenge, Challenger, Val};
use pico_zorch_golden::{pcs, pico_perm, MyPcs};

/// The shapes `//export:export_pcs_commit_core --shapes=4x3,16x2,8x1,16x4`
/// produces — the same batch `golden`'s `emit_batch_commit` pins, deliberately
/// unsorted so a consumer that forgets to sort by height fails.
const SHAPES: [(usize, usize); 4] = [(4, 3), (16, 2), (8, 1), (16, 4)];

fn core_path() -> PathBuf {
    if let Ok(explicit) = std::env::var("PICO_ZORCH_PCS_CORE_MLIRBC") {
        return PathBuf::from(explicit);
    }
    let stem: Vec<String> = SHAPES.iter().map(|(h, w)| format!("{h}x{w}")).collect();
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate has a parent directory")
        .join("artifacts")
        .join(format!("pcs_commit_core_{}.mlirbc", stem.join("_")))
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
    TwoAdicFriPcs::new(pcs(), &core_path()).unwrap_or_else(|e| {
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
fn prover_data_matches_the_reference() {
    // The opening path reads this, so an equal root over unequal prover data
    // would surface as a failure far from its cause.
    let reference = pcs();
    let (_, want) =
        <MyPcs as Pcs<Challenge, Challenger>>::commit(&reference, domains_and_matrices(&reference));
    let (_, got) = Pcs::<Challenge, Challenger>::commit(&gpu_pcs(), domains_and_matrices(&reference));

    assert_eq!(
        serde_json::to_value(got).unwrap(),
        serde_json::to_value(want).unwrap(),
        "GPU prover data differs from the reference's"
    );
}

#[test]
#[ignore = "needs a GPU and an exported commit core (see the module docs)"]
fn reference_opening_accepts_our_prover_data() {
    // The end the swap exists for: commit on GPU, then let Pico's untouched
    // opening argument run against it.
    let reference = pcs();
    let gpu = gpu_pcs();
    let (commit, data) = Pcs::<Challenge, Challenger>::commit(&gpu, domains_and_matrices(&reference));

    let zeta = Challenge::from_canonical_u32(97);
    let points: Vec<Vec<Challenge>> = SHAPES.iter().map(|_| vec![zeta]).collect();

    let mut prover_challenger = Challenger::new(pico_perm());
    let (opened, proof) = Pcs::<Challenge, Challenger>::open(
        &gpu,
        vec![(&data, points.clone())],
        &mut prover_challenger,
    );

    let rounds = SHAPES
        .iter()
        .zip(&opened[0])
        .map(|(&(height, _), values)| {
            let domain =
                <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&reference, height);
            (domain, vec![(zeta, values[0].clone())])
        })
        .collect();

    let mut verifier_challenger = Challenger::new(pico_perm());
    <MyPcs as Pcs<Challenge, Challenger>>::verify(
        &reference,
        vec![(commit, rounds)],
        &proof,
        &mut verifier_challenger,
    )
    .expect("the reference verifier must accept an opening of our commitment");
}
