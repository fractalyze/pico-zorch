//! `Pcs` over the GPU commit — the seam Pico's machine prover swaps at.
//!
//! `vm/src/machine/prover.rs` reaches every heavy kernel through `config.pcs()`,
//! so accelerating `commit_main` needs no fork of Pico: it is a config type
//! change from `p3_fri::TwoAdicFriPcs` to this.
//!
//! Only [`Pcs::commit`] is ours. Everything else delegates to the reference,
//! and can, because our `ProverData` *is* the reference's `MerkleTree` — the
//! GPU returns the extensions and digest layers, and we rebuild that struct.
//! `get_evaluations_on_domain` is then pure host-side slicing the reference
//! already does correctly, and `open` runs its untouched CPU path.
//!
//! # What crosses to the host
//!
//! `commit` reads back only the commitment; the extensions and digest layers
//! stay on the device and `open` consumes them there. Between the two stages
//! the host sees 32 bytes of root and the sponge state.
//!
//! [`Pcs::get_evaluations_on_domain`] is the exception, and deliberately so.
//! Pico calls it from inside a rayon `par_iter` while computing quotients, and
//! this crate's PJRT session is thread-local — a second client in one process
//! aborts — so that method must not touch the device at all. It rebuilds the
//! extension on the CPU from the trace instead, once, cached. The cost is paid
//! only if the consumer's quotient stage actually reads it, and it disappears
//! when that stage moves to the device too.

use std::sync::{Arc, OnceLock};

use p3_commit::{Mmcs, Pcs, PolynomialSpace, TwoAdicMultiplicativeCoset};
use p3_field::{Field, FieldAlgebra};
use p3_matrix::bitrev::BitReversableMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_dft::TwoAdicSubgroupDft;
use p3_matrix::Matrix;
use p3_field::FieldExtensionAlgebra;
use pico_zorch_golden::{
    Challenge, Challenger, Dft, MyPcs, Val, ValMmcs, LOG_BLOWUP, NUM_QUERIES,
};
use serde::{Deserialize, Serialize};

use crate::gpu::{self, Core};
use crate::opening::{self, OpenShape};
use crate::transcript;
use crate::wire;

/// Digest width of Pico's Merkle commitments.
const DIGEST: usize = 8;

/// The reference impl every method but `commit` forwards to. Spelled out at
/// each call site because `MyPcs` implements `Pcs<Challenge, _>` for more than
/// one challenger, which makes an inherent-looking call ambiguous.
type Reference = MyPcs;

type Commitment = <ValMmcs as Mmcs<Val>>::Commitment;
type ProverData = <ValMmcs as Mmcs<Val>>::ProverData<RowMajorMatrix<Val>>;
type Domain = TwoAdicMultiplicativeCoset<Val>;

/// Pico's `TwoAdicFriPcs` with commit and open running on exported cores.
pub struct TwoAdicFriPcs {
    /// The reference, for `verify` and the domain arithmetic.
    inner: MyPcs,
    commit_core: &'static Core,
    open_core: &'static Core,
}

/// What `commit` hands to `open`.
///
/// The matrices are the committed traces: small, serializable, and enough to
/// rebuild everything else. `device` is a handle to the extensions and digest
/// layers left on the device — a cache, so it is skipped by serde and absent
/// after a round trip or on another thread, which callers treat as
/// "re-upload" rather than as an error.
#[derive(Clone, Serialize, Deserialize)]
pub struct CommittedData {
    matrices: Vec<HostMatrix>,
    #[serde(skip)]
    device: Option<gpu::Handle>,
    /// CPU-rebuilt extensions, for `get_evaluations_on_domain`. Built at most
    /// once and only if something asks.
    #[serde(skip)]
    ldes: Arc<OnceLock<Vec<RowMajorMatrix<Val>>>>,
}

/// A matrix in the shape serde can carry.
#[derive(Clone, Serialize, Deserialize)]
struct HostMatrix {
    values: Vec<Val>,
    width: usize,
}

impl TwoAdicFriPcs {
    /// Wrap the reference PCS, running commit and open on exported cores.
    ///
    /// Both cores are compiled on first use and cached for the process, so a
    /// prover committing many batches of one shape pays that once.
    pub fn new(
        inner: MyPcs,
        commit_core: &std::path::Path,
        open_core: &std::path::Path,
    ) -> Result<Self, String> {
        Ok(Self {
            inner,
            commit_core: gpu::load(commit_core)?,
            open_core: gpu::load(open_core)?,
        })
    }
}

impl TwoAdicFriPcs {
    /// The blowup the cores were exported with. A protocol constant here, not
    /// a knob: the commit core has it baked in, so a mismatch would be a
    /// differently-shaped executable, not a differently-configured one.
    fn log_blowup(&self) -> usize {
        LOG_BLOWUP
    }

    /// Fold layers the argument implies: one per halving from the tallest
    /// committed height down to the blowup.
    fn fold_layers(&self, rounds: &[(&CommittedData, Vec<Vec<Challenge>>)]) -> usize {
        let tallest = rounds
            .iter()
            .flat_map(|(data, _)| data.matrices.iter())
            .map(|m| (m.values.len() / m.width) << self.log_blowup())
            .max()
            .expect("an opening has at least one matrix");
        tallest.trailing_zeros() as usize - self.log_blowup()
    }

    /// The commitment, the one output `commit` reads back.
    fn read_root(&self, handle: gpu::Handle) -> Commitment {
        let kept = gpu::resident(handle).expect("commit outputs are on this thread");
        let index = self.commit_core.manifest.output_index();
        let i = *index.get("root").expect("commit core emits a root");
        let bytes = gpu::to_host(&kept.buffers[i]);
        let vals = wire::vals(&bytes).expect("root buffer");
        let root = <[Val; DIGEST]>::try_from(&vals[..]).expect("root is 8 elements");
        Commitment::from(root)
    }
}

impl CommittedData {
    /// The extensions, rebuilt on the CPU and cached.
    ///
    /// Deliberately not a device call — see the module docs on why
    /// `get_evaluations_on_domain` cannot touch PJRT.
    fn ldes(&self, log_blowup: usize) -> &[RowMajorMatrix<Val>] {
        self.ldes.get_or_init(|| {
            let dft = Dft::default();
            self.matrices
                .iter()
                .map(|m| {
                    let mat = RowMajorMatrix::new(m.values.clone(), m.width);
                    dft.coset_lde_batch(mat, log_blowup, Val::GENERATOR)
                        .bit_reverse_rows()
                        .to_row_major_matrix()
                })
                .collect()
        })
    }
}

impl Pcs<Challenge, Challenger> for TwoAdicFriPcs {
    type Domain = Domain;
    type Commitment = Commitment;
    type ProverData = CommittedData;
    type Proof = <MyPcs as Pcs<Challenge, Challenger>>::Proof;
    type Error = <MyPcs as Pcs<Challenge, Challenger>>::Error;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        <Reference as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
            &self.inner,
            degree,
        )
    }

    fn commit(
        &self,
        evaluations: Vec<(Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        // The reference extends against `Val::GENERATOR / domain.shift`; the
        // core bakes in the natural-domain case, so anything else would be
        // committed on the wrong coset.
        for (domain, _) in &evaluations {
            assert_eq!(
                domain.shift,
                Val::ONE,
                "the exported core commits natural domains only"
            );
        }

        let matrices: Vec<HostMatrix> = evaluations
            .iter()
            .map(|(_, m)| HostMatrix {
                values: m.values.clone(),
                width: m.width(),
            })
            .collect();
        let inputs: Vec<&[Val]> = evaluations.iter().map(|(_, m)| &m.values[..]).collect();

        let (handle, _) =
            gpu::run_many_to_device(self.commit_core, &inputs).expect("run the commit core");
        let root = self.read_root(handle);

        (
            root,
            CommittedData {
                matrices,
                device: Some(handle),
                ldes: Arc::new(OnceLock::new()),
            },
        )
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> impl Matrix<Val> + 'a {
        // Runs inside the consumer's rayon pool; must not touch PJRT.
        assert_eq!(
            domain.shift,
            Val::GENERATOR,
            "evaluations are only held on the generator coset"
        );
        let lde = &prover_data.ldes(self.log_blowup())[idx];
        assert!(lde.height() >= domain.size());
        lde.split_rows(domain.size()).0.bit_reverse_rows()
    }

    fn open(
        &self,
        rounds: Vec<(&Self::ProverData, Vec<Vec<Challenge>>)>,
        challenger: &mut Challenger,
    ) -> (p3_commit::OpenedValues<Challenge>, Self::Proof) {
        let shape = OpenShape {
            rounds: rounds
                .iter()
                .map(|(data, points)| {
                    data.matrices
                        .iter()
                        .zip(points)
                        .map(|(m, pts)| (m.width, pts.len()))
                        .collect()
                })
                .collect(),
            layers: self.fold_layers(&rounds),
            queries: NUM_QUERIES,
        };

        let state = transcript::to_state(challenger);
        // Extensions the commit left on the device, or rebuilt if a handle no
        // longer resolves — a different thread, or a serde round trip.
        let fallback: Vec<Vec<u8>> = rounds
            .iter()
            .filter(|(data, _)| data.device.and_then(gpu::resident).is_none())
            .flat_map(|(data, _)| {
                data.ldes(self.log_blowup())
                    .iter()
                    .map(|m| wire::as_bytes(&m.values).to_vec())
                    .collect::<Vec<_>>()
            })
            .collect();
        let point_bytes: Vec<Vec<u8>> = rounds
            .iter()
            .flat_map(|(_, points)| points.iter().flatten())
            .map(|z| wire::as_bytes(z.as_base_slice()).to_vec())
            .collect();

        let mut args: Vec<gpu::Arg> =
            state.args().iter().map(|b| gpu::Arg::Host(b)).collect();
        let mut spare = fallback.iter();
        for (data, _) in &rounds {
            match data.device.and_then(gpu::resident) {
                // The commit core emits the root first, then one extension per
                // matrix, then the digest layers; only the extensions are
                // arguments here.
                Some(kept) => {
                    for i in 0..data.matrices.len() {
                        args.push(gpu::Arg::Resident(&kept.buffers[1 + i]));
                    }
                }
                None => {
                    for _ in 0..data.matrices.len() {
                        args.push(gpu::Arg::Host(
                            spare.next().expect("one fallback per matrix"),
                        ));
                    }
                }
            }
        }
        for bytes in &point_bytes {
            args.push(gpu::Arg::Host(bytes));
        }

        let (raw, _) = gpu::run_mixed(self.open_core, &args).expect("run the open core");
        let (opened, proof, next) =
            opening::decode(&shape, &raw).expect("decode the opening");
        transcript::from_state(challenger, &next).expect("restore the challenger");
        (opened, proof)
    }

    fn verify(
        &self,
        rounds: Vec<(
            Self::Commitment,
            Vec<(Self::Domain, Vec<(Challenge, Vec<Challenge>)>)>,
        )>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        <Reference as Pcs<Challenge, Challenger>>::verify(
            &self.inner,
            rounds,
            proof,
            challenger,
        )
    }
}
