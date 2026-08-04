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
//! # The cost of swapping only the commit
//!
//! Because the opening argument stays on CPU, `commit` has to bring the full
//! LDE and every digest layer back to the host — roughly 380 MB per commit at
//! 2^20 x 32. This step therefore buys a byte-matched seam and a measurement
//! harness, not a speedup; the win arrives when `open` moves too and the data
//! stays device-resident. That ordering is deliberate: it gets a golden anchor
//! in place before the larger change.

use std::marker::PhantomData;

use p3_commit::{Mmcs, Pcs, TwoAdicMultiplicativeCoset};
use p3_field::FieldAlgebra;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use pico_zorch_golden::{Challenge, Challenger, MyPcs, Val, ValMmcs};
use serde::Serialize;

use crate::gpu::{self, Core};
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

/// Pico's `TwoAdicFriPcs` with the commit running on an exported core.
pub struct TwoAdicFriPcs {
    /// The reference, for everything except `commit`.
    inner: MyPcs,
    core: &'static Core,
}

impl TwoAdicFriPcs {
    /// Wrap the reference PCS, committing through the core at `core_path`.
    ///
    /// The core is compiled on first use and cached for the process, so a
    /// prover committing many batches of one shape pays that once.
    pub fn new(inner: MyPcs, core_path: &std::path::Path) -> Result<Self, String> {
        Ok(Self {
            inner,
            core: gpu::load(core_path)?,
        })
    }

    /// The reference `MerkleTree`, rebuilt from the core's outputs.
    ///
    /// Its fields are public but a private `PhantomData` blocks a struct
    /// literal, so this goes through serde — the same door `proof::assemble`
    /// uses for `Proof`. bincode rather than JSON: the hop carries no
    /// information and this one moves the whole LDE.
    fn prover_data(&self, raw: &[Vec<u8>]) -> Result<(Commitment, ProverData), String> {
        let manifest = &self.core.manifest;
        let index = manifest.output_index();
        let fetch = |name: &str| -> Result<Vec<Val>, String> {
            let i = *index
                .get(name)
                .ok_or_else(|| format!("core has no output {name:?}"))?;
            wire::vals(&raw[i]).map_err(|e| format!("output {name:?}: {e}"))
        };
        let dims = |name: &str| -> Result<&[usize], String> {
            let i = *index
                .get(name)
                .ok_or_else(|| format!("core has no output {name:?}"))?;
            Ok(&manifest.outputs[i].dims)
        };

        let root = <[Val; DIGEST]>::try_from(&fetch("root")?[..])
            .map_err(|_| "root must be 8 elements".to_string())?;

        let mut leaves = Vec::new();
        for i in 0.. {
            let name = format!("lde{i}");
            if !index.contains_key(name.as_str()) {
                break;
            }
            let width = *dims(&name)?
                .get(1)
                .ok_or_else(|| format!("{name} must be [height, width]"))?;
            leaves.push(RowMajorMatrix::new(fetch(&name)?, width));
        }
        if leaves.is_empty() {
            return Err("core returned no extensions".into());
        }

        let mut digest_layers: Vec<Vec<[Val; DIGEST]>> = Vec::new();
        for i in 0.. {
            let name = format!("digest_layer{i}");
            if !index.contains_key(name.as_str()) {
                break;
            }
            digest_layers.push(
                fetch(&name)?
                    .chunks_exact(DIGEST)
                    .map(|c| <[Val; DIGEST]>::try_from(c).expect("chunk is one digest"))
                    .collect(),
            );
        }

        let wire_tree = MerkleTreeWire {
            leaves,
            digest_layers,
            _phantom: PhantomData,
        };
        let encoded = bincode::serialize(&wire_tree)
            .map_err(|e| format!("serialize the committed tree: {e}"))?;
        let data = bincode::deserialize(&encoded)
            .map_err(|e| format!("rebuild MerkleTree: {e}"))?;
        Ok((Commitment::from(root), data))
    }
}

/// Field-for-field mirror of Plonky3's `MerkleTree`, including the private
/// `PhantomData` — bincode matches by position, so a missing field would
/// shift every later one.
#[derive(Serialize)]
struct MerkleTreeWire {
    leaves: Vec<RowMajorMatrix<Val>>,
    digest_layers: Vec<Vec<[Val; DIGEST]>>,
    _phantom: PhantomData<Val>,
}

impl Pcs<Challenge, Challenger> for TwoAdicFriPcs {
    type Domain = Domain;
    type Commitment = Commitment;
    type ProverData = ProverData;
    type Proof = <MyPcs as Pcs<Challenge, Challenger>>::Proof;
    type Error = <MyPcs as Pcs<Challenge, Challenger>>::Error;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        <Reference as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&self.inner, degree)
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
        let matrices: Vec<&[Val]> = evaluations.iter().map(|(_, m)| &m.values[..]).collect();
        let (raw, _) = gpu::run_many(self.core, &matrices).expect("run the commit core");
        self.prover_data(&raw).expect("rebuild the committed tree")
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> impl Matrix<Val> + 'a {
        <Reference as Pcs<Challenge, Challenger>>::get_evaluations_on_domain(&self.inner, prover_data, idx, domain)
    }

    fn open(
        &self,
        rounds: Vec<(&Self::ProverData, Vec<Vec<Challenge>>)>,
        challenger: &mut Challenger,
    ) -> (p3_commit::OpenedValues<Challenge>, Self::Proof) {
        <Reference as Pcs<Challenge, Challenger>>::open(&self.inner, rounds, challenger)
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
        <Reference as Pcs<Challenge, Challenger>>::verify(&self.inner, rounds, proof, challenger)
    }
}
