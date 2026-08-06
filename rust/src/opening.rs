//! Decoding the open core's outputs into the reference's proof types.
//!
//! The core returns 29 flat buffers; `OpenedValues` and `FriProof` are nested.
//! Every field of both is public, so this is direct construction rather than a
//! serde hop — unlike `crate::proof`, which rebuilds a `Proof` whose fields are
//! private.
//!
//! The layout is not hardcoded. It follows from the argument's shape — how many
//! matrices each round holds, how wide they are, how many points each is opened
//! at, and how many fold layers the tallest height implies — so a differently
//! shaped opening decodes without touching this file.

use p3_commit::OpenedValues;
use p3_fri::{BatchOpening, CommitPhaseProofStep, FriProof, QueryProof};
use pico_zorch_golden::{Challenge, Val, ValMmcs};

use crate::transcript;
use crate::wire;

/// Digest width of Pico's Merkle commitments.
const DIGEST: usize = 8;

type Commitment = <ValMmcs as p3_commit::Mmcs<Val>>::Commitment;
type Proof = FriProof<Challenge, ChallengeMmcsOf, Val, Vec<BatchOpening<Val, ValMmcs>>>;
type ChallengeMmcsOf = pico_zorch_golden::ChallengeMmcs;

/// How many buffers each section of the output occupies.
///
/// Derived from the argument rather than the manifest: the manifest records
/// shapes, but only the caller knows which matrix belongs to which round.
pub struct OpenShape {
    /// `[round][matrix] = (width, points)`.
    pub rounds: Vec<Vec<(usize, usize)>>,
    pub layers: usize,
    pub queries: usize,
}

impl OpenShape {
    fn matrices(&self) -> usize {
        self.rounds.iter().map(|r| r.len()).sum()
    }

    fn openings(&self) -> usize {
        self.rounds
            .iter()
            .flat_map(|r| r.iter())
            .map(|(_, points)| *points)
            .sum()
    }

    /// Total buffers the core returns, so a mismatch is caught here rather
    /// than as a confusing decode further down.
    pub fn outputs(&self) -> usize {
        self.openings()          // opened values, round -> matrix -> point
            + self.layers        // commit phase roots
            + 3                  // final_poly, pow_witness, query indices
            + self.matrices()    // input rows, one per matrix
            + self.rounds.len()  // input paths, one per round
            + 1                  // fold-layer siblings, stacked
            + self.layers        // fold-layer paths, ragged so one each
            + 5 // sponge state
    }
}

/// A cursor over the flat outputs, so each section is read in order and the
/// offsets cannot drift apart from the layout above.
struct Cursor<'a> {
    raw: &'a [Vec<u8>],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn next(&mut self) -> Result<&'a [u8], String> {
        let out = self
            .raw
            .get(self.at)
            .ok_or_else(|| format!("core returned {} outputs, wanted more", self.raw.len()))?;
        self.at += 1;
        Ok(out)
    }

    fn vals(&mut self) -> Result<Vec<Val>, String> {
        wire::vals(self.next()?)
    }

    fn challenges(&mut self) -> Result<Vec<Challenge>, String> {
        let base = self.vals()?;
        wire::challenges(&base)
    }

    /// A `[queries, n, DIGEST]` buffer as one path per query.
    fn paths(&mut self, queries: usize) -> Result<Vec<Vec<[Val; DIGEST]>>, String> {
        let flat = self.vals()?;
        let per_query = flat.len() / queries.max(1);
        if per_query % DIGEST != 0 || per_query * queries != flat.len() {
            return Err(format!(
                "path buffer of {} elements does not divide into {queries} queries of \
                 {DIGEST}-element digests",
                flat.len()
            ));
        }
        Ok(flat
            .chunks_exact(per_query)
            .map(|q| {
                q.chunks_exact(DIGEST)
                    .map(|d| <[Val; DIGEST]>::try_from(d).expect("chunked to DIGEST"))
                    .collect()
            })
            .collect())
    }
}

/// `(opened values, proof, sponge state)` from the core's outputs.
pub fn decode(
    shape: &OpenShape,
    raw: &[Vec<u8>],
) -> Result<(OpenedValues<Challenge>, Proof, transcript::State), String> {
    if raw.len() != shape.outputs() {
        return Err(format!(
            "core returned {} outputs but this argument's shape wants {}",
            raw.len(),
            shape.outputs()
        ));
    }
    let mut c = Cursor { raw, at: 0 };

    // Opened values, in the order the argument declares them.
    let mut opened: OpenedValues<Challenge> = Vec::with_capacity(shape.rounds.len());
    for round in &shape.rounds {
        let mut per_round = Vec::with_capacity(round.len());
        for &(_, points) in round {
            let mut per_matrix = Vec::with_capacity(points);
            for _ in 0..points {
                per_matrix.push(c.challenges()?);
            }
            per_round.push(per_matrix);
        }
        opened.push(per_round);
    }

    let mut commit_phase_commits = Vec::with_capacity(shape.layers);
    for _ in 0..shape.layers {
        let digest = <[Val; DIGEST]>::try_from(&c.vals()?[..])
            .map_err(|_| "a commit phase root is not 8 elements".to_string())?;
        commit_phase_commits.push(Commitment::from(digest));
    }

    let final_poly = wire::challenge(&c.vals()?)?;
    let pow_witness = *c
        .vals()?
        .first()
        .ok_or_else(|| "pow witness buffer is empty".to_string())?;
    // Indices are sampled but not carried in the proof; the verifier resamples
    // them. Read past them so the cursor stays aligned.
    let _indices = c.next()?;

    // Input openings: rows per matrix, then one path per round.
    let mut rows_by_round = Vec::with_capacity(shape.rounds.len());
    for round in &shape.rounds {
        let mut per_matrix = Vec::with_capacity(round.len());
        for &(width, _) in round {
            // One row per query, each `width` wide. `wire::rows` splits into a
            // row *count*, so this is the query count, not the width — and the
            // width is checked rather than assumed, since both are plausible
            // divisors of the same buffer.
            let rows = wire::rows(&c.vals()?, shape.queries)?;
            if rows.first().is_some_and(|r| r.len() != width) {
                return Err(format!(
                    "input rows are {} wide, want {width}",
                    rows[0].len()
                ));
            }
            per_matrix.push(rows);
        }
        rows_by_round.push(per_matrix);
    }
    let mut paths_by_round = Vec::with_capacity(shape.rounds.len());
    for _ in &shape.rounds {
        paths_by_round.push(c.paths(shape.queries)?);
    }

    // Fold chain: siblings stacked over layers, paths ragged so one each.
    let siblings = c.challenges()?;
    let mut layer_paths = Vec::with_capacity(shape.layers);
    for _ in 0..shape.layers {
        layer_paths.push(c.paths(shape.queries)?);
    }

    let query_proofs = (0..shape.queries)
        .map(|q| QueryProof {
            input_proof: (0..shape.rounds.len())
                .map(|r| BatchOpening {
                    opened_values: rows_by_round[r]
                        .iter()
                        .map(|per_matrix| per_matrix[q].clone())
                        .collect(),
                    opening_proof: paths_by_round[r][q].clone(),
                })
                .collect(),
            commit_phase_openings: (0..shape.layers)
                .map(|l| CommitPhaseProofStep {
                    sibling_value: siblings[q * shape.layers + l],
                    opening_proof: layer_paths[l][q].clone(),
                })
                .collect(),
        })
        .collect();

    let state = transcript::State {
        input_buffer: c.next()?.to_vec(),
        output_buffer: c.next()?.to_vec(),
        sponge_state: c.next()?.to_vec(),
        in_pos: c.next()?.to_vec(),
        out_pos: c.next()?.to_vec(),
    };

    Ok((
        opened,
        FriProof {
            commit_phase_commits,
            query_proofs,
            final_poly,
            pow_witness,
        },
        state,
    ))
}
