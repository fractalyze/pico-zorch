//! Carrying the challenger across a stage boundary.
//!
//! Pico drives its transcript on the host and calls the PCS as separate
//! stages, so an exported stage has to receive the sponge and hand it back.
//! The core takes it as five fixed-size buffers; this converts to and from
//! Plonky3's `DuplexChallenger`.
//!
//! The two representations agree on more than they look like they do. Both
//! keep a *prefix* of values still to be consumed and take from the back:
//! Plonky3's `sample` is `output_buffer.pop()`, and zorch's is
//! `out_pos -= 1; output_buffer[out_pos]`. So the conversion is a copy and a
//! pad, not a reordering — the padding beyond each position is never read.

use p3_field::FieldAlgebra;
use pico_zorch_golden::{Challenger, Val};

use crate::wire;

/// Sponge width and rate of Pico's challenger.
const WIDTH: usize = 16;
const RATE: usize = 8;

/// The core's view of the sponge: the five buffers it takes and returns.
///
/// Held as bytes because that is what crosses to the device, and the two
/// integer positions are `i32` while the rest are field elements — so there is
/// no single element type to hold them in.
pub struct State {
    pub input_buffer: Vec<u8>,
    pub output_buffer: Vec<u8>,
    pub sponge_state: Vec<u8>,
    pub in_pos: Vec<u8>,
    pub out_pos: Vec<u8>,
}

impl State {
    /// The core's inputs in manifest order.
    pub fn args(&self) -> [&[u8]; 5] {
        [
            &self.input_buffer,
            &self.output_buffer,
            &self.sponge_state,
            &self.in_pos,
            &self.out_pos,
        ]
    }
}

fn padded(values: &[Val], len: usize) -> Vec<u8> {
    let mut out = values.to_vec();
    out.resize(len, Val::ZERO);
    wire::as_bytes(&out).to_vec()
}

/// Read a challenger's state for the core.
pub fn to_state(challenger: &Challenger) -> State {
    assert!(
        challenger.input_buffer.len() <= RATE && challenger.output_buffer.len() <= RATE,
        "challenger buffers exceed the sponge rate"
    );
    State {
        input_buffer: padded(&challenger.input_buffer, RATE),
        output_buffer: padded(&challenger.output_buffer, RATE),
        sponge_state: wire::as_bytes(&challenger.sponge_state).to_vec(),
        in_pos: (challenger.input_buffer.len() as i32).to_le_bytes().to_vec(),
        out_pos: (challenger.output_buffer.len() as i32).to_le_bytes().to_vec(),
    }
}

/// Write the core's returned state back, so the consumer keeps transcripting.
///
/// Truncates each buffer to its position: the core's buffers are always
/// rate-wide, and the values past the position are padding the sponge never
/// looks at. Carrying them over would leave the challenger claiming values it
/// does not have.
pub fn from_state(challenger: &mut Challenger, state: &State) -> Result<(), String> {
    let input = wire::vals(&state.input_buffer)?;
    let output = wire::vals(&state.output_buffer)?;
    let sponge = wire::vals(&state.sponge_state)?;
    let in_pos = read_pos(&state.in_pos, "in_pos")?;
    let out_pos = read_pos(&state.out_pos, "out_pos")?;

    if sponge.len() != WIDTH {
        return Err(format!("sponge state is {} wide, want {WIDTH}", sponge.len()));
    }
    if in_pos > RATE || out_pos > RATE {
        return Err(format!("positions {in_pos}/{out_pos} exceed the rate {RATE}"));
    }

    challenger.sponge_state = <[Val; WIDTH]>::try_from(&sponge[..]).expect("checked above");
    challenger.input_buffer = input[..in_pos].to_vec();
    challenger.output_buffer = output[..out_pos].to_vec();
    Ok(())
}

fn read_pos(bytes: &[u8], what: &str) -> Result<usize, String> {
    let raw: [u8; 4] = bytes
        .try_into()
        .map_err(|_| format!("{what} is {} bytes, want 4", bytes.len()))?;
    let value = i32::from_le_bytes(raw);
    usize::try_from(value).map_err(|_| format!("{what} is negative ({value})"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_challenger::{CanObserve, CanSample, FieldChallenger};
    use pico_zorch_golden::{pico_perm, Challenge};

    /// Mid-absorb: values pending in the input buffer.
    ///
    /// The two buffers are never both non-empty — `observe` clears the output
    /// buffer and the duplexing behind `sample` clears the input one — so the
    /// two states are tested separately rather than sought in one fixture.
    fn mid_absorb() -> Challenger {
        let mut c = Challenger::new(pico_perm());
        c.observe_slice(&(0..13).map(Val::from_canonical_u32).collect::<Vec<_>>());
        let _: Challenge = c.sample_ext_element();
        c.observe_slice(&(0..3).map(Val::from_canonical_u32).collect::<Vec<_>>());
        assert!(!c.input_buffer.is_empty() && c.output_buffer.is_empty());
        c
    }

    /// Mid-squeeze: an output buffer partly drained, so `out_pos` is neither
    /// zero nor the full rate.
    fn mid_squeeze() -> Challenger {
        let mut c = Challenger::new(pico_perm());
        c.observe_slice(&(0..13).map(Val::from_canonical_u32).collect::<Vec<_>>());
        let _: Challenge = c.sample_ext_element();
        let _: Val = c.sample();
        assert!(c.input_buffer.is_empty() && !c.output_buffer.is_empty());
        assert!(c.output_buffer.len() < RATE, "buffer must be partly drained");
        c
    }

    #[test]
    fn round_trip_preserves_the_sponge() {
        for original in [mid_absorb(), mid_squeeze()] {
            let mut restored = Challenger::new(pico_perm());
            from_state(&mut restored, &to_state(&original)).expect("round trip");

            assert_eq!(restored.sponge_state, original.sponge_state);
            assert_eq!(restored.input_buffer, original.input_buffer);
            assert_eq!(restored.output_buffer, original.output_buffer);
        }
    }

    #[test]
    fn a_restored_challenger_samples_the_same_continuation() {
        // The property that matters: state equality is a means, not the end.
        // Two challengers agreeing on their next twenty samples is what makes
        // a stage boundary invisible to the protocol.
        for mut original in [mid_absorb(), mid_squeeze()] {
            let mut restored = Challenger::new(pico_perm());
            from_state(&mut restored, &to_state(&original)).expect("round trip");

            for i in 0..20 {
                let a: Val = original.sample();
                let b: Val = restored.sample();
                assert_eq!(a, b, "sample {i} diverged after the round trip");
            }
        }
    }

    #[test]
    fn padding_past_a_position_is_not_carried_over() {
        // The core's buffers are always rate-wide; the values past each
        // position are padding it never reads. Carrying them back would leave
        // the challenger claiming values it does not have.
        let original = mid_absorb();
        let mut state = to_state(&original);
        // Scribble on the padding — a correct restore ignores it.
        let scribble = wire::as_bytes(&[Val::from_canonical_u32(7); RATE]).to_vec();
        let keep = original.input_buffer.len() * std::mem::size_of::<Val>();
        state.input_buffer[keep..].copy_from_slice(&scribble[keep..]);

        let mut restored = Challenger::new(pico_perm());
        from_state(&mut restored, &state).expect("round trip");
        assert_eq!(restored.input_buffer, original.input_buffer);
    }
}
