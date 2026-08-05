//! The PJRT side: compile a core once, then run it per proof.
//!
//! A whole Pico proof is one executable, so there is exactly one dispatch and
//! one readback per call. The trace goes up as a reinterpreted byte view (see
//! [`crate::wire`]) — no per-element work on the host in either direction.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use pico_zorch_golden::Val;
use xla_pjrt::sys::PJRT_Buffer_Type_KOALABEAR_MONT as KOALABEAR_MONT;

use crate::manifest::Manifest;
use crate::wire;

/// The persistent client plus a cache of compiled cores keyed by path.
struct Gpu {
    session: xla_pjrt::Session,
    cores: RefCell<HashMap<String, &'static Core>>,
    /// Outputs a caller asked to keep on the device, by handle. Leaked to
    /// `'static` like the cores: a PJRT buffer outlives any borrow of the
    /// map, and the client it belongs to is never torn down anyway.
    resident: RefCell<HashMap<u64, &'static Resident>>,
    next_handle: RefCell<u64>,
}

/// One execution's outputs, still on the device.
pub struct Resident {
    pub buffers: Vec<xla_pjrt::Buffer>,
}

/// A handle to [`Resident`] buffers held by this thread's session.
///
/// Not the buffers themselves: the session is thread-local (a second PJRT
/// client in one process aborts), so a caller that could hold buffers directly
/// could move them to a thread that cannot use them. A handle fails to resolve
/// there instead, and the caller falls back to re-uploading.
pub type Handle = u64;

thread_local! {
    /// One [`Gpu`] per thread, leaked to `'static` — a second PJRT client in
    /// one process aborts, and tearing the client down against a live CUDA
    /// context can fault, so its destructor never runs.
    static GPU: RefCell<Option<&'static Gpu>> = const { RefCell::new(None) };
}

fn gpu() -> &'static Gpu {
    GPU.with(|cell| {
        *cell.borrow_mut().get_or_insert_with(|| {
            let session = unsafe { xla_pjrt::Session::new() };
            Box::leak(Box::new(Gpu {
                session,
                cores: RefCell::new(HashMap::new()),
                resident: RefCell::new(HashMap::new()),
                next_handle: RefCell::new(0),
            }))
        })
    })
}

/// A compiled core and the manifest describing its buffers.
pub struct Core {
    exe: xla_pjrt::Executable,
    pub manifest: Manifest,
}

/// Host-side phase timings for one proof (profiling aid).
#[derive(Clone, Copy, Default, Debug)]
pub struct Phases {
    /// Uploading the trace and public values.
    pub h2d: Duration,
    /// Enqueueing the executable.
    pub dispatch: Duration,
    /// Reading the proof back — dominated by waiting on the computation.
    pub readback: Duration,
    /// Rebuilding `p3_uni_stark::Proof` from the raw buffers. Pure host work,
    /// so it is worth watching: it is not amortised by anything.
    pub assemble: Duration,
}

/// Compile (once) the core at `path`, or return the cached one.
///
/// Compilation is per-process and can take a while for a large instance, so a
/// caller proving repeatedly should hold the returned reference rather than
/// calling this per proof (it is cheap after the first call, but not free).
pub fn load(path: &Path) -> Result<&'static Core, String> {
    let key = path.to_string_lossy().into_owned();
    let g = gpu();
    if let Some(core) = g.cores.borrow().get(&key) {
        return Ok(core);
    }
    let manifest = Manifest::beside(path)?;
    let code = std::fs::read(path).map_err(|e| format!("read core {}: {e}", path.display()))?;
    let exe = unsafe { g.session.compile(&code) };
    let core: &'static Core = Box::leak(Box::new(Core { exe, manifest }));
    g.cores.borrow_mut().insert(key, core);
    Ok(core)
}

/// Run `core`, keeping its outputs on the device.
///
/// Returns `(handle, phases)`; resolve the handle with [`resident`] on the same
/// thread. Use when the outputs are an intermediate the host never reads — a
/// commit whose extensions only the opening consumes, for instance. The
/// readback phase is zero by construction, so the dispatch time is the whole
/// cost.
pub fn run_many_to_device(
    core: &Core,
    inputs: &[&[Val]],
) -> Result<(Handle, Phases), String> {
    let s = gpu();
    let (buffers, h2d) = upload(s, core, inputs)?;
    let refs: Vec<&xla_pjrt::Buffer> = buffers.iter().collect();

    let t = Instant::now();
    let outs = unsafe {
        s.session
            .run_buffers_to_device(&core.exe, &refs, core.manifest.outputs.len())
    };
    let dispatch = t.elapsed();

    // The inputs have been consumed by the execution; only the outputs are
    // worth keeping.
    buffers.into_iter().for_each(|b| unsafe { s.session.free_buffer(b) });

    let handle = {
        let mut next = s.next_handle.borrow_mut();
        *next += 1;
        *next
    };
    let kept: &'static Resident = Box::leak(Box::new(Resident { buffers: outs }));
    s.resident.borrow_mut().insert(handle, kept);
    Ok((
        handle,
        Phases {
            h2d,
            dispatch,
            readback: Duration::default(),
            assemble: Duration::default(),
        },
    ))
}

/// Copy one resident buffer to the host.
///
/// For the rare output a caller does need — a commitment, say — without
/// dragging the rest of an execution's outputs back with it.
pub fn to_host(buffer: &xla_pjrt::Buffer) -> Vec<u8> {
    unsafe { gpu().session.buffer_to_host(buffer) }
}

/// The buffers behind `handle`, if this thread's session holds them.
///
/// `None` on a different thread than the one that produced them, or after the
/// owning data round-tripped through serde. Callers treat that as "re-upload",
/// not as an error.
pub fn resident(handle: Handle) -> Option<&'static Resident> {
    gpu().resident.borrow().get(&handle).copied()
}

/// Release a handle's device memory.
pub fn release(handle: Handle) {
    let s = gpu();
    if let Some(kept) = s.resident.borrow_mut().remove(&handle) {
        // SAFETY: `run_many_to_device` leaked this box and the map held the
        // only pointer to it, which the `remove` above just took. Reclaiming
        // it here frees the allocation as well as the buffers it owns.
        let owned = unsafe { Box::from_raw(kept as *const Resident as *mut Resident) };
        for buffer in owned.buffers {
            unsafe { s.session.free_buffer(buffer) };
        }
    }
}

/// Run `core` against buffers already on the device, reading its outputs back.
pub fn run_resident(
    core: &Core,
    inputs: &[&xla_pjrt::Buffer],
) -> Result<(Vec<Vec<u8>>, Phases), String> {
    let s = gpu();
    let (outs, dispatch, readback) = unsafe {
        s.session
            .run_buffers_timed(&core.exe, inputs, core.manifest.outputs.len())
    };
    Ok((
        outs,
        Phases {
            h2d: Duration::default(),
            dispatch,
            readback,
            assemble: Duration::default(),
        },
    ))
}

/// Upload each input as a device buffer, checking it against the manifest.
fn upload(
    s: &'static Gpu,
    core: &Core,
    inputs: &[&[Val]],
) -> Result<(Vec<xla_pjrt::Buffer>, Duration), String> {
    let m = &core.manifest;
    if inputs.len() != m.inputs.len() {
        return Err(format!(
            "core takes {} inputs but {} were given",
            m.inputs.len(),
            inputs.len()
        ));
    }
    for (i, (given, spec)) in inputs.iter().zip(&m.inputs).enumerate() {
        if given.len() != spec.len() {
            return Err(format!(
                "input {i} ({}) has {} elements but the core expects {} ({:?})",
                spec.name,
                given.len(),
                spec.len(),
                spec.dims
            ));
        }
    }
    let t = Instant::now();
    let buffers = inputs
        .iter()
        .zip(&m.inputs)
        .map(|(values, spec)| {
            let dims: Vec<i64> = spec.dims.iter().map(|&n| n as i64).collect();
            unsafe { s.session.input_buffer(wire::as_bytes(values), &dims, KOALABEAR_MONT) }
        })
        .collect();
    Ok((buffers, t.elapsed()))
}

/// Run `core` over an arbitrary list of field-array inputs, in manifest order.
///
/// The generic sibling of [`run`]: a PCS commit takes one matrix per chip, so
/// its arity is a property of the exported core rather than of the protocol.
pub fn run_many(core: &Core, inputs: &[&[Val]]) -> Result<(Vec<Vec<u8>>, Phases), String> {
    let m = &core.manifest;
    if inputs.len() != m.inputs.len() {
        return Err(format!(
            "core takes {} inputs but {} were given",
            m.inputs.len(),
            inputs.len()
        ));
    }
    for (i, (given, spec)) in inputs.iter().zip(&m.inputs).enumerate() {
        if given.len() != spec.len() {
            return Err(format!(
                "input {i} ({}) has {} elements but the core expects {} ({:?})",
                spec.name,
                given.len(),
                spec.len(),
                spec.dims
            ));
        }
    }

    let s = gpu();
    let t = Instant::now();
    let buffers: Vec<xla_pjrt::Buffer> = inputs
        .iter()
        .zip(&m.inputs)
        .map(|(values, spec)| {
            let dims: Vec<i64> = spec.dims.iter().map(|&n| n as i64).collect();
            unsafe { s.session.input_buffer(wire::as_bytes(values), &dims, KOALABEAR_MONT) }
        })
        .collect();
    let h2d = t.elapsed();

    let refs: Vec<&xla_pjrt::Buffer> = buffers.iter().collect();
    let (outs, dispatch, readback) =
        unsafe { s.session.run_buffers_timed(&core.exe, &refs, m.outputs.len()) };
    Ok((
        outs,
        Phases {
            h2d,
            dispatch,
            readback,
            assemble: Duration::default(),
        },
    ))
}

/// Run `core` on one instance, returning its raw output buffers in manifest
/// order.
pub fn run(
    core: &Core,
    trace: &[Val],
    public_values: &[Val],
) -> Result<(Vec<Vec<u8>>, Phases), String> {
    let m = &core.manifest;
    let expected_trace = m.inputs[0].len();
    if trace.len() != expected_trace {
        return Err(format!(
            "trace has {} elements but the core expects {expected_trace} ({:?})",
            trace.len(),
            m.inputs[0].dims
        ));
    }
    let expected_pv = m.inputs[1].len();
    if public_values.len() != expected_pv {
        return Err(format!(
            "{} public values but the core expects {expected_pv}",
            public_values.len()
        ));
    }

    let s = gpu();
    let dims = |d: &[usize]| d.iter().map(|&n| n as i64).collect::<Vec<_>>();

    let t = Instant::now();
    let trace_buf = unsafe {
        s.session
            .input_buffer(wire::as_bytes(trace), &dims(&m.inputs[0].dims), KOALABEAR_MONT)
    };
    let pv_buf = unsafe {
        s.session.input_buffer(
            wire::as_bytes(public_values),
            &dims(&m.inputs[1].dims),
            KOALABEAR_MONT,
        )
    };
    let h2d = t.elapsed();

    let (outs, dispatch, readback) = unsafe {
        s.session
            .run_buffers_timed(&core.exe, &[&trace_buf, &pv_buf], m.outputs.len())
    };
    Ok((
        outs,
        Phases {
            h2d,
            dispatch,
            readback,
            // Filled in by the caller, which owns the reassembly.
            assemble: Duration::default(),
        },
    ))
}
