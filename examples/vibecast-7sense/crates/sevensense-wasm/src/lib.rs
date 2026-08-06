//! Browser bindings for the 7sense analysis pipeline.
//!
//! This exposes a raw C ABI rather than using `wasm-bindgen`. The compiled
//! module is inlined into a self-contained page as a data URI, so binary size
//! is a correctness constraint -- `wasm-bindgen` would add a JS shim and
//! several hundred kilobytes of glue for an interface that is, in the end, a
//! handful of float buffers.
//!
//! # Calling convention
//!
//! The host allocates an input buffer with [`alloc_f32`], writes samples into
//! WebAssembly memory, and calls an analysis function. Each returns a count (or
//! a negative error code) and leaves its packed result in a module-owned output
//! buffer, readable via [`out_ptr`] and [`out_len`].
//!
//! Results are packed as flat `f32` arrays with a fixed stride per record, so
//! the host reads them with a single `Float32Array` view and no per-field
//! decoding. Field order is declared by the `*_STRIDE` constants below and must
//! stay in step with the JavaScript that reads it.

#![allow(clippy::missing_safety_doc)]

use sevensense_audio::features::{FeatureConfig, FeatureExtractor};
use sevensense_audio::spectrogram::{MelSpectrogram, SpectrogramConfig};
use sevensense_audio::streaming::{
    MemorySource, StreamConfig, StreamPipeline, StreamSegmenter, StreamSegmenterConfig,
};
use sevensense_vector::projection::{to_poincare_ball, PcaProjection};

/// Values per frame in the [`analyze_features`] result.
pub const FRAME_STRIDE: usize = 10;

/// Values per segment in the [`detect_segments`] result.
pub const SEGMENT_STRIDE: usize = 6;

/// Values per neighbour in the [`knn_search`] result.
pub const NEIGHBOUR_STRIDE: usize = 2;

/// Cosine distance beyond which [`knn_search`] reports no confident match.
///
/// ADR-013 sets this: returning a label for a sound unlike anything indexed is
/// the failure most likely to produce a false record, so it gets an explicit
/// branch rather than a quiet nearest-neighbour answer.
pub const UNKNOWN_THRESHOLD: f32 = 0.55;

/// Input was empty, too short, or otherwise unusable.
const ERR_INVALID_INPUT: i32 = -1;
/// Analysis failed for a reason reported by the underlying crate.
const ERR_ANALYSIS_FAILED: i32 = -2;

/// Module-owned buffer holding the most recent result.
///
/// Single-threaded by construction: WebAssembly without the threads proposal
/// runs one instance per worker, and each call fully overwrites this before
/// returning.
static mut OUTPUT: Vec<f32> = Vec::new();

/// Replaces the output buffer and returns how many records it holds.
fn publish(values: Vec<f32>, stride: usize) -> i32 {
    let records = if stride == 0 { 0 } else { values.len() / stride };
    // Safety: single-threaded module; no other reference is live here.
    unsafe {
        OUTPUT = values;
    }
    i32::try_from(records).unwrap_or(i32::MAX)
}

/// Allocates `len` floats in module memory and returns a pointer to them.
///
/// The host owns the result and must release it with [`dealloc_f32`].
#[no_mangle]
pub extern "C" fn alloc_f32(len: usize) -> *mut f32 {
    let mut buffer = Vec::<f32>::with_capacity(len);
    let ptr = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    ptr
}

/// Releases a buffer previously returned by [`alloc_f32`].
///
/// # Safety
/// `ptr` must come from [`alloc_f32`] with the same `len`, and must not have
/// been freed already.
#[no_mangle]
pub unsafe extern "C" fn dealloc_f32(ptr: *mut f32, len: usize) {
    if !ptr.is_null() {
        drop(Vec::from_raw_parts(ptr, 0, len));
    }
}

/// Pointer to the packed result of the most recent analysis call.
#[no_mangle]
pub extern "C" fn out_ptr() -> *const f32 {
    // Safety: returns a shared pointer; the host reads before the next call.
    unsafe { core::ptr::addr_of!(OUTPUT).cast::<Vec<f32>>().as_ref() }
        .map_or(core::ptr::null(), |v| v.as_ptr())
}

/// Number of floats in the most recent result.
#[no_mangle]
pub extern "C" fn out_len() -> usize {
    unsafe { core::ptr::addr_of!(OUTPUT).cast::<Vec<f32>>().as_ref() }.map_or(0, Vec::len)
}

/// Computes per-frame acoustic descriptors over `samples`.
///
/// Packs [`FRAME_STRIDE`] values per frame: time in milliseconds, spectral
/// centroid, spread, rolloff, tonality, crest, entropy, dominant frequency,
/// energy in dBFS, and a voiced flag.
///
/// Returns the frame count, or a negative error code.
///
/// # Safety
/// `ptr` must point to `len` readable floats.
#[no_mangle]
pub unsafe extern "C" fn analyze_features(ptr: *const f32, len: usize, sample_rate: u32) -> i32 {
    if ptr.is_null() || len == 0 || sample_rate == 0 {
        return ERR_INVALID_INPUT;
    }
    let samples = std::slice::from_raw_parts(ptr, len);

    let config = FeatureConfig {
        sample_rate,
        ..Default::default()
    };
    let Ok(extractor) = FeatureExtractor::new(config) else {
        return ERR_INVALID_INPUT;
    };
    let Ok(features) = extractor.extract(samples) else {
        return ERR_ANALYSIS_FAILED;
    };

    let mut packed = Vec::with_capacity(features.frames.len() * FRAME_STRIDE);
    for frame in &features.frames {
        packed.push(frame.time_ms as f32);
        packed.push(frame.centroid_hz);
        packed.push(frame.spread_hz);
        packed.push(frame.rolloff_hz);
        packed.push(frame.tonality);
        packed.push(frame.crest);
        packed.push(frame.entropy);
        packed.push(frame.dominant_hz);
        // A silent frame yields -inf dB, which serializes to NaN in JS and
        // breaks any chart drawn from it. Clamp to the display floor instead.
        packed.push(if frame.energy_db.is_finite() {
            frame.energy_db
        } else {
            -120.0
        });
        packed.push(if frame.voiced { 1.0 } else { 0.0 });
    }

    publish(packed, FRAME_STRIDE)
}

/// Returns the summary statistics for the last [`analyze_features`] call's
/// input, recomputed over `samples`.
///
/// Packs 12 values: mean centroid, centroid deviation, mean spread, mean
/// tonality, mean crest, mean entropy, mean dominant frequency, mean energy,
/// voiced fraction, AM rate, AM depth, and FM extent. Modulation fields are
/// zero when the input is too short to measure them.
///
/// # Safety
/// `ptr` must point to `len` readable floats.
#[no_mangle]
pub unsafe extern "C" fn analyze_summary(ptr: *const f32, len: usize, sample_rate: u32) -> i32 {
    if ptr.is_null() || len == 0 || sample_rate == 0 {
        return ERR_INVALID_INPUT;
    }
    let samples = std::slice::from_raw_parts(ptr, len);

    let config = FeatureConfig {
        sample_rate,
        ..Default::default()
    };
    let Ok(extractor) = FeatureExtractor::new(config) else {
        return ERR_INVALID_INPUT;
    };
    let Ok(features) = extractor.extract(samples) else {
        return ERR_ANALYSIS_FAILED;
    };

    let s = &features.summary;
    let modulation = s.modulation;
    let packed = vec![
        s.centroid_hz.mean,
        s.centroid_hz.std,
        s.spread_hz.mean,
        s.tonality.mean,
        s.crest.mean,
        s.entropy.mean,
        s.dominant_hz.mean,
        if s.energy_db.mean.is_finite() {
            s.energy_db.mean
        } else {
            -120.0
        },
        s.voiced_fraction,
        modulation.map_or(0.0, |m| m.am_rate_hz),
        modulation.map_or(0.0, |m| m.am_depth),
        modulation.map_or(0.0, |m| m.fm_extent_hz),
    ];

    publish(packed, 12)
}

/// Segments `samples` with the streaming pipeline.
///
/// Packs [`SEGMENT_STRIDE`] values per segment: start and end time in
/// milliseconds, peak amplitude, RMS energy, SNR in dB, and a truncated flag.
///
/// Returns the segment count, or a negative error code.
///
/// # Safety
/// `ptr` must point to `len` readable floats.
#[no_mangle]
pub unsafe extern "C" fn detect_segments(ptr: *const f32, len: usize, sample_rate: u32) -> i32 {
    if ptr.is_null() || len == 0 || sample_rate == 0 {
        return ERR_INVALID_INPUT;
    }
    let samples = std::slice::from_raw_parts(ptr, len).to_vec();

    let config = StreamSegmenterConfig {
        sample_rate,
        ..Default::default()
    };
    let Ok(mut segmenter) = StreamSegmenter::new(config) else {
        return ERR_INVALID_INPUT;
    };

    let mut events = segmenter.push(&samples);
    events.extend(segmenter.flush());

    let to_ms = |s: u64| s as f32 * 1000.0 / sample_rate as f32;
    let mut packed = Vec::new();
    for event in events {
        if let sevensense_audio::streaming::SegmentEvent::Closed(segment) = event {
            packed.push(to_ms(segment.start_sample));
            packed.push(to_ms(segment.end_sample));
            packed.push(segment.peak_amplitude);
            packed.push(segment.rms_energy);
            packed.push(segment.snr_db());
            packed.push(if segment.truncated { 1.0 } else { 0.0 });
        }
    }

    publish(packed, SEGMENT_STRIDE)
}

/// Runs the full streaming pipeline and reports how many analysis windows a
/// recording would produce, alongside the loss counters.
///
/// Packs 5 values: window count, segment count, discarded count, samples read,
/// and samples dropped to ring-buffer overwrite.
///
/// # Safety
/// `ptr` must point to `len` readable floats.
#[no_mangle]
pub unsafe extern "C" fn pipeline_stats(ptr: *const f32, len: usize, sample_rate: u32) -> i32 {
    if ptr.is_null() || len == 0 || sample_rate == 0 {
        return ERR_INVALID_INPUT;
    }
    let samples = std::slice::from_raw_parts(ptr, len).to_vec();

    let source = MemorySource::mono(samples, sample_rate);
    let Ok(mut pipeline) = StreamPipeline::new(source, StreamConfig::default()) else {
        return ERR_INVALID_INPUT;
    };
    let Ok(windows) = pipeline.run_to_completion() else {
        return ERR_ANALYSIS_FAILED;
    };

    let stats = pipeline.stats();
    publish(
        vec![
            windows.len() as f32,
            stats.segments as f32,
            stats.discarded as f32,
            stats.samples_read as f32,
            stats.samples_dropped as f32,
        ],
        5,
    )
}

/// Fits a PCA projection over `n_points` vectors of `dim` values each and
/// projects them to `out_dims` dimensions.
///
/// Input is row-major: point `i` occupies `[i * dim, (i + 1) * dim)`.
/// Packs `out_dims` coordinates per point, followed by one final record holding
/// the explained-variance ratio per component.
///
/// Returns the number of records (points plus the trailing variance record), or
/// a negative error code.
///
/// # Safety
/// `ptr` must point to `n_points * dim` readable floats.
#[no_mangle]
pub unsafe extern "C" fn project_pca(
    ptr: *const f32,
    n_points: usize,
    dim: usize,
    out_dims: usize,
    seed: u32,
) -> i32 {
    if ptr.is_null() || n_points == 0 || dim == 0 || out_dims == 0 || out_dims > dim {
        return ERR_INVALID_INPUT;
    }
    let flat = std::slice::from_raw_parts(ptr, n_points * dim);
    let data: Vec<Vec<f32>> = flat.chunks_exact(dim).map(<[f32]>::to_vec).collect();

    let Ok(pca) = PcaProjection::fit(&data, out_dims, u64::from(seed)) else {
        return ERR_INVALID_INPUT;
    };
    let Ok(mut projected) = pca.project_batch(&data) else {
        return ERR_ANALYSIS_FAILED;
    };

    // Bounds are discarded: the host frames the view from the normalized
    // coordinates alone, so the original scale is not needed here.
    let _bounds = sevensense_vector::projection::normalize_coordinates(&mut projected);

    let mut packed: Vec<f32> = projected.into_iter().flatten().collect();
    // Trailing record: explained variance per component, padded to the stride
    // so the host can read everything with one uniformly-strided view.
    let mut variance = pca.explained_variance().to_vec();
    variance.resize(out_dims, 0.0);
    packed.extend(variance);

    publish(packed, out_dims)
}

/// Computes a mel spectrogram and packs it row-major, one row per mel band.
///
/// Returns the number of time frames, or a negative error code. The buffer
/// holds `n_mels * frames` values laid out row-major by mel band, so band `m`
/// at frame `t` is at `m * frames + t`.
///
/// Values are in dB and are min-max normalised to 0-1 across the whole
/// spectrogram, so the host can map them straight onto a colour ramp without
/// needing to know the absolute scale.
///
/// # Safety
/// `ptr` must point to `len` readable floats.
#[no_mangle]
pub unsafe extern "C" fn compute_spectrogram(
    ptr: *const f32,
    len: usize,
    sample_rate: u32,
    n_mels: usize,
    hop_length: usize,
) -> i32 {
    if ptr.is_null() || len == 0 || sample_rate == 0 || n_mels == 0 || hop_length == 0 {
        return ERR_INVALID_INPUT;
    }
    let samples = std::slice::from_raw_parts(ptr, len);

    let config = SpectrogramConfig {
        n_mels,
        hop_length,
        sample_rate,
        f_max: (sample_rate as f32 / 2.0).min(16_000.0),
        ..Default::default()
    };

    let Ok(spectrogram) = MelSpectrogram::compute(samples, config) else {
        return ERR_ANALYSIS_FAILED;
    };

    let data = &spectrogram.data;
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for value in data.iter().filter(|v| v.is_finite()) {
        lo = lo.min(*value);
        hi = hi.max(*value);
    }
    if !lo.is_finite() || !hi.is_finite() {
        return ERR_ANALYSIS_FAILED;
    }
    let span = if hi - lo > f32::EPSILON { hi - lo } else { 1.0 };

    let frames = data.ncols();
    let rows = data.nrows();
    let mut packed = Vec::with_capacity(data.len());
    for row in 0..rows {
        for col in 0..frames {
            let value = data[[row, col]];
            packed.push(if value.is_finite() {
                ((value - lo) / span).clamp(0.0, 1.0)
            } else {
                0.0
            });
        }
    }

    // `publish` returns len / stride. Striding by the row count therefore
    // returns the frame count, which is the dimension the host cannot infer --
    // it already knows how many mel bands it asked for.
    publish(packed, rows.max(1))
}

/// Finds the `k` nearest reference vectors to `query` by cosine distance.
///
/// This is ADR-013's identification path: retrieval against a labelled index
/// rather than classification. Perch emits embeddings, not logits, and
/// retrieval is open-set -- a sound unlike anything indexed comes back as
/// distant neighbours rather than being forced into the nearest class.
///
/// `references` is row-major, `n_refs` vectors of `dim` values. Packs
/// [`NEIGHBOUR_STRIDE`] values per neighbour: the reference's index and its
/// cosine distance, nearest first.
///
/// Returns the neighbour count, or a negative error code.
///
/// # Safety
/// `query` must point to `dim` readable floats and `references` to
/// `n_refs * dim`.
#[no_mangle]
pub unsafe extern "C" fn knn_search(
    query: *const f32,
    references: *const f32,
    n_refs: usize,
    dim: usize,
    k: usize,
) -> i32 {
    if query.is_null() || references.is_null() || n_refs == 0 || dim == 0 || k == 0 {
        return ERR_INVALID_INPUT;
    }
    let q = std::slice::from_raw_parts(query, dim);
    let refs = std::slice::from_raw_parts(references, n_refs * dim);

    let mut scored: Vec<(usize, f32)> = refs
        .chunks_exact(dim)
        .enumerate()
        .map(|(i, candidate)| (i, sevensense_vector::cosine_distance(q, candidate)))
        .collect();

    // Partial ordering: cosine distance is finite here, but sort_by with a
    // total_cmp keeps this well-defined even if a degenerate vector slips in.
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));

    let mut packed = Vec::with_capacity(k.min(scored.len()) * NEIGHBOUR_STRIDE);
    for (index, distance) in scored.into_iter().take(k) {
        packed.push(index as f32);
        packed.push(distance);
    }

    publish(packed, NEIGHBOUR_STRIDE)
}

/// Reports acoustic quality gates for `samples`.
///
/// Packs 4 values: a likely-noise flag, a likely-clipped flag, the voiced
/// fraction, and mean tonality. ADR-013 rejects segments failing these before
/// spending inference budget on them -- wind and rain are the common case.
///
/// # Safety
/// `ptr` must point to `len` readable floats.
#[no_mangle]
pub unsafe extern "C" fn quality_flags(ptr: *const f32, len: usize, sample_rate: u32) -> i32 {
    if ptr.is_null() || len == 0 || sample_rate == 0 {
        return ERR_INVALID_INPUT;
    }
    let samples = std::slice::from_raw_parts(ptr, len);

    let config = FeatureConfig {
        sample_rate,
        ..Default::default()
    };
    let Ok(extractor) = FeatureExtractor::new(config) else {
        return ERR_INVALID_INPUT;
    };
    let Ok(features) = extractor.extract(samples) else {
        return ERR_ANALYSIS_FAILED;
    };

    publish(
        vec![
            f32::from(u8::from(features.is_likely_noise())),
            f32::from(u8::from(features.is_likely_clipped())),
            features.summary.voiced_fraction,
            features.summary.tonality.mean,
        ],
        4,
    )
}

/// Projects vectors into the Poincare ball via PCA.
///
/// Hierarchical structure reads as radial depth: general calls sit near the
/// origin, specialised variants near the boundary, where hyperbolic space has
/// exponentially more room. `scale` controls how aggressively the radius is
/// compressed.
///
/// Packs the same layout as [`project_pca`]: `out_dims` coordinates per point,
/// then one record of explained variance.
///
/// # Safety
/// `ptr` must point to `n_points * dim` readable floats.
#[no_mangle]
pub unsafe extern "C" fn project_poincare(
    ptr: *const f32,
    n_points: usize,
    dim: usize,
    out_dims: usize,
    seed: u32,
    scale: f32,
) -> i32 {
    if ptr.is_null() || n_points == 0 || dim == 0 || out_dims == 0 || out_dims > dim {
        return ERR_INVALID_INPUT;
    }
    let flat = std::slice::from_raw_parts(ptr, n_points * dim);
    let data: Vec<Vec<f32>> = flat.chunks_exact(dim).map(<[f32]>::to_vec).collect();

    let Ok(pca) = PcaProjection::fit(&data, out_dims, u64::from(seed)) else {
        return ERR_INVALID_INPUT;
    };
    let Ok(mut projected) = pca.project_batch(&data) else {
        return ERR_ANALYSIS_FAILED;
    };

    let _bounds = sevensense_vector::projection::normalize_coordinates(&mut projected);
    let ball = to_poincare_ball(&projected, if scale > 0.0 { scale } else { 1.0 });

    let mut packed: Vec<f32> = ball.into_iter().flatten().collect();
    let mut variance = pca.explained_variance().to_vec();
    variance.resize(out_dims, 0.0);
    packed.extend(variance);

    publish(packed, out_dims)
}

/// Cosine distance between two vectors of `dim` values.
///
/// Exposed so the host can score a single pair without staging a full
/// reference set through [`knn_search`].
///
/// # Safety
/// Both pointers must reference `dim` readable floats.
#[no_mangle]
pub unsafe extern "C" fn cosine_distance_of(a: *const f32, b: *const f32, dim: usize) -> f32 {
    if a.is_null() || b.is_null() || dim == 0 {
        return f32::NAN;
    }
    sevensense_vector::cosine_distance(
        std::slice::from_raw_parts(a, dim),
        std::slice::from_raw_parts(b, dim),
    )
}

/// The open-set rejection threshold, so the host does not hardcode its own.
#[no_mangle]
pub extern "C" fn unknown_threshold() -> f32 {
    UNKNOWN_THRESHOLD
}

/// Version of the analysis pipeline this module was built from.
///
/// Encoded as `major * 10000 + minor * 100 + patch` so it crosses the ABI as a
/// single integer.
#[no_mangle]
pub extern "C" fn pipeline_version() -> i32 {
    let parse = |s: Option<&str>| -> i32 { s.and_then(|v| v.parse().ok()).unwrap_or(0) };
    let mut parts = env!("CARGO_PKG_VERSION").split('.');
    parse(parts.next()) * 10_000 + parse(parts.next()) * 100 + parse(parts.next())
}
