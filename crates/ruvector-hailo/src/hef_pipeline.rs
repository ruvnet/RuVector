//! HEF inference pipeline — push embeddings through the Hailo-8 NPU.
//!
//! ADR-176 P1 (`hailo-backend`, iter 158-159). Reads a compiled
//! `model.hef` (produced by `deploy/compile-encoder-hef.py`),
//! configures it on an existing `HailoDevice` vdevice, opens
//! input + output vstreams, and exposes a `forward()` that takes
//! FP32 `[1, seq, hidden]` embeddings and returns FP32
//! `[1, seq, hidden]` post-encoder hidden states.
//!
//! The HEF compiled in iter 156b has these vstream shapes:
//!
//! ```text
//!   Input  minilm_encoder/input_layer1   UINT8, FCR(1x128x384)
//!   Output minilm_encoder/normalization12 UINT8, FCR(1x128x384)
//! ```
//!
//! Quantization scale + zero-point come from `hailo_vstream_info_t`
//! at HEF-load time. We dequantize on read so callers see FP32.
//!
//! **Phase boundary**: this module owns NPU forward pass only. The
//! tokenize → host-side embedding lookup → NPU forward → mean-pool →
//! L2-normalize chain lives in `HefEmbedder` (P3, iter 161).

#![cfg(feature = "hailo")]
// Iter 158 scaffold: several fields are populated by iter-159's
// open_inner + forward bodies. Dead-code warnings would mask
// real-progress signals during the EPIC roll-out.
#![allow(dead_code)]

use crate::device::HailoDevice;
use crate::error::HailoError;
use std::path::Path;
use std::ptr;

/// Quantization parameters for an INT8/UINT8 vstream tensor.
#[derive(Clone, Copy, Debug)]
pub struct QuantInfo {
    /// `dequantized = scale * (raw - zero_point)`.
    pub scale: f32,
    pub zero_point: f32,
}

/// HEF-driven NPU forward pass for the all-MiniLM-L6-v2 encoder.
///
/// Held by `HailoEmbedder` when `model.hef` exists in the model dir.
/// Single-input, single-output: input is the post-embedding hidden
/// states (host-computed via candle's `BertEmbeddings`); output is
/// `last_hidden_state` (the encoder's final LayerNorm output).
///
/// **Lifetime contract**: the underlying HailoRT handles
/// (`hailo_hef`, configured network group, vstreams) are released in
/// the `Drop` impl in this order: vstreams → network group → HEF.
/// Reverse-order release is what the C API expects.
pub struct HefPipeline {
    /// Loaded HEF artifact. Owned; released on drop.
    hef: hailort_sys::hailo_hef,
    /// Configured network group bound to a vdevice. The vdevice itself
    /// is owned by `HailoDevice` higher up the call stack, not here.
    network_group: hailort_sys::hailo_configured_network_group,
    /// Single input vstream (`hidden_states`). UINT8 over the wire.
    input_vstream: hailort_sys::hailo_input_vstream,
    /// Single output vstream (`last_hidden_state`). UINT8 over the wire.
    output_vstream: hailort_sys::hailo_output_vstream,

    /// Quant for the input — host computes float embeddings then we
    /// quantize before `vstream_write`.
    input_quant: QuantInfo,
    /// Quant for the output — NPU returns UINT8 then we dequantize
    /// back to FP32 for the host-side mean-pool.
    output_quant: QuantInfo,

    /// Logical input shape `[batch, seq, hidden]`. Iter 156b: `[1, 128, 384]`.
    input_shape: [usize; 3],
    /// Logical output shape `[batch, seq, hidden]`. Iter 156b: `[1, 128, 384]`.
    output_shape: [usize; 3],

    /// Raw input buffer size in bytes (UINT8). Cached so `forward()`
    /// doesn't recompute per call.
    input_frame_bytes: usize,
    /// Raw output buffer size in bytes.
    output_frame_bytes: usize,
}

impl HefPipeline {
    /// Open `hef_path` and configure it onto `device`'s vdevice.
    ///
    /// The HEF must contain exactly one network group with exactly
    /// one input and one output vstream — the iter-156b compile
    /// produces this shape. Multi-network HEFs are out of scope for
    /// this iteration.
    pub fn open(device: &HailoDevice, hef_path: &Path) -> Result<Self, HailoError> {
        let path_c =
            std::ffi::CString::new(hef_path.to_str().ok_or_else(|| HailoError::BadModelDir {
                path: hef_path.display().to_string(),
                what: "non-UTF8 HEF path",
            })?)
            .map_err(|_| HailoError::BadModelDir {
                path: hef_path.display().to_string(),
                what: "HEF path contains nul byte",
            })?;

        // Iter 173 — security defense in depth: verify the HEF magic
        // before handing the bytes to libhailort. The Hailo HEF format
        // starts with `0x01 0x48 0x45 0x46` (`\x01HEF`). Catches:
        //   * accidental file corruption / truncation
        //   * wrong-file mistakes (operator drops a .onnx where .hef
        //     was expected)
        //   * targeted substitution with a non-HEF payload
        // Iter 198 — verification lives in `hef_verify` so the magic-
        // byte + sha256-pin path can be unit-tested without standing
        // up the rest of HailoRT FFI. Behavior unchanged from iters
        // 173/174 (magic + optional sha256 pin via RUVECTOR_HEF_SHA256).
        crate::hef_verify::verify_hef_header_and_pin(
            hef_path,
            std::env::var("RUVECTOR_HEF_SHA256").ok().as_deref(),
        )?;

        // 1. Load HEF from disk.
        let mut hef: hailort_sys::hailo_hef = ptr::null_mut();
        // SAFETY: path is valid CString; HailoRT writes through `&mut hef`.
        let status =
            unsafe { hailort_sys::hailo_create_hef_file(&mut hef as *mut _, path_c.as_ptr()) };
        if status != 0 {
            return Err(HailoError::Hailort {
                status: status as i32,
                where_: "hailo_create_hef_file",
            });
        }

        // From here on we own `hef`; release it on any error path
        // before propagating.
        Self::open_inner(device, hef, hef_path).inspect_err(|_| {
            // SAFETY: `hef` was returned by hailo_create_hef_file
            // and hasn't been transferred elsewhere yet.
            unsafe {
                hailort_sys::hailo_release_hef(hef);
            }
        })
    }

    fn open_inner(
        device: &HailoDevice,
        hef: hailort_sys::hailo_hef,
        _hef_path: &Path,
    ) -> Result<Self, HailoError> {
        let vdevice = device.raw_vdevice();

        // 1. Init default configure params for this HEF + vdevice.
        //
        // SAFETY (zeroed): `hailo_configure_params_t` is a HailoRT POD
        // C struct (verified iter-178 against /usr/include/hailo/hailort.h
        // — only contains `hailo_configure_network_group_params_t[8]`
        // arrays and primitive ints). All-zero bits are a valid
        // initial state; the SDK overwrites fields via the pointer
        // in the next call.
        let mut params: hailort_sys::hailo_configure_params_t = unsafe { std::mem::zeroed() };
        // SAFETY (FFI call): `hef` is the valid handle returned by
        // `hailo_create_hef_file` above and not yet released.
        // `vdevice` came from `HailoDevice::raw_vdevice()` which is
        // owned by `HailoDevice` whose lifetime outlives `self` via
        // the iter-137 lib.rs Mutex. `&mut params` points at the
        // freshly-zeroed struct on this stack frame which lives until
        // the call returns.
        let status = unsafe {
            hailort_sys::hailo_init_configure_params_by_vdevice(hef, vdevice, &mut params as *mut _)
        };
        if status != 0 {
            return Err(HailoError::Hailort {
                status: status as i32,
                where_: "hailo_init_configure_params_by_vdevice",
            });
        }

        // 2. Configure the vdevice with this HEF. Iter-156b's HEF
        // contains exactly one network group; n_ng >1 would mean a
        // different HEF and we surface the mismatch as an error.
        let mut n_ng: usize = 1;
        let mut network_group: hailort_sys::hailo_configured_network_group = ptr::null_mut();
        // SAFETY (FFI call): `vdevice` and `hef` valid as above.
        // `&mut params` was just initialized by the previous SDK call.
        // `&mut network_group` is a single-element out-buffer (n_ng=1
        // before, the SDK writes `n_ng` to actual count and one
        // network-group handle into the slot). HailoRT documents that
        // failing to open writes 0 to n_ng so the post-check at
        // n_ng != 1 catches both the >1 multi-group case and the 0
        // case.
        let status = unsafe {
            hailort_sys::hailo_configure_vdevice(
                vdevice,
                hef,
                &mut params as *mut _,
                &mut network_group as *mut _,
                &mut n_ng as *mut _,
            )
        };
        if status != 0 {
            return Err(HailoError::Hailort {
                status: status as i32,
                where_: "hailo_configure_vdevice",
            });
        }
        if n_ng != 1 {
            return Err(HailoError::Hailort {
                status: -1,
                where_: "hailo_configure_vdevice — expected 1 network group",
            });
        }

        // 3. Build input vstream params, format=FLOAT32 so HailoRT
        // does the quantize for us. iter-156b HEF has one input.
        let mut input_count: usize = 1;
        // SAFETY (zeroed): `hailo_input_vstream_params_by_name_t` is a
        // POD C struct holding a name (fixed-size char array) plus
        // `hailo_vstream_params_t`. Zero-init is a valid starting
        // state; SDK overwrites all fields when populating.
        let mut input_params: hailort_sys::hailo_input_vstream_params_by_name_t =
            unsafe { std::mem::zeroed() };
        // SAFETY (FFI call): `network_group` is the just-configured
        // handle from above. The `unused` bool param is `false` per
        // HailoRT 4.23 (formerly toggled scale-by-feature; now ignored).
        // `&mut input_params` is a single-element out-buffer; we set
        // input_count=1 beforehand so the SDK writes one params struct.
        let status = unsafe {
            hailort_sys::hailo_make_input_vstream_params(
                network_group,
                false,
                hailort_sys::hailo_format_type_t_HAILO_FORMAT_TYPE_FLOAT32,
                &mut input_params as *mut _,
                &mut input_count as *mut _,
            )
        };
        if status != 0 {
            return Err(HailoError::Hailort {
                status: status as i32,
                where_: "hailo_make_input_vstream_params",
            });
        }
        if input_count != 1 {
            return Err(HailoError::Hailort {
                status: -1,
                where_: "expected 1 input vstream",
            });
        }

        // Iter 191 — override per-call FFI timeout. HailoRT's default
        // `hailo_vstream_params_t.timeout_ms` is 10 s, which is ~700×
        // a steady-state embed (14 ms NPU compute on iter-156b HEF).
        // If the NPU hangs (driver wedge, PCIe link issue), the Mutex
        // in HefEmbedder::Inner blocks for the full 10 s before any
        // caller sees an error, well beyond our iter-182 30 s tonic
        // bound. Cap at 2 s by default (~143× the steady-state cost,
        // still room for tail latency under thermal throttling) so
        // `HAILO_TIMEOUT` (status 4) surfaces fast and the worker
        // can release the Mutex for the next request.
        // Operators tune via `RUVECTOR_NPU_VSTREAM_TIMEOUT_MS`,
        // floor 100 ms so a misconfig can't fail every embed.
        let vstream_timeout_ms: u32 = std::env::var("RUVECTOR_NPU_VSTREAM_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(2_000)
            .max(100);
        input_params.params.timeout_ms = vstream_timeout_ms;

        // 4. Create the input vstream from the params.
        let mut input_vstream: hailort_sys::hailo_input_vstream = ptr::null_mut();
        // SAFETY (FFI call): network_group + input_params valid from
        // the previous calls. Count `1` matches the iter-156b HEF's
        // single input; passing a wrong count would write past the
        // single-slot &mut input_vstream out-buffer, which the
        // input_count=1 check after step 3 prevents.
        let status = unsafe {
            hailort_sys::hailo_create_input_vstreams(
                network_group,
                &input_params as *const _,
                1,
                &mut input_vstream as *mut _,
            )
        };
        if status != 0 {
            return Err(HailoError::Hailort {
                status: status as i32,
                where_: "hailo_create_input_vstreams",
            });
        }

        // 5. Same for output vstream.
        let mut output_count: usize = 1;
        // SAFETY (zeroed): mirror of the input_params POD invariant.
        let mut output_params: hailort_sys::hailo_output_vstream_params_by_name_t =
            unsafe { std::mem::zeroed() };
        // SAFETY (FFI call): mirror of make_input_vstream_params; same
        // single-output assumption holds (iter-156b HEF emits one
        // last_hidden_state tensor).
        let status = unsafe {
            hailort_sys::hailo_make_output_vstream_params(
                network_group,
                false,
                hailort_sys::hailo_format_type_t_HAILO_FORMAT_TYPE_FLOAT32,
                &mut output_params as *mut _,
                &mut output_count as *mut _,
            )
        };
        if status != 0 {
            return Err(HailoError::Hailort {
                status: status as i32,
                where_: "hailo_make_output_vstream_params",
            });
        }
        // Iter 191 — same vstream timeout cap on the output side.
        // The read path (`hailo_vstream_read_raw_buffer`) blocks until
        // the DMA completes; matching the input timeout keeps the
        // forward-pass bound symmetric.
        output_params.params.timeout_ms = vstream_timeout_ms;

        let mut output_vstream: hailort_sys::hailo_output_vstream = ptr::null_mut();
        // SAFETY (FFI call): mirror of create_input_vstreams.
        let status = unsafe {
            hailort_sys::hailo_create_output_vstreams(
                network_group,
                &output_params as *const _,
                1,
                &mut output_vstream as *mut _,
            )
        };
        if status != 0 {
            return Err(HailoError::Hailort {
                status: status as i32,
                where_: "hailo_create_output_vstreams",
            });
        }

        // 6. Read vstream metadata for shape + quant. We use FLOAT32
        // format so HailoRT does quant for us; we keep the quant info
        // for diagnostics only.
        // SAFETY (zeroed): hailo_vstream_info_t is a POD struct
        // containing primitives + a `format_t` + a tagged union of
        // `shape: hailo_3d_image_shape_t` xor `nms_shape: hailo_nms_shape_t`.
        // Zero-init is valid; SDK fills both the discriminant
        // (`format.order`) and the union body.
        let mut input_info: hailort_sys::hailo_vstream_info_t = unsafe { std::mem::zeroed() };
        // SAFETY (FFI call): input_vstream returned by
        // hailo_create_input_vstreams above and not yet released.
        let status = unsafe {
            hailort_sys::hailo_get_input_vstream_info(input_vstream, &mut input_info as *mut _)
        };
        if status != 0 {
            return Err(HailoError::Hailort {
                status: status as i32,
                where_: "hailo_get_input_vstream_info",
            });
        }
        // SAFETY (zeroed/FFI): same invariants as input.
        let mut output_info: hailort_sys::hailo_vstream_info_t = unsafe { std::mem::zeroed() };
        let status = unsafe {
            hailort_sys::hailo_get_output_vstream_info(output_vstream, &mut output_info as *mut _)
        };
        if status != 0 {
            return Err(HailoError::Hailort {
                status: status as i32,
                where_: "hailo_get_output_vstream_info",
            });
        }

        // SAFETY (union access): hailo_vstream_info_t holds a tagged
        // union — `shape: hailo_3d_image_shape_t` for non-NMS layouts
        // (everything our encoder produces) xor `nms_shape:
        // hailo_nms_shape_t` for NMS post-process layouts. Discriminant
        // lives in `format.order`. Iter-156b's HEF compiles a transformer
        // encoder with no NMS — the parse log confirmed
        // `End nodes mapped: '/encoder/layer.5/output/LayerNorm/Add_1'`
        // which is a plain rank-3 tensor. If a future HEF added NMS we'd
        // need to gate this read on `format.order != HAILO_FORMAT_ORDER_HAILO_NMS`
        // before reading the union; for the iter-156b HEF this is
        // unconditionally `shape`.
        let in_shape = unsafe { input_info.__bindgen_anon_1.shape };
        let out_shape = unsafe { output_info.__bindgen_anon_1.shape };

        // Logical [batch=1, seq=128, hidden=384] maps to
        // (height=1, width=128, features=384) for our HEF. Buffer is
        // row-major over h×w×f. We use max(height, width) since the
        // mapping isn't strict — Hailo can route either axis to the
        // longer one based on its placement decisions.
        let input_shape = [
            1usize,
            in_shape.height.max(in_shape.width) as usize,
            in_shape.features as usize,
        ];
        let output_shape = [
            1usize,
            out_shape.height.max(out_shape.width) as usize,
            out_shape.features as usize,
        ];

        // FP32 frame size = sum of dims * 4 bytes. The vstream API
        // also exposes `hailo_get_input_vstream_frame_size` if we
        // want HailoRT to compute it; using the shape is equivalent
        // and avoids one more FFI hop.
        let input_frame_bytes = input_shape[0] * input_shape[1] * input_shape[2] * 4;
        let output_frame_bytes = output_shape[0] * output_shape[1] * output_shape[2] * 4;

        let input_quant = QuantInfo {
            scale: input_info.quant_info.qp_scale as f32,
            zero_point: input_info.quant_info.qp_zp as f32,
        };
        let output_quant = QuantInfo {
            scale: output_info.quant_info.qp_scale as f32,
            zero_point: output_info.quant_info.qp_zp as f32,
        };

        Ok(Self {
            hef,
            network_group,
            input_vstream,
            output_vstream,
            input_quant,
            output_quant,
            input_shape,
            output_shape,
            input_frame_bytes,
            output_frame_bytes,
        })
    }

    /// FP32 forward pass. Takes a flat `[batch * seq * hidden]` input
    /// in row-major order, returns the same shape post-encoder.
    ///
    /// HailoRT does the FP32 → INT8 quantize on write and INT8 → FP32
    /// dequantize on read because we configured both vstreams with
    /// `HAILO_FORMAT_TYPE_FLOAT32`. We pass FP32 bytes in, get FP32
    /// bytes out.
    ///
    /// Convenience wrapper around `forward_into` that allocates the
    /// output Vec each call. For hot paths use `forward_into` and
    /// reuse a buffer (iter 175).
    pub fn forward(&mut self, input: &[f32]) -> Result<Vec<f32>, HailoError> {
        let mut out = vec![0.0f32; self.output_frame_bytes / 4];
        self.forward_into(input, &mut out)?;
        Ok(out)
    }

    /// FP32 forward pass writing into a caller-provided buffer.
    /// Iter 175 — buffer pooling: lets `HefEmbedder` reuse a single
    /// `last_hidden` Vec across calls so the NPU output (~196 KB for
    /// the iter-156b 1×128×384 shape) doesn't churn the allocator at
    /// 67 embeds/sec.
    ///
    /// `output` is resized to `output_frame_bytes / 4` if shorter.
    pub fn forward_into(&mut self, input: &[f32], output: &mut Vec<f32>) -> Result<(), HailoError> {
        let expected_floats = self.input_frame_bytes / 4;
        if input.len() != expected_floats {
            return Err(HailoError::Shape {
                expected: expected_floats,
                actual: input.len(),
            });
        }
        let out_floats = self.output_frame_bytes / 4;
        if output.len() < out_floats {
            output.resize(out_floats, 0.0);
        }

        // Push the FP32 input. HailoRT internally quantizes to UINT8
        // using the embedded scale + zero-point from the HEF.
        //
        // SAFETY (input write):
        //   * `self.input_vstream` is a non-null handle from
        //     `hailo_create_input_vstreams` (`open_inner`); not yet
        //     released because Drop runs after the last `&mut self`
        //     borrow ends.
        //   * `input.as_ptr() as *const c_void` points at
        //     `input.len() * 4` immutable, properly-aligned bytes
        //     (Vec<f32> on x86/aarch64 is 4-byte aligned).
        //   * `self.input_frame_bytes` was computed from the
        //     `input_shape` Hailo reported in `open_inner` and the
        //     bounds-check above (`input.len() == input_frame_bytes/4`)
        //     guarantees we don't ask HailoRT to read past the buffer.
        //   * `&mut self` serializes concurrent calls; no other thread
        //     can mutate or drop `self.input_vstream` while this runs.
        let status = unsafe {
            hailort_sys::hailo_vstream_write_raw_buffer(
                self.input_vstream,
                input.as_ptr() as *const std::ffi::c_void,
                self.input_frame_bytes,
            )
        };
        if status != 0 {
            return Err(HailoError::Hailort {
                status: status as i32,
                where_: "hailo_vstream_write_raw_buffer",
            });
        }

        // Pull the FP32 output. HailoRT dequantizes for us.
        //
        // SAFETY (output read):
        //   * `self.output_vstream` is a non-null handle from
        //     `hailo_create_output_vstreams` (`open_inner`).
        //   * `output.as_mut_ptr() as *mut c_void` points at
        //     `output.len() * 4` writable, properly-aligned bytes.
        //     The `output.resize(out_floats, 0.0)` above ensures
        //     `output.len() >= out_floats`, so HailoRT writing exactly
        //     `output_frame_bytes` cannot overrun.
        //   * `&mut self` again serializes; no other writer for the
        //     output buffer.
        let status = unsafe {
            hailort_sys::hailo_vstream_read_raw_buffer(
                self.output_vstream,
                output.as_mut_ptr() as *mut std::ffi::c_void,
                self.output_frame_bytes,
            )
        };
        if status != 0 {
            return Err(HailoError::Hailort {
                status: status as i32,
                where_: "hailo_vstream_read_raw_buffer",
            });
        }

        Ok(())
    }

    pub fn input_shape(&self) -> [usize; 3] {
        self.input_shape
    }

    pub fn output_shape(&self) -> [usize; 3] {
        self.output_shape
    }

    pub fn input_quant(&self) -> QuantInfo {
        self.input_quant
    }

    pub fn output_quant(&self) -> QuantInfo {
        self.output_quant
    }
}

impl Drop for HefPipeline {
    fn drop(&mut self) {
        // SAFETY: each handle was returned by HailoRT and hasn't been
        // released yet. Release order is reverse of acquisition:
        // vstreams first (they hold refs into the network group), then
        // the HEF (the configured network group is owned by the
        // vdevice and released when the vdevice is — HailoRT C API
        // doesn't expose a separate release for it).
        unsafe {
            if !self.input_vstream.is_null() {
                hailort_sys::hailo_release_input_vstreams(&mut self.input_vstream as *mut _, 1);
            }
            if !self.output_vstream.is_null() {
                hailort_sys::hailo_release_output_vstreams(&mut self.output_vstream as *mut _, 1);
            }
            if !self.hef.is_null() {
                hailort_sys::hailo_release_hef(self.hef);
            }
        }
    }
}

// SAFETY: HailoRT documents handles as thread-safe for inference
// when external serialisation prevents config changes during traffic.
// `HefPipeline` is held behind `Mutex` in `HailoEmbedder`.
unsafe impl Send for HefPipeline {}
unsafe impl Sync for HefPipeline {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_returns_not_yet_implemented_until_iter_159() {
        // Open HailoDevice would fail without a real /dev/hailo0
        // present, so we can't even reach HefPipeline::open here on
        // a dev box. The test exists to assert the public type
        // signatures compile.
        let _ = std::mem::size_of::<HefPipeline>();
        let _ = std::mem::size_of::<QuantInfo>();
    }
}
