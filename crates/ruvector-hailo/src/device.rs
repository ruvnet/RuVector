//! Safe wrapper around HailoRT's `hailo_vdevice` handle.
//!
//! ADR-167 §5 step 4 (`hailo-backend` branch). Lifetime contract:
//!   - `HailoDevice::open()` calls `hailo_create_vdevice` (defaults).
//!   - `HailoDevice::version()` reads the library version via
//!     `hailo_get_library_version`.
//!   - `Drop` releases the underlying handle via `hailo_release_vdevice`.
//!
//! `Send + Sync` are sound: HailoRT documents the vdevice handle as
//! thread-safe for inference operations performed against the same
//! configured network group. Configuration changes still need external
//! serialisation; we provide that via `Mutex` higher up in `lib.rs`.

#![allow(dead_code)]

use crate::error::HailoError;
#[cfg(feature = "hailo")]
use std::ptr;

/// Opaque safe handle to a HailoRT virtual device.
///
/// `_handle` is meaningful only when the `hailo` feature is on. Without
/// the feature, the type still exists (so the rest of the crate compiles
/// on non-Pi developer machines) but `open()` returns `FeatureDisabled`.
pub struct HailoDevice {
    #[cfg(feature = "hailo")]
    handle: hailort_sys::hailo_vdevice,

    #[cfg(not(feature = "hailo"))]
    _phantom: std::marker::PhantomData<()>,
}

impl HailoDevice {
    /// Raw vdevice handle for the HEF pipeline (iter 159+). Crate-
    /// internal because callers shouldn't reach into the FFI; the
    /// `HefPipeline` and friends use it under their own SAFETY
    /// invariants.
    #[cfg(feature = "hailo")]
    pub(crate) fn raw_vdevice(&self) -> hailort_sys::hailo_vdevice {
        self.handle
    }

    /// Open a virtual Hailo device with default parameters. On a Pi 5 with
    /// the AI HAT+ this enumerates `/dev/hailo0` and brings up firmware.
    pub fn open() -> Result<Self, HailoError> {
        #[cfg(feature = "hailo")]
        {
            // SAFETY: passing NULL params requests defaults — pulls in any
            // available device. The output `vdevice` is written through.
            let mut handle: hailort_sys::hailo_vdevice = ptr::null_mut();
            let status = unsafe {
                hailort_sys::hailo_create_vdevice(ptr::null_mut(), &mut handle as *mut _)
            };
            if status != 0 {
                return Err(HailoError::Hailort {
                    status: status as i32,
                    where_: "hailo_create_vdevice",
                });
            }
            if handle.is_null() {
                return Err(HailoError::NoDevice(
                    "hailo_create_vdevice returned success but null handle".into(),
                ));
            }
            Ok(Self { handle })
        }
        #[cfg(not(feature = "hailo"))]
        {
            Err(HailoError::FeatureDisabled)
        }
    }

    /// Return `(major, minor, revision)` from `hailo_get_library_version`.
    /// Lives on `HailoDevice` rather than free-standing because it implies
    /// the runtime is loaded and reachable through this device's session.
    pub fn version(&self) -> Option<(u32, u32, u32)> {
        hailort_sys::version_triple()
    }

    /// Read the on-die NPU temperature(s) from the Hailo-8 chip.
    /// Returns `(ts0_celsius, ts1_celsius)` — two thermal sensors per
    /// die. Iter 95 deliverable from ADR-174 §93 (NPU sensor read).
    ///
    /// Implementation: walks the vdevice's physical devices via
    /// `hailo_get_physical_devices`, then calls
    /// `hailo_get_chip_temperature` on the first one. Returns `None`
    /// if either call fails or if the feature is disabled.
    ///
    /// **Without the `hailo` feature** this always returns `None` so
    /// the rest of the crate compiles on non-Pi developer machines.
    pub fn chip_temperature(&self) -> Option<(f32, f32)> {
        #[cfg(feature = "hailo")]
        {
            use std::ptr;

            // Step 1: enumerate the physical devices behind this vdevice.
            // Pi 5 + AI HAT+ has exactly one (the Hailo-8 over PCIe).
            let mut count: usize = 8;
            let mut handles: [hailort_sys::hailo_device; 8] = [ptr::null_mut(); 8];
            let status = unsafe {
                hailort_sys::hailo_get_physical_devices(
                    self.handle,
                    handles.as_mut_ptr(),
                    &mut count as *mut _,
                )
            };
            if status != 0 || count == 0 || handles[0].is_null() {
                return None;
            }
            // Step 2: read the temperature info from device 0.
            let mut info = hailort_sys::hailo_chip_temperature_info_t {
                ts0_temperature: 0.0,
                ts1_temperature: 0.0,
                sample_count: 0,
            };
            let status =
                unsafe { hailort_sys::hailo_get_chip_temperature(handles[0], &mut info as *mut _) };
            if status != 0 {
                return None;
            }
            Some((info.ts0_temperature, info.ts1_temperature))
        }
        #[cfg(not(feature = "hailo"))]
        {
            None
        }
    }
}

#[cfg(feature = "hailo")]
impl Drop for HailoDevice {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle is non-null and was produced by
            // `hailo_create_vdevice`; the contract says exactly one
            // matching release call. We never expose the raw pointer.
            unsafe {
                let _ = hailort_sys::hailo_release_vdevice(self.handle);
            }
            self.handle = ptr::null_mut();
        }
    }
}

// SAFETY: HailoRT vdevice handles are documented thread-safe across
// inference calls. Wrapping in our `Mutex<()>` at the embedder level
// covers the residual config-mutation cases.
#[cfg(feature = "hailo")]
unsafe impl Send for HailoDevice {}
#[cfg(feature = "hailo")]
unsafe impl Sync for HailoDevice {}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "hailo"))]
    #[test]
    fn open_without_feature_returns_feature_disabled() {
        match HailoDevice::open() {
            Err(HailoError::FeatureDisabled) => {}
            Err(e) => panic!("expected FeatureDisabled, got error: {}", e),
            Ok(_) => panic!("expected FeatureDisabled, got Ok"),
        }
    }

    #[cfg(feature = "hailo")]
    #[test]
    fn open_close_cycle_reads_version() {
        let dev = HailoDevice::open().expect("open vdevice on Pi 5 with HAT");
        let v = dev.version().expect("version triple");
        eprintln!("HailoRT {}.{}.{} via HailoDevice", v.0, v.1, v.2);
        assert!(v.0 >= 4);
        // Drop runs hailo_release_vdevice.
    }
}
