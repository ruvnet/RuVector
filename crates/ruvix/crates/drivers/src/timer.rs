//! # ARM Generic Timer Driver
//!
//! This module implements a driver for the ARM Generic Timer, which provides
//! a system-wide monotonic counter and per-CPU timer interrupts.
//!
//! ## Architecture
//!
//! The ARM Generic Timer has several components:
//!
//! - **CNTPCT_EL0** - Physical counter (read-only, monotonic)
//! - **CNTFRQ_EL0** - Counter frequency in Hz
//! - **CNTP_CTL_EL0** - Timer control register
//! - **CNTP_CVAL_EL0** - Timer compare value (absolute time)
//! - **CNTP_TVAL_EL0** - Timer value (relative time)
//!
//! ## Timer Modes
//!
//! - **Relative (TVAL)** - Set timer to fire after N ticks
//! - **Absolute (CVAL)** - Set timer to fire at specific counter value
//!
//! ## Example
//!
//! ```rust,no_run
//! use ruvix_drivers::timer::ArmGenericTimer;
//!
//! let timer = ArmGenericTimer::new();
//!
//! // Get current time in nanoseconds
//! let now = timer.now_ns();
//!
//! // Set timer to fire in 1 second
//! let deadline = now + 1_000_000_000;
//! timer.set_deadline_ns(deadline).expect("Failed to set timer");
//!
//! // Enable timer interrupt
//! timer.enable().expect("Failed to enable timer");
//! ```

/// Timer Control Register (CNTP_CTL_EL0) bits
const CTL_ENABLE: u64 = 1 << 0; // Enable timer
const CTL_IMASK: u64 = 1 << 1; // Interrupt mask
const CTL_ISTATUS: u64 = 1 << 2; // Interrupt status

/// ARM Generic Timer driver
pub struct ArmGenericTimer {
    frequency_hz: u64,
}

impl ArmGenericTimer {
    /// Create a new ARM Generic Timer driver.
    ///
    /// This reads the counter frequency from `CNTFRQ_EL0`.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use ruvix_drivers::timer::ArmGenericTimer;
    ///
    /// let timer = ArmGenericTimer::new();
    /// ```
    #[inline]
    pub fn new() -> Self {
        let frequency_hz = Self::read_frequency();
        Self { frequency_hz }
    }

    /// Read the counter frequency from CNTFRQ_EL0.
    #[inline]
    fn read_frequency() -> u64 {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let freq: u64;
            core::arch::asm!(
                "mrs {freq}, cntfrq_el0",
                freq = out(reg) freq,
                options(nostack, preserves_flags)
            );
            freq
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            // Default to 62.5 MHz for testing
            62_500_000
        }
    }

    /// Read the physical counter value.
    #[inline]
    fn read_counter() -> u64 {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cnt: u64;
            core::arch::asm!(
                "mrs {cnt}, cntpct_el0",
                cnt = out(reg) cnt,
                options(nostack, preserves_flags)
            );
            cnt
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            // Return a dummy value for testing
            0
        }
    }

    /// Get the current time in nanoseconds.
    ///
    /// This is a monotonic timestamp based on the physical counter.
    #[inline]
    pub fn now_ns(&self) -> u64 {
        let counter = Self::read_counter();
        // Convert ticks to nanoseconds: (counter * 1_000_000_000) / frequency
        // Use 128-bit arithmetic to avoid overflow
        ((counter as u128 * 1_000_000_000) / self.frequency_hz as u128) as u64
    }

    /// Get the current counter value (raw ticks).
    #[inline]
    pub fn now_ticks(&self) -> u64 {
        Self::read_counter()
    }

    /// Get the timer frequency in Hz.
    #[inline]
    pub fn frequency(&self) -> u64 {
        self.frequency_hz
    }

    /// Set timer compare value (absolute time in ticks).
    ///
    /// # Arguments
    ///
    /// - `cval` - Absolute counter value at which interrupt should fire
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if the operation fails.
    pub fn set_compare(&self, cval: u64) -> Result<(), ()> {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!(
                "msr cntp_cval_el0, {cval}",
                cval = in(reg) cval,
                options(nostack, preserves_flags)
            );
        }

        let _ = cval; // Suppress unused warning on non-aarch64
        Ok(())
    }

    /// Set timer to fire at an absolute deadline (in nanoseconds).
    ///
    /// # Arguments
    ///
    /// - `deadline_ns` - Absolute time in nanoseconds when interrupt should fire
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if the operation fails.
    pub fn set_deadline_ns(&self, deadline_ns: u64) -> Result<(), ()> {
        // Convert nanoseconds to ticks: (deadline_ns * frequency) / 1_000_000_000
        let ticks = ((deadline_ns as u128 * self.frequency_hz as u128) / 1_000_000_000) as u64;
        self.set_compare(ticks)
    }

    /// Set timer to fire after a relative delay (in nanoseconds).
    ///
    /// # Arguments
    ///
    /// - `delay_ns` - Relative delay in nanoseconds
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if the operation fails.
    pub fn set_timeout_ns(&self, delay_ns: u64) -> Result<(), ()> {
        let now = self.now_ns();
        self.set_deadline_ns(now.saturating_add(delay_ns))
    }

    /// Enable the timer and unmask interrupts.
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if the operation fails.
    pub fn enable(&self) -> Result<(), ()> {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!(
                "msr cntp_ctl_el0, {ctl}",
                ctl = in(reg) CTL_ENABLE,
                options(nostack, preserves_flags)
            );
        }

        Ok(())
    }

    /// Disable the timer.
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if the operation fails.
    pub fn disable(&self) -> Result<(), ()> {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!(
                "msr cntp_ctl_el0, {ctl}",
                ctl = in(reg) 0u64,
                options(nostack, preserves_flags)
            );
        }

        Ok(())
    }

    /// Check if timer interrupt is pending.
    pub fn is_pending(&self) -> bool {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let ctl: u64;
            core::arch::asm!(
                "mrs {ctl}, cntp_ctl_el0",
                ctl = out(reg) ctl,
                options(nostack, preserves_flags)
            );
            (ctl & CTL_ISTATUS) != 0
        }

        #[cfg(not(target_arch = "aarch64"))]
        false
    }

    /// Acknowledge timer interrupt by disabling and re-enabling.
    ///
    /// # Errors
    ///
    /// Returns `Err(())` if the operation fails.
    pub fn acknowledge(&self) -> Result<(), ()> {
        // Clear pending state by disabling timer momentarily
        self.disable()?;
        self.enable()
    }
}

impl Default for ArmGenericTimer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer_new() {
        let timer = ArmGenericTimer::new();
        assert!(timer.frequency() > 0);
    }

    #[test]
    fn test_time_conversion() {
        let timer = ArmGenericTimer::new();

        // Test nanosecond conversion
        let now_ns = timer.now_ns();
        let now_ticks = timer.now_ticks();

        // now_ns should be proportional to now_ticks
        let expected_ns = ((now_ticks as u128 * 1_000_000_000) / timer.frequency() as u128) as u64;
        assert_eq!(now_ns, expected_ns);
    }

    #[test]
    fn test_deadline_conversion() {
        let timer = ArmGenericTimer::new();

        // Set deadline 1 second in the future
        let deadline_ns = timer.now_ns() + 1_000_000_000;
        assert!(timer.set_deadline_ns(deadline_ns).is_ok());
    }
}
