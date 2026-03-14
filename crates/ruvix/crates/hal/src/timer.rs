//! # Timer Abstraction
//!
//! Provides traits for hardware timers, essential for scheduling and
//! deadline-based task management.
//!
//! ## Design
//!
//! - **Monotonic time** - Time always increases, never wraps
//! - **Nanosecond precision** - u64 nanoseconds (584 years before wrap)
//! - **Deadline scheduling** - Interrupt at specific time
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvix_hal::{Timer, timer::TimerError};
//!
//! fn schedule_task<T: Timer>(timer: &mut T, delay_ms: u64) -> Result<(), TimerError> {
//!     let now = timer.now_ns();
//!     let deadline = now + (delay_ms * 1_000_000);
//!     timer.set_deadline(deadline)?;
//!     Ok(())
//! }
//! ```

/// Timer error types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimerError {
    /// Deadline is in the past
    DeadlineInPast,
    /// Hardware is not initialized
    NotInitialized,
    /// Hardware fault (e.g., counter overflow)
    HardwareFault,
}

impl core::fmt::Display for TimerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DeadlineInPast => write!(f, "deadline is in the past"),
            Self::NotInitialized => write!(f, "timer not initialized"),
            Self::HardwareFault => write!(f, "timer hardware fault"),
        }
    }
}

/// Timer abstraction for monotonic time and deadline scheduling
///
/// This trait provides access to a hardware timer with nanosecond precision.
/// Typical implementations use ARM Generic Timer or similar hardware.
///
/// ## Thread Safety
///
/// All methods MUST be safe to call from interrupt context.
///
/// ## Example Implementation (ARM Generic Timer)
///
/// ```rust,ignore
/// use ruvix_hal::{Timer, timer::TimerError};
///
/// struct ArmGenericTimer {
///     frequency_hz: u64,
/// }
///
/// impl ArmGenericTimer {
///     fn new() -> Self {
///         // Read CNTFRQ_EL0 to get timer frequency
///         let frequency_hz = unsafe {
///             let mut freq: u64;
///             core::arch::asm!(
///                 "mrs {}, cntfrq_el0",
///                 out(reg) freq,
///             );
///             freq
///         };
///
///         Self { frequency_hz }
///     }
///
///     fn read_counter(&self) -> u64 {
///         // Read CNTPCT_EL0 (physical counter)
///         unsafe {
///             let mut count: u64;
///             core::arch::asm!(
///                 "mrs {}, cntpct_el0",
///                 out(reg) count,
///             );
///             count
///         }
///     }
///
///     fn write_compare(&mut self, value: u64) {
///         // Write CNTP_CVAL_EL0 (compare value)
///         unsafe {
///             core::arch::asm!(
///                 "msr cntp_cval_el0, {}",
///                 in(reg) value,
///             );
///         }
///     }
///
///     fn enable_interrupt(&mut self) {
///         // Enable timer interrupt (CNTP_CTL_EL0.ENABLE = 1, IMASK = 0)
///         unsafe {
///             core::arch::asm!(
///                 "msr cntp_ctl_el0, {val}",
///                 val = in(reg) 1u64, // ENABLE=1, IMASK=0
///             );
///         }
///     }
///
///     fn disable_interrupt(&mut self) {
///         // Disable timer interrupt (CNTP_CTL_EL0.ENABLE = 0)
///         unsafe {
///             core::arch::asm!(
///                 "msr cntp_ctl_el0, {val}",
///                 val = in(reg) 0u64,
///             );
///         }
///     }
/// }
///
/// impl Timer for ArmGenericTimer {
///     fn now_ns(&self) -> u64 {
///         let ticks = self.read_counter();
///         // Convert ticks to nanoseconds
///         ticks * 1_000_000_000 / self.frequency_hz
///     }
///
///     fn frequency_hz(&self) -> u64 {
///         self.frequency_hz
///     }
///
///     fn set_deadline(&mut self, deadline_ns: u64) -> Result<(), TimerError> {
///         let now = self.now_ns();
///         if deadline_ns <= now {
///             return Err(TimerError::DeadlineInPast);
///         }
///
///         // Convert nanoseconds to ticks
///         let deadline_ticks = deadline_ns * self.frequency_hz / 1_000_000_000;
///         self.write_compare(deadline_ticks);
///         self.enable_interrupt();
///
///         Ok(())
///     }
///
///     fn clear_deadline(&mut self) -> Result<(), TimerError> {
///         self.disable_interrupt();
///         Ok(())
///     }
///
///     fn is_deadline_pending(&self) -> bool {
///         // Read CNTP_CTL_EL0.ISTATUS bit
///         unsafe {
///             let mut ctl: u64;
///             core::arch::asm!(
///                 "mrs {}, cntp_ctl_el0",
///                 out(reg) ctl,
///             );
///             (ctl & 0b100) != 0 // ISTATUS is bit 2
///         }
///     }
/// }
/// ```
pub trait Timer {
    /// Get current monotonic time in nanoseconds
    ///
    /// This MUST be a monotonic clock (never goes backwards).
    /// The epoch is undefined (typically boot time).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ruvix_hal::Timer;
    ///
    /// fn measure_latency<T: Timer>(timer: &T, f: impl FnOnce()) -> u64 {
    ///     let start = timer.now_ns();
    ///     f();
    ///     let end = timer.now_ns();
    ///     end - start
    /// }
    /// ```
    fn now_ns(&self) -> u64;

    /// Get timer frequency in Hz
    ///
    /// This is the tick rate of the underlying hardware counter.
    fn frequency_hz(&self) -> u64;

    /// Set deadline for timer interrupt
    ///
    /// The timer will fire an interrupt when `now_ns() >= deadline_ns`.
    ///
    /// # Arguments
    ///
    /// * `deadline_ns` - Absolute time in nanoseconds
    ///
    /// # Errors
    ///
    /// Returns `DeadlineInPast` if deadline has already passed.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ruvix_hal::Timer;
    ///
    /// fn sleep_ms<T: Timer>(timer: &mut T, ms: u64) -> Result<(), ruvix_hal::timer::TimerError> {
    ///     let deadline = timer.now_ns() + (ms * 1_000_000);
    ///     timer.set_deadline(deadline)?;
    ///     // Wait for interrupt...
    ///     Ok(())
    /// }
    /// ```
    fn set_deadline(&mut self, deadline_ns: u64) -> Result<(), TimerError>;

    /// Clear pending deadline
    ///
    /// Disables the timer interrupt.
    ///
    /// # Errors
    ///
    /// Returns `NotInitialized` if timer hardware is not ready.
    fn clear_deadline(&mut self) -> Result<(), TimerError>;

    /// Check if deadline interrupt is pending
    ///
    /// Returns `true` if the deadline has passed and interrupt is pending.
    fn is_deadline_pending(&self) -> bool;

    /// Busy-wait for a duration in nanoseconds
    ///
    /// This is a blocking spin-wait, useful for very short delays.
    ///
    /// # Arguments
    ///
    /// * `duration_ns` - Duration to wait in nanoseconds
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ruvix_hal::Timer;
    ///
    /// fn delay_us<T: Timer>(timer: &T, us: u64) {
    ///     timer.busy_wait(us * 1_000);
    /// }
    /// ```
    fn busy_wait(&self, duration_ns: u64) {
        let start = self.now_ns();
        while self.now_ns() - start < duration_ns {
            core::hint::spin_loop();
        }
    }

    /// Get elapsed time since a previous timestamp
    ///
    /// # Arguments
    ///
    /// * `since_ns` - Previous timestamp from `now_ns()`
    ///
    /// # Returns
    ///
    /// Elapsed nanoseconds, or 0 if `since_ns` is in the future.
    fn elapsed_since(&self, since_ns: u64) -> u64 {
        let now = self.now_ns();
        now.saturating_sub(since_ns)
    }
}

/// Helper for measuring execution time
///
/// # Example
///
/// ```rust,ignore
/// use ruvix_hal::{Timer, timer::Stopwatch};
///
/// fn benchmark<T: Timer>(timer: &T) {
///     let stopwatch = Stopwatch::start(timer);
///     expensive_operation();
///     let elapsed = stopwatch.elapsed_ns(timer);
///     println!("Took {} ns", elapsed);
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Stopwatch {
    start_ns: u64,
}

impl Stopwatch {
    /// Start a new stopwatch
    pub fn start<T: Timer>(timer: &T) -> Self {
        Self {
            start_ns: timer.now_ns(),
        }
    }

    /// Get elapsed time in nanoseconds
    pub fn elapsed_ns<T: Timer>(&self, timer: &T) -> u64 {
        timer.elapsed_since(self.start_ns)
    }

    /// Get elapsed time in microseconds
    pub fn elapsed_us<T: Timer>(&self, timer: &T) -> u64 {
        self.elapsed_ns(timer) / 1_000
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed_ms<T: Timer>(&self, timer: &T) -> u64 {
        self.elapsed_ns(timer) / 1_000_000
    }

    /// Restart the stopwatch
    pub fn restart<T: Timer>(&mut self, timer: &T) {
        self.start_ns = timer.now_ns();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockTimer {
        current_ns: u64,
        frequency: u64,
        deadline: Option<u64>,
    }

    impl MockTimer {
        fn new() -> Self {
            Self {
                current_ns: 0,
                frequency: 1_000_000_000, // 1 GHz
                deadline: None,
            }
        }

        fn advance(&mut self, ns: u64) {
            self.current_ns += ns;
        }
    }

    impl Timer for MockTimer {
        fn now_ns(&self) -> u64 {
            self.current_ns
        }

        fn frequency_hz(&self) -> u64 {
            self.frequency
        }

        fn set_deadline(&mut self, deadline_ns: u64) -> Result<(), TimerError> {
            if deadline_ns <= self.current_ns {
                return Err(TimerError::DeadlineInPast);
            }
            self.deadline = Some(deadline_ns);
            Ok(())
        }

        fn clear_deadline(&mut self) -> Result<(), TimerError> {
            self.deadline = None;
            Ok(())
        }

        fn is_deadline_pending(&self) -> bool {
            self.deadline.map_or(false, |d| self.current_ns >= d)
        }
    }

    #[test]
    fn test_now_ns() {
        let mut timer = MockTimer::new();
        assert_eq!(timer.now_ns(), 0);

        timer.advance(1_000_000);
        assert_eq!(timer.now_ns(), 1_000_000);
    }

    #[test]
    fn test_set_deadline() {
        let mut timer = MockTimer::new();

        timer.set_deadline(5_000_000).unwrap();
        assert_eq!(timer.deadline, Some(5_000_000));
        assert!(!timer.is_deadline_pending());

        timer.advance(5_000_000);
        assert!(timer.is_deadline_pending());
    }

    #[test]
    fn test_deadline_in_past() {
        let mut timer = MockTimer::new();
        timer.advance(10_000_000);

        let result = timer.set_deadline(5_000_000);
        assert_eq!(result, Err(TimerError::DeadlineInPast));
    }

    #[test]
    fn test_stopwatch() {
        let mut timer = MockTimer::new();

        let stopwatch = Stopwatch::start(&timer);
        timer.advance(2_000_000);

        assert_eq!(stopwatch.elapsed_ns(&timer), 2_000_000);
        assert_eq!(stopwatch.elapsed_us(&timer), 2_000);
        assert_eq!(stopwatch.elapsed_ms(&timer), 2);
    }

    #[test]
    fn test_elapsed_since() {
        let mut timer = MockTimer::new();

        let start = timer.now_ns();
        timer.advance(3_000_000);

        assert_eq!(timer.elapsed_since(start), 3_000_000);
    }
}
