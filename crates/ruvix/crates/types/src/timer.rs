//! Timer types.
//!
//! Timers are deadline-driven scheduling primitives. The `timer_wait` syscall
//! allows tasks to sleep until a deadline or for a duration.

/// Timer specification for `timer_wait`.
///
/// Timers can be absolute (wait until a specific instant) or relative
/// (wait for a duration from now).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimerSpec {
    /// Wait until an absolute instant.
    ///
    /// The instant is represented as nanoseconds since kernel boot.
    Absolute {
        /// Nanoseconds since kernel boot.
        nanos_since_boot: u64,
    },

    /// Wait for a relative duration.
    ///
    /// The duration is represented as nanoseconds.
    Relative {
        /// Duration in nanoseconds.
        nanos: u64,
    },
}

impl TimerSpec {
    /// Creates an absolute timer specification.
    #[inline]
    #[must_use]
    pub const fn absolute(nanos_since_boot: u64) -> Self {
        Self::Absolute { nanos_since_boot }
    }

    /// Creates a relative timer specification.
    #[inline]
    #[must_use]
    pub const fn relative(nanos: u64) -> Self {
        Self::Relative { nanos }
    }

    /// Creates a relative timer from milliseconds.
    #[inline]
    #[must_use]
    pub const fn from_millis(millis: u64) -> Self {
        Self::Relative {
            nanos: millis * 1_000_000,
        }
    }

    /// Creates a relative timer from microseconds.
    #[inline]
    #[must_use]
    pub const fn from_micros(micros: u64) -> Self {
        Self::Relative {
            nanos: micros * 1_000,
        }
    }

    /// Creates a relative timer from seconds.
    #[inline]
    #[must_use]
    pub const fn from_secs(secs: u64) -> Self {
        Self::Relative {
            nanos: secs * 1_000_000_000,
        }
    }

    /// Returns true if this is an absolute timer.
    #[inline]
    #[must_use]
    pub const fn is_absolute(&self) -> bool {
        matches!(self, Self::Absolute { .. })
    }

    /// Returns true if this is a relative timer.
    #[inline]
    #[must_use]
    pub const fn is_relative(&self) -> bool {
        matches!(self, Self::Relative { .. })
    }

    /// Returns the nanoseconds value.
    #[inline]
    #[must_use]
    pub const fn nanos(&self) -> u64 {
        match self {
            Self::Absolute { nanos_since_boot } => *nanos_since_boot,
            Self::Relative { nanos } => *nanos,
        }
    }
}

impl Default for TimerSpec {
    fn default() -> Self {
        Self::Relative { nanos: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer_absolute() {
        let timer = TimerSpec::absolute(1_000_000_000);
        assert!(timer.is_absolute());
        assert!(!timer.is_relative());
        assert_eq!(timer.nanos(), 1_000_000_000);
    }

    #[test]
    fn test_timer_relative() {
        let timer = TimerSpec::from_millis(100);
        assert!(timer.is_relative());
        assert_eq!(timer.nanos(), 100_000_000);
    }

    #[test]
    fn test_timer_from_secs() {
        let timer = TimerSpec::from_secs(2);
        assert_eq!(timer.nanos(), 2_000_000_000);
    }
}
