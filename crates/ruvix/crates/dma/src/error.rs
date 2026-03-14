//! DMA error types and result alias.

use crate::DmaErrorKind;
use core::fmt;

/// Result type alias for DMA operations.
pub type DmaResult<T> = Result<T, DmaError>;

/// Error type for DMA operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DmaError {
    /// The kind of error that occurred.
    kind: DmaErrorKind,
    /// Optional channel ID associated with the error.
    channel_id: Option<u8>,
    /// Optional additional context (e.g., address that caused the error).
    context: u64,
}

impl DmaError {
    /// Create a new DMA error.
    #[must_use]
    pub const fn new(kind: DmaErrorKind) -> Self {
        Self {
            kind,
            channel_id: None,
            context: 0,
        }
    }

    /// Create a DMA error with channel context.
    #[must_use]
    pub const fn with_channel(kind: DmaErrorKind, channel_id: u8) -> Self {
        Self {
            kind,
            channel_id: Some(channel_id),
            context: 0,
        }
    }

    /// Create a DMA error with address context.
    #[must_use]
    pub const fn with_address(kind: DmaErrorKind, address: u64) -> Self {
        Self {
            kind,
            channel_id: None,
            context: address,
        }
    }

    /// Create a DMA error with full context.
    #[must_use]
    pub const fn with_full_context(kind: DmaErrorKind, channel_id: u8, address: u64) -> Self {
        Self {
            kind,
            channel_id: Some(channel_id),
            context: address,
        }
    }

    /// Get the error kind.
    #[must_use]
    pub const fn kind(&self) -> DmaErrorKind {
        self.kind
    }

    /// Get the associated channel ID, if any.
    #[must_use]
    pub const fn channel_id(&self) -> Option<u8> {
        self.channel_id
    }

    /// Get the context value (typically an address).
    #[must_use]
    pub const fn context(&self) -> u64 {
        self.context
    }

    /// Create a bus error.
    #[must_use]
    pub const fn bus_error() -> Self {
        Self::new(DmaErrorKind::BusError)
    }

    /// Create an alignment error.
    #[must_use]
    pub const fn alignment_error(address: u64) -> Self {
        Self::with_address(DmaErrorKind::AlignmentError, address)
    }

    /// Create a timeout error.
    #[must_use]
    pub const fn timeout() -> Self {
        Self::new(DmaErrorKind::Timeout)
    }

    /// Create a descriptor error.
    #[must_use]
    pub const fn descriptor_error() -> Self {
        Self::new(DmaErrorKind::DescriptorError)
    }

    /// Create a configuration error.
    #[must_use]
    pub const fn config_error() -> Self {
        Self::new(DmaErrorKind::ConfigError)
    }

    /// Create a permission denied error.
    #[must_use]
    pub const fn permission_denied() -> Self {
        Self::new(DmaErrorKind::PermissionDenied)
    }

    /// Create a channel not available error.
    #[must_use]
    pub const fn channel_not_available() -> Self {
        Self::new(DmaErrorKind::ConfigError)
    }

    /// Create a channel busy error.
    #[must_use]
    pub const fn channel_busy(channel_id: u8) -> Self {
        Self::with_channel(DmaErrorKind::ConfigError, channel_id)
    }

    /// Create an invalid descriptor chain error.
    #[must_use]
    pub const fn invalid_descriptor_chain() -> Self {
        Self::new(DmaErrorKind::DescriptorError)
    }
}

impl fmt::Display for DmaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            DmaErrorKind::BusError => write!(f, "DMA bus error"),
            DmaErrorKind::AlignmentError => {
                write!(f, "DMA alignment error at address 0x{:016x}", self.context)
            }
            DmaErrorKind::Timeout => write!(f, "DMA transfer timeout"),
            DmaErrorKind::DescriptorError => write!(f, "DMA descriptor chain error"),
            DmaErrorKind::ConfigError => write!(f, "DMA configuration error"),
            DmaErrorKind::PermissionDenied => write!(f, "DMA permission denied"),
            DmaErrorKind::Unknown => write!(f, "Unknown DMA error"),
        }?;

        if let Some(ch) = self.channel_id {
            write!(f, " (channel {})", ch)?;
        }

        Ok(())
    }
}

impl From<DmaErrorKind> for DmaError {
    fn from(kind: DmaErrorKind) -> Self {
        Self::new(kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = DmaError::new(DmaErrorKind::BusError);
        assert_eq!(err.kind(), DmaErrorKind::BusError);
        assert_eq!(err.channel_id(), None);
        assert_eq!(err.context(), 0);
    }

    #[test]
    fn test_error_with_channel() {
        let err = DmaError::with_channel(DmaErrorKind::Timeout, 5);
        assert_eq!(err.kind(), DmaErrorKind::Timeout);
        assert_eq!(err.channel_id(), Some(5));
    }

    #[test]
    fn test_error_with_address() {
        let err = DmaError::alignment_error(0xDEAD_BEEF);
        assert_eq!(err.kind(), DmaErrorKind::AlignmentError);
        assert_eq!(err.context(), 0xDEAD_BEEF);
    }

    #[test]
    fn test_error_convenience_constructors() {
        assert_eq!(DmaError::bus_error().kind(), DmaErrorKind::BusError);
        assert_eq!(DmaError::timeout().kind(), DmaErrorKind::Timeout);
        assert_eq!(
            DmaError::descriptor_error().kind(),
            DmaErrorKind::DescriptorError
        );
        assert_eq!(DmaError::config_error().kind(), DmaErrorKind::ConfigError);
        assert_eq!(
            DmaError::permission_denied().kind(),
            DmaErrorKind::PermissionDenied
        );
    }

    #[test]
    fn test_error_from_kind() {
        let err: DmaError = DmaErrorKind::BusError.into();
        assert_eq!(err.kind(), DmaErrorKind::BusError);
    }
}
