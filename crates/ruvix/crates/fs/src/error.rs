//! Filesystem error types for RuVix.
//!
//! This module provides error types used throughout the filesystem layer,
//! designed to be compatible with both `no_std` and `std` environments.

use core::fmt;

/// Result type alias for filesystem operations.
pub type FsResult<T> = Result<T, FsError>;

/// Filesystem error types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsError {
    /// File or directory not found.
    NotFound,

    /// Permission denied.
    PermissionDenied,

    /// File or directory already exists.
    AlreadyExists,

    /// Not a directory.
    NotADirectory,

    /// Is a directory (when file expected).
    IsADirectory,

    /// Directory not empty.
    DirectoryNotEmpty,

    /// Invalid path (malformed or too long).
    InvalidPath,

    /// Invalid filename.
    InvalidFilename,

    /// Path too long.
    PathTooLong,

    /// Name too long.
    NameTooLong,

    /// Too many symbolic links encountered.
    TooManySymlinks,

    /// Invalid argument.
    InvalidArgument,

    /// I/O error during block device operation.
    IoError,

    /// Device or resource busy.
    Busy,

    /// No space left on device.
    NoSpace,

    /// Read-only filesystem.
    ReadOnly,

    /// Invalid filesystem (corrupted or unsupported).
    InvalidFilesystem,

    /// Invalid FAT entry.
    InvalidFatEntry,

    /// Invalid cluster chain.
    InvalidClusterChain,

    /// Block device error.
    BlockDeviceError,

    /// Mount point not found.
    MountNotFound,

    /// Mount point already exists.
    MountAlreadyExists,

    /// Too many mount points.
    TooManyMounts,

    /// File too large.
    FileTooLarge,

    /// Invalid seek position.
    InvalidSeek,

    /// End of file.
    EndOfFile,

    /// Invalid file descriptor.
    InvalidFileDescriptor,

    /// Too many open files.
    TooManyOpenFiles,

    /// Cross-device link (rename across filesystems).
    CrossDeviceLink,

    /// Not supported by this filesystem.
    NotSupported,

    /// Inode not found.
    InodeNotFound,

    /// Out of inodes.
    OutOfInodes,

    /// Invalid boot sector.
    InvalidBootSector,

    /// Feature not enabled.
    FeatureNotEnabled,
}

impl FsError {
    /// Convert error to a POSIX-like error number.
    #[must_use]
    pub const fn to_errno(self) -> i32 {
        match self {
            Self::NotFound => 2,            // ENOENT
            Self::PermissionDenied => 13,   // EACCES
            Self::AlreadyExists => 17,      // EEXIST
            Self::NotADirectory => 20,      // ENOTDIR
            Self::IsADirectory => 21,       // EISDIR
            Self::DirectoryNotEmpty => 39,  // ENOTEMPTY
            Self::InvalidPath => 22,        // EINVAL
            Self::InvalidFilename => 22,    // EINVAL
            Self::PathTooLong => 36,        // ENAMETOOLONG
            Self::NameTooLong => 36,        // ENAMETOOLONG
            Self::TooManySymlinks => 40,    // ELOOP
            Self::InvalidArgument => 22,    // EINVAL
            Self::IoError => 5,             // EIO
            Self::Busy => 16,               // EBUSY
            Self::NoSpace => 28,            // ENOSPC
            Self::ReadOnly => 30,           // EROFS
            Self::InvalidFilesystem => 5,   // EIO
            Self::InvalidFatEntry => 5,     // EIO
            Self::InvalidClusterChain => 5, // EIO
            Self::BlockDeviceError => 5,    // EIO
            Self::MountNotFound => 2,       // ENOENT
            Self::MountAlreadyExists => 17, // EEXIST
            Self::TooManyMounts => 12,      // ENOMEM
            Self::FileTooLarge => 27,       // EFBIG
            Self::InvalidSeek => 29,        // ESPIPE
            Self::EndOfFile => 0,           // Success (EOF is not an error)
            Self::InvalidFileDescriptor => 9, // EBADF
            Self::TooManyOpenFiles => 24,   // EMFILE
            Self::CrossDeviceLink => 18,    // EXDEV
            Self::NotSupported => 95,       // EOPNOTSUPP
            Self::InodeNotFound => 2,       // ENOENT
            Self::OutOfInodes => 28,        // ENOSPC
            Self::InvalidBootSector => 5,   // EIO
            Self::FeatureNotEnabled => 95,  // EOPNOTSUPP
        }
    }

    /// Returns a human-readable error message.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotFound => "No such file or directory",
            Self::PermissionDenied => "Permission denied",
            Self::AlreadyExists => "File exists",
            Self::NotADirectory => "Not a directory",
            Self::IsADirectory => "Is a directory",
            Self::DirectoryNotEmpty => "Directory not empty",
            Self::InvalidPath => "Invalid path",
            Self::InvalidFilename => "Invalid filename",
            Self::PathTooLong => "Path too long",
            Self::NameTooLong => "Filename too long",
            Self::TooManySymlinks => "Too many symbolic links",
            Self::InvalidArgument => "Invalid argument",
            Self::IoError => "I/O error",
            Self::Busy => "Device or resource busy",
            Self::NoSpace => "No space left on device",
            Self::ReadOnly => "Read-only file system",
            Self::InvalidFilesystem => "Invalid or corrupted filesystem",
            Self::InvalidFatEntry => "Invalid FAT entry",
            Self::InvalidClusterChain => "Invalid cluster chain",
            Self::BlockDeviceError => "Block device error",
            Self::MountNotFound => "Mount point not found",
            Self::MountAlreadyExists => "Mount point already exists",
            Self::TooManyMounts => "Too many mount points",
            Self::FileTooLarge => "File too large",
            Self::InvalidSeek => "Invalid seek",
            Self::EndOfFile => "End of file",
            Self::InvalidFileDescriptor => "Bad file descriptor",
            Self::TooManyOpenFiles => "Too many open files",
            Self::CrossDeviceLink => "Cross-device link",
            Self::NotSupported => "Operation not supported",
            Self::InodeNotFound => "Inode not found",
            Self::OutOfInodes => "Out of inodes",
            Self::InvalidBootSector => "Invalid boot sector",
            Self::FeatureNotEnabled => "Feature not enabled",
        }
    }
}

impl fmt::Display for FsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FsError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_to_errno() {
        assert_eq!(FsError::NotFound.to_errno(), 2);
        assert_eq!(FsError::PermissionDenied.to_errno(), 13);
        assert_eq!(FsError::AlreadyExists.to_errno(), 17);
        assert_eq!(FsError::IoError.to_errno(), 5);
        assert_eq!(FsError::NoSpace.to_errno(), 28);
        assert_eq!(FsError::ReadOnly.to_errno(), 30);
    }

    #[test]
    fn test_error_display() {
        assert_eq!(FsError::NotFound.as_str(), "No such file or directory");
        assert_eq!(FsError::PermissionDenied.as_str(), "Permission denied");
        assert_eq!(FsError::ReadOnly.as_str(), "Read-only file system");
    }

    #[test]
    fn test_error_clone_eq() {
        let e1 = FsError::NotFound;
        let e2 = e1;
        assert_eq!(e1, e2);
    }
}
