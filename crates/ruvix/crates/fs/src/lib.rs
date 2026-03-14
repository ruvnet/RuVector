//! # RuVix Filesystem Layer
//!
//! This crate provides a minimal filesystem abstraction for the RuVix Cognition Kernel
//! as specified in ADR-087 Phase E. It implements a VFS layer with pluggable filesystem
//! backends including FAT32 (read-only) and RamFS (read-write).
//!
//! ## Architecture
//!
//! The filesystem layer is designed around these core abstractions:
//!
//! - **`BlockDevice`**: Hardware abstraction for block-level I/O
//! - **`FileSystem`**: Filesystem implementation (FAT32, RamFS, etc.)
//! - **`Inode`**: File/directory representation with read/write operations
//! - **`VfsMountPoint`**: Mount point management for the VFS tree
//!
//! ## Features
//!
//! - `std`: Enable standard library support
//! - `alloc`: Enable alloc crate support for heap allocation
//! - `lfn`: Enable FAT32 long filename support
//! - `fat32-write`: Enable write support for FAT32 (Phase 2)
//! - `stats`: Enable statistics collection
//!
//! ## Example
//!
//! ```ignore
//! use ruvix_fs::{VfsMountTable, RamFs, Fat32Fs};
//!
//! // Create mount table
//! let mut mounts = VfsMountTable::new();
//!
//! // Mount RamFS at /tmp
//! let ramfs = RamFs::new();
//! mounts.mount("/tmp", ramfs)?;
//!
//! // Mount FAT32 at /boot
//! let fat32 = Fat32Fs::new(block_device)?;
//! mounts.mount("/boot", fat32)?;
//! ```

#![no_std]
#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

mod block;
mod error;
mod fat32;
mod path;
mod ramfs;
mod vfs;

pub use block::{BlockDevice, NullBlockDevice};
#[cfg(feature = "alloc")]
pub use block::{MemoryBlockDevice, MemoryBlockDeviceMut};
pub use error::{FsError, FsResult};
pub use fat32::{Fat32BootSector, Fat32DirEntry};
#[cfg(feature = "alloc")]
pub use fat32::{Fat32Fs, Fat32Inode};
pub use path::{Path, PathComponent, PathIter};
#[cfg(feature = "alloc")]
pub use path::PathBuf;
#[cfg(feature = "alloc")]
pub use ramfs::{RamFs, RamInode, RamInodeType};
pub use vfs::{
    DirEntry, FileSystem, FileType, Inode, InodeId, InodeOps, MountId, OpenFile,
    OpenFlags, SeekFrom,
};
#[cfg(feature = "alloc")]
pub use vfs::{OpenFileTable, VfsMountPoint, VfsMountTable};

/// Maximum path length in bytes (POSIX PATH_MAX equivalent).
pub const MAX_PATH_LEN: usize = 4096;

/// Maximum filename length in bytes.
pub const MAX_NAME_LEN: usize = 255;

/// Default block size for filesystems (4KB).
pub const DEFAULT_BLOCK_SIZE: usize = 4096;

/// FAT32 sector size (always 512 bytes per specification).
pub const FAT32_SECTOR_SIZE: usize = 512;

/// Maximum number of mount points supported.
pub const MAX_MOUNTS: usize = 64;

/// Maximum number of open files per process.
pub const MAX_OPEN_FILES: usize = 256;

/// Root inode ID (always 1 by convention).
pub const ROOT_INODE_ID: InodeId = InodeId(1);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(MAX_PATH_LEN, 4096);
        assert_eq!(MAX_NAME_LEN, 255);
        assert_eq!(DEFAULT_BLOCK_SIZE, 4096);
        assert_eq!(FAT32_SECTOR_SIZE, 512);
        assert_eq!(MAX_MOUNTS, 64);
        assert_eq!(MAX_OPEN_FILES, 256);
        assert_eq!(ROOT_INODE_ID, InodeId(1));
    }
}
