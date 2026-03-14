//! Virtual Filesystem (VFS) layer for RuVix.
//!
//! This module provides the core VFS abstractions including the `FileSystem` trait,
//! `Inode` operations, and mount table management.

use crate::error::{FsError, FsResult};
use crate::path::Path;
use crate::{MAX_MOUNTS, MAX_NAME_LEN, MAX_OPEN_FILES};
use core::fmt;

#[cfg(feature = "alloc")]
use alloc::{string::String, vec::Vec};

/// Unique identifier for an inode within a filesystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct InodeId(pub u64);

impl InodeId {
    /// The root inode ID (always 1).
    pub const ROOT: Self = Self(1);

    /// Invalid inode ID.
    pub const INVALID: Self = Self(0);

    /// Check if this is a valid inode ID.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.0 != 0
    }
}

impl fmt::Display for InodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a mount point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct MountId(pub u32);

impl MountId {
    /// The root mount ID.
    pub const ROOT: Self = Self(0);

    /// Invalid mount ID.
    pub const INVALID: Self = Self(u32::MAX);
}

/// File type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    /// Regular file.
    Regular,
    /// Directory.
    Directory,
    /// Symbolic link.
    Symlink,
    /// Block device.
    BlockDevice,
    /// Character device.
    CharDevice,
    /// Named pipe (FIFO).
    Fifo,
    /// Unix domain socket.
    Socket,
    /// Unknown file type.
    Unknown,
}

impl FileType {
    /// Returns the mode bits for this file type.
    #[must_use]
    pub const fn to_mode(self) -> u32 {
        match self {
            Self::Regular => 0o100000,
            Self::Directory => 0o040000,
            Self::Symlink => 0o120000,
            Self::BlockDevice => 0o060000,
            Self::CharDevice => 0o020000,
            Self::Fifo => 0o010000,
            Self::Socket => 0o140000,
            Self::Unknown => 0,
        }
    }

    /// Create a `FileType` from mode bits.
    #[must_use]
    pub const fn from_mode(mode: u32) -> Self {
        match mode & 0o170000 {
            0o100000 => Self::Regular,
            0o040000 => Self::Directory,
            0o120000 => Self::Symlink,
            0o060000 => Self::BlockDevice,
            0o020000 => Self::CharDevice,
            0o010000 => Self::Fifo,
            0o140000 => Self::Socket,
            _ => Self::Unknown,
        }
    }
}

impl Default for FileType {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Directory entry structure.
#[derive(Debug, Clone)]
pub struct DirEntry {
    /// Entry name.
    pub name: [u8; MAX_NAME_LEN],
    /// Name length.
    pub name_len: usize,
    /// Inode ID.
    pub inode_id: InodeId,
    /// File type.
    pub file_type: FileType,
}

impl DirEntry {
    /// Create a new directory entry.
    #[must_use]
    pub fn new(name: &str, inode_id: InodeId, file_type: FileType) -> Self {
        let mut entry = Self {
            name: [0u8; MAX_NAME_LEN],
            name_len: 0,
            inode_id,
            file_type,
        };
        entry.set_name(name);
        entry
    }

    /// Set the entry name.
    pub fn set_name(&mut self, name: &str) {
        let bytes = name.as_bytes();
        let len = bytes.len().min(MAX_NAME_LEN);
        self.name[..len].copy_from_slice(&bytes[..len]);
        self.name_len = len;
    }

    /// Get the entry name as a string slice.
    #[must_use]
    pub fn name(&self) -> &str {
        // SAFETY: We only store valid UTF-8 in name
        core::str::from_utf8(&self.name[..self.name_len]).unwrap_or("")
    }
}

impl Default for DirEntry {
    fn default() -> Self {
        Self {
            name: [0u8; MAX_NAME_LEN],
            name_len: 0,
            inode_id: InodeId::INVALID,
            file_type: FileType::Unknown,
        }
    }
}

/// Flags for opening files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenFlags(u32);

impl OpenFlags {
    /// Read-only access.
    pub const READ: Self = Self(0x0001);
    /// Write-only access.
    pub const WRITE: Self = Self(0x0002);
    /// Read-write access.
    pub const RDWR: Self = Self(0x0003);
    /// Create file if it doesn't exist.
    pub const CREATE: Self = Self(0x0100);
    /// Fail if file exists (with CREATE).
    pub const EXCL: Self = Self(0x0200);
    /// Truncate file to zero length.
    pub const TRUNC: Self = Self(0x0400);
    /// Append mode.
    pub const APPEND: Self = Self(0x0800);
    /// Directory (opendir).
    pub const DIRECTORY: Self = Self(0x1000);
    /// No follow symlinks.
    pub const NOFOLLOW: Self = Self(0x2000);

    /// Check if read access is requested.
    #[must_use]
    pub const fn can_read(&self) -> bool {
        (self.0 & 0x0001) != 0
    }

    /// Check if write access is requested.
    #[must_use]
    pub const fn can_write(&self) -> bool {
        (self.0 & 0x0002) != 0
    }

    /// Check if create flag is set.
    #[must_use]
    pub const fn create(&self) -> bool {
        (self.0 & 0x0100) != 0
    }

    /// Check if exclusive flag is set.
    #[must_use]
    pub const fn exclusive(&self) -> bool {
        (self.0 & 0x0200) != 0
    }

    /// Check if truncate flag is set.
    #[must_use]
    pub const fn truncate(&self) -> bool {
        (self.0 & 0x0400) != 0
    }

    /// Check if append flag is set.
    #[must_use]
    pub const fn append(&self) -> bool {
        (self.0 & 0x0800) != 0
    }

    /// Check if directory flag is set.
    #[must_use]
    pub const fn directory(&self) -> bool {
        (self.0 & 0x1000) != 0
    }

    /// Check if nofollow flag is set.
    #[must_use]
    pub const fn nofollow(&self) -> bool {
        (self.0 & 0x2000) != 0
    }

    /// Combine two flag sets.
    #[must_use]
    pub const fn or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Get raw flag value.
    #[must_use]
    pub const fn bits(&self) -> u32 {
        self.0
    }
}

impl Default for OpenFlags {
    fn default() -> Self {
        Self::READ
    }
}

impl core::ops::BitOr for OpenFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// Seek position for file operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeekFrom {
    /// Seek from the start of the file.
    Start(u64),
    /// Seek from the end of the file.
    End(i64),
    /// Seek from the current position.
    Current(i64),
}

/// An open file handle with seek position.
#[derive(Debug)]
pub struct OpenFile {
    /// Mount ID.
    pub mount_id: MountId,
    /// Inode ID.
    pub inode_id: InodeId,
    /// Current seek position.
    pub position: u64,
    /// Open flags.
    pub flags: OpenFlags,
}

impl OpenFile {
    /// Create a new open file.
    #[must_use]
    pub const fn new(mount_id: MountId, inode_id: InodeId, flags: OpenFlags) -> Self {
        Self {
            mount_id,
            inode_id,
            position: 0,
            flags,
        }
    }

    /// Seek to a new position.
    pub fn seek(&mut self, pos: SeekFrom, file_size: u64) -> FsResult<u64> {
        let new_pos = match pos {
            SeekFrom::Start(offset) => offset,
            SeekFrom::End(offset) => {
                if offset >= 0 {
                    file_size.saturating_add(offset as u64)
                } else {
                    file_size.saturating_sub((-offset) as u64)
                }
            }
            SeekFrom::Current(offset) => {
                if offset >= 0 {
                    self.position.saturating_add(offset as u64)
                } else {
                    self.position.saturating_sub((-offset) as u64)
                }
            }
        };

        self.position = new_pos;
        Ok(new_pos)
    }
}

/// Inode operations trait.
///
/// This trait defines the operations that can be performed on filesystem inodes
/// (files, directories, symlinks, etc.).
pub trait InodeOps {
    /// Get the inode ID.
    fn inode_id(&self) -> InodeId;

    /// Get the file type.
    fn file_type(&self) -> FileType;

    /// Get the file size in bytes.
    fn size(&self) -> u64;

    /// Get the file mode (permissions + type).
    fn mode(&self) -> u32 {
        self.file_type().to_mode() | 0o644
    }

    /// Get the number of hard links.
    fn nlink(&self) -> u32 {
        1
    }

    /// Read data from the file.
    ///
    /// # Arguments
    ///
    /// * `offset` - Byte offset to read from
    /// * `buf` - Buffer to read into
    ///
    /// # Returns
    ///
    /// Number of bytes read, or error.
    fn read(&self, offset: u64, buf: &mut [u8]) -> FsResult<usize>;

    /// Write data to the file.
    ///
    /// # Arguments
    ///
    /// * `offset` - Byte offset to write to
    /// * `buf` - Data to write
    ///
    /// # Returns
    ///
    /// Number of bytes written, or error.
    fn write(&self, offset: u64, buf: &[u8]) -> FsResult<usize>;

    /// Truncate the file to the specified size.
    fn truncate(&self, size: u64) -> FsResult<()>;

    /// Lookup a child by name (for directories).
    fn lookup(&self, name: &str) -> FsResult<InodeId>;

    /// Create a new child inode (for directories).
    fn create(&self, name: &str, file_type: FileType, mode: u32) -> FsResult<InodeId>;

    /// Remove a child by name (for directories).
    fn unlink(&self, name: &str) -> FsResult<()>;

    /// Link an existing inode as a child (for directories).
    fn link(&self, name: &str, inode_id: InodeId) -> FsResult<()>;

    /// Read directory entries.
    ///
    /// # Arguments
    ///
    /// * `offset` - Entry offset to start from
    /// * `entries` - Buffer to store entries
    ///
    /// # Returns
    ///
    /// Number of entries read, or error.
    #[cfg(feature = "alloc")]
    fn readdir(&self, offset: usize, entries: &mut Vec<DirEntry>) -> FsResult<usize>;

    /// Read the target of a symbolic link.
    #[cfg(feature = "alloc")]
    fn readlink(&self) -> FsResult<String>;

    /// Create a symbolic link.
    #[cfg(feature = "alloc")]
    fn symlink(&self, name: &str, target: &str) -> FsResult<InodeId>;

    /// Rename a child entry.
    fn rename(&self, old_name: &str, new_dir: InodeId, new_name: &str) -> FsResult<()>;

    /// Sync file data to storage.
    fn sync(&self) -> FsResult<()> {
        Ok(())
    }
}

/// A generic inode wrapper that implements common operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inode {
    /// Inode ID.
    pub id: InodeId,
    /// File type.
    pub file_type: FileType,
    /// File size.
    pub size: u64,
    /// File mode (permissions).
    pub mode: u32,
    /// Number of hard links.
    pub nlink: u32,
    /// User ID.
    pub uid: u32,
    /// Group ID.
    pub gid: u32,
    /// Access time (Unix timestamp).
    pub atime: u64,
    /// Modification time (Unix timestamp).
    pub mtime: u64,
    /// Change time (Unix timestamp).
    pub ctime: u64,
    /// Block size.
    pub blksize: u32,
    /// Number of blocks.
    pub blocks: u64,
}

impl Inode {
    /// Create a new inode with default values.
    #[must_use]
    pub const fn new(id: InodeId, file_type: FileType) -> Self {
        Self {
            id,
            file_type,
            size: 0,
            mode: 0o644,
            nlink: 1,
            uid: 0,
            gid: 0,
            atime: 0,
            mtime: 0,
            ctime: 0,
            blksize: 4096,
            blocks: 0,
        }
    }

    /// Create a directory inode.
    #[must_use]
    pub const fn directory(id: InodeId) -> Self {
        let mut inode = Self::new(id, FileType::Directory);
        inode.mode = 0o755;
        inode.nlink = 2; // . and parent
        inode
    }

    /// Create a regular file inode.
    #[must_use]
    pub const fn file(id: InodeId) -> Self {
        Self::new(id, FileType::Regular)
    }

    /// Check if this is a directory.
    #[must_use]
    pub const fn is_dir(&self) -> bool {
        matches!(self.file_type, FileType::Directory)
    }

    /// Check if this is a regular file.
    #[must_use]
    pub const fn is_file(&self) -> bool {
        matches!(self.file_type, FileType::Regular)
    }

    /// Check if this is a symlink.
    #[must_use]
    pub const fn is_symlink(&self) -> bool {
        matches!(self.file_type, FileType::Symlink)
    }
}

impl Default for Inode {
    fn default() -> Self {
        Self::new(InodeId::INVALID, FileType::Unknown)
    }
}

/// Filesystem trait.
///
/// This trait defines the operations that must be implemented by any filesystem.
pub trait FileSystem {
    /// Mount the filesystem.
    fn mount(&mut self) -> FsResult<()>;

    /// Unmount the filesystem.
    fn unmount(&mut self) -> FsResult<()>;

    /// Get the root inode.
    fn root(&self) -> FsResult<InodeId>;

    /// Get the filesystem name.
    fn name(&self) -> &str;

    /// Check if the filesystem is read-only.
    fn is_read_only(&self) -> bool;

    /// Get the total size in bytes.
    fn total_size(&self) -> u64;

    /// Get the free space in bytes.
    fn free_space(&self) -> u64;

    /// Get the block size.
    fn block_size(&self) -> u32;

    /// Sync all pending writes.
    fn sync(&self) -> FsResult<()>;

    /// Lookup an inode by path from the root.
    fn lookup_path(&self, path: &Path) -> FsResult<InodeId>;

    /// Get inode information.
    fn stat(&self, inode_id: InodeId) -> FsResult<Inode>;

    /// Read from a file.
    fn read(&self, inode_id: InodeId, offset: u64, buf: &mut [u8]) -> FsResult<usize>;

    /// Write to a file.
    fn write(&self, inode_id: InodeId, offset: u64, buf: &[u8]) -> FsResult<usize>;

    /// Truncate a file.
    fn truncate(&self, inode_id: InodeId, size: u64) -> FsResult<()>;

    /// Lookup a child in a directory.
    fn lookup(&self, dir_inode: InodeId, name: &str) -> FsResult<InodeId>;

    /// Create a new file or directory.
    fn create(&self, dir_inode: InodeId, name: &str, file_type: FileType, mode: u32)
        -> FsResult<InodeId>;

    /// Remove a file or empty directory.
    fn unlink(&self, dir_inode: InodeId, name: &str) -> FsResult<()>;

    /// Read directory entries.
    #[cfg(feature = "alloc")]
    fn readdir(&self, dir_inode: InodeId, offset: usize) -> FsResult<Vec<DirEntry>>;
}

/// A mount point in the VFS.
#[cfg(feature = "alloc")]
#[derive(Debug)]
pub struct VfsMountPoint {
    /// Mount ID.
    pub id: MountId,
    /// Mount path.
    pub path: crate::path::PathBuf,
    /// Is this mount point active?
    pub active: bool,
    /// Device name or description.
    pub device: String,
    /// Filesystem type name.
    pub fs_type: String,
    /// Mount options.
    pub options: String,
}

#[cfg(feature = "alloc")]
impl VfsMountPoint {
    /// Create a new mount point.
    #[must_use]
    pub fn new(id: MountId, path: &str, device: &str, fs_type: &str) -> Self {
        Self {
            id,
            path: crate::path::PathBuf::from(path),
            active: true,
            device: String::from(device),
            fs_type: String::from(fs_type),
            options: String::new(),
        }
    }

    /// Check if a path is under this mount point.
    #[must_use]
    pub fn contains_path(&self, path: &Path) -> bool {
        path.starts_with(self.path.as_path())
    }

    /// Get the relative path from this mount point.
    #[must_use]
    pub fn relative_path<'a>(&self, path: &'a Path) -> Option<&'a Path> {
        path.strip_prefix(self.path.as_path())
    }
}

/// VFS mount table managing all mount points.
#[cfg(feature = "alloc")]
#[derive(Debug, Default)]
pub struct VfsMountTable {
    mounts: Vec<VfsMountPoint>,
    next_id: u32,
}

#[cfg(feature = "alloc")]
impl VfsMountTable {
    /// Create a new empty mount table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mounts: Vec::new(),
            next_id: 0,
        }
    }

    /// Add a mount point.
    pub fn mount(&mut self, path: &str, device: &str, fs_type: &str) -> FsResult<MountId> {
        if self.mounts.len() >= MAX_MOUNTS {
            return Err(FsError::TooManyMounts);
        }

        // Check for existing mount at this path
        if self.find_exact(path).is_some() {
            return Err(FsError::MountAlreadyExists);
        }

        let id = MountId(self.next_id);
        self.next_id += 1;

        let mount = VfsMountPoint::new(id, path, device, fs_type);
        self.mounts.push(mount);

        Ok(id)
    }

    /// Remove a mount point by ID.
    pub fn unmount(&mut self, id: MountId) -> FsResult<()> {
        if let Some(pos) = self.mounts.iter().position(|m| m.id == id) {
            self.mounts.remove(pos);
            Ok(())
        } else {
            Err(FsError::MountNotFound)
        }
    }

    /// Find the mount point for a path.
    #[must_use]
    pub fn find(&self, path: &Path) -> Option<&VfsMountPoint> {
        self.mounts
            .iter()
            .filter(|m| m.active && m.contains_path(path))
            .max_by_key(|m| m.path.len())
    }

    /// Find an exact mount point match.
    #[must_use]
    pub fn find_exact(&self, path: &str) -> Option<&VfsMountPoint> {
        self.mounts
            .iter()
            .find(|m| m.active && m.path.as_str() == path)
    }

    /// Get a mount point by ID.
    #[must_use]
    pub fn get(&self, id: MountId) -> Option<&VfsMountPoint> {
        self.mounts.iter().find(|m| m.id == id)
    }

    /// List all active mount points.
    #[must_use]
    pub fn list(&self) -> &[VfsMountPoint] {
        &self.mounts
    }

    /// Get the number of active mounts.
    #[must_use]
    pub fn count(&self) -> usize {
        self.mounts.iter().filter(|m| m.active).count()
    }
}

/// Open file table for managing open file descriptors.
#[derive(Debug)]
pub struct OpenFileTable {
    files: [Option<OpenFile>; MAX_OPEN_FILES],
    count: usize,
}

impl OpenFileTable {
    /// Create a new open file table.
    #[must_use]
    pub const fn new() -> Self {
        const NONE: Option<OpenFile> = None;
        Self {
            files: [NONE; MAX_OPEN_FILES],
            count: 0,
        }
    }

    /// Allocate a new file descriptor.
    pub fn alloc(&mut self, file: OpenFile) -> FsResult<usize> {
        for (fd, slot) in self.files.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(file);
                self.count += 1;
                return Ok(fd);
            }
        }
        Err(FsError::TooManyOpenFiles)
    }

    /// Get a reference to an open file.
    #[must_use]
    pub fn get(&self, fd: usize) -> Option<&OpenFile> {
        self.files.get(fd).and_then(|f| f.as_ref())
    }

    /// Get a mutable reference to an open file.
    pub fn get_mut(&mut self, fd: usize) -> Option<&mut OpenFile> {
        self.files.get_mut(fd).and_then(|f| f.as_mut())
    }

    /// Close a file descriptor.
    pub fn close(&mut self, fd: usize) -> FsResult<()> {
        if fd >= MAX_OPEN_FILES {
            return Err(FsError::InvalidFileDescriptor);
        }

        if self.files[fd].take().is_some() {
            self.count -= 1;
            Ok(())
        } else {
            Err(FsError::InvalidFileDescriptor)
        }
    }

    /// Get the number of open files.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }

    /// Check if a file descriptor is valid.
    #[must_use]
    pub fn is_valid(&self, fd: usize) -> bool {
        fd < MAX_OPEN_FILES && self.files[fd].is_some()
    }
}

impl Default for OpenFileTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inode_id() {
        let id = InodeId(42);
        assert!(id.is_valid());
        assert!(!InodeId::INVALID.is_valid());
        assert!(InodeId::ROOT.is_valid());
        assert_eq!(InodeId::ROOT.0, 1);
    }

    #[test]
    fn test_file_type_mode() {
        assert_eq!(FileType::Regular.to_mode(), 0o100000);
        assert_eq!(FileType::Directory.to_mode(), 0o040000);
        assert_eq!(FileType::Symlink.to_mode(), 0o120000);

        assert_eq!(FileType::from_mode(0o100644), FileType::Regular);
        assert_eq!(FileType::from_mode(0o040755), FileType::Directory);
    }

    #[test]
    fn test_dir_entry() {
        let entry = DirEntry::new("test.txt", InodeId(5), FileType::Regular);
        assert_eq!(entry.name(), "test.txt");
        assert_eq!(entry.inode_id, InodeId(5));
        assert_eq!(entry.file_type, FileType::Regular);
    }

    #[test]
    fn test_open_flags() {
        let flags = OpenFlags::READ | OpenFlags::WRITE;
        assert!(flags.can_read());
        assert!(flags.can_write());

        let flags = OpenFlags::READ | OpenFlags::CREATE;
        assert!(flags.can_read());
        assert!(!flags.can_write());
        assert!(flags.create());
    }

    #[test]
    fn test_open_file_seek() {
        let mut file = OpenFile::new(MountId(0), InodeId(1), OpenFlags::READ);

        // Seek from start
        assert_eq!(file.seek(SeekFrom::Start(100), 1000).unwrap(), 100);
        assert_eq!(file.position, 100);

        // Seek from current
        assert_eq!(file.seek(SeekFrom::Current(50), 1000).unwrap(), 150);
        assert_eq!(file.position, 150);

        // Seek from end
        assert_eq!(file.seek(SeekFrom::End(-100), 1000).unwrap(), 900);
        assert_eq!(file.position, 900);
    }

    #[test]
    fn test_inode() {
        let inode = Inode::directory(InodeId(1));
        assert!(inode.is_dir());
        assert!(!inode.is_file());
        assert_eq!(inode.mode, 0o755);
        assert_eq!(inode.nlink, 2);

        let inode = Inode::file(InodeId(2));
        assert!(!inode.is_dir());
        assert!(inode.is_file());
        assert_eq!(inode.mode, 0o644);
        assert_eq!(inode.nlink, 1);
    }

    #[test]
    fn test_open_file_table() {
        let mut table = OpenFileTable::new();

        let fd1 = table.alloc(OpenFile::new(MountId(0), InodeId(1), OpenFlags::READ)).unwrap();
        let fd2 = table.alloc(OpenFile::new(MountId(0), InodeId(2), OpenFlags::WRITE)).unwrap();

        assert_eq!(fd1, 0);
        assert_eq!(fd2, 1);
        assert_eq!(table.count(), 2);

        assert!(table.is_valid(fd1));
        assert!(table.is_valid(fd2));

        table.close(fd1).unwrap();
        assert!(!table.is_valid(fd1));
        assert_eq!(table.count(), 1);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_mount_table() {
        let mut table = VfsMountTable::new();

        let id1 = table.mount("/", "rootfs", "ramfs").unwrap();
        let id2 = table.mount("/mnt/data", "disk0", "fat32").unwrap();

        assert_eq!(table.count(), 2);

        let mount = table.find(Path::new("/mnt/data/file.txt")).unwrap();
        assert_eq!(mount.id, id2);

        let mount = table.find(Path::new("/etc/passwd")).unwrap();
        assert_eq!(mount.id, id1);

        table.unmount(id1).unwrap();
        assert_eq!(table.count(), 1);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_mount_duplicate() {
        let mut table = VfsMountTable::new();
        table.mount("/mnt/data", "disk0", "fat32").unwrap();
        assert_eq!(
            table.mount("/mnt/data", "disk1", "fat32"),
            Err(FsError::MountAlreadyExists)
        );
    }
}
