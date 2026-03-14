//! In-memory filesystem (RamFS) implementation for RuVix.
//!
//! This module provides a fully read-write in-memory filesystem suitable
//! for `/tmp` and other temporary storage needs. All data is lost on unmount.

use crate::error::{FsError, FsResult};
use crate::path::{Path, PathComponent};
use crate::vfs::{DirEntry, FileSystem, FileType, Inode, InodeId, InodeOps};
use crate::MAX_NAME_LEN;
use core::cell::RefCell;

#[cfg(feature = "alloc")]
use alloc::{collections::BTreeMap, string::String, vec::Vec};

/// Maximum file size in RamFS (64 MB default).
const MAX_FILE_SIZE: u64 = 64 * 1024 * 1024;

/// Maximum number of inodes in RamFS.
const MAX_INODES: usize = 65536;

/// Type of RamFS inode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RamInodeType {
    /// Regular file.
    File,
    /// Directory.
    Directory,
    /// Symbolic link.
    Symlink,
}

impl From<RamInodeType> for FileType {
    fn from(t: RamInodeType) -> Self {
        match t {
            RamInodeType::File => FileType::Regular,
            RamInodeType::Directory => FileType::Directory,
            RamInodeType::Symlink => FileType::Symlink,
        }
    }
}

impl From<FileType> for RamInodeType {
    fn from(t: FileType) -> Self {
        match t {
            FileType::Regular => RamInodeType::File,
            FileType::Directory => RamInodeType::Directory,
            FileType::Symlink => RamInodeType::Symlink,
            _ => RamInodeType::File,
        }
    }
}

/// A RamFS inode representing a file, directory, or symlink.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct RamInode {
    /// Inode ID.
    pub id: InodeId,
    /// Inode type.
    pub inode_type: RamInodeType,
    /// File mode (permissions).
    pub mode: u32,
    /// Number of hard links.
    pub nlink: u32,
    /// User ID.
    pub uid: u32,
    /// Group ID.
    pub gid: u32,
    /// File data (for files) or symlink target (for symlinks).
    pub data: Vec<u8>,
    /// Directory entries (for directories).
    pub children: BTreeMap<String, InodeId>,
    /// Parent inode ID (for directories).
    pub parent: InodeId,
    /// Access time.
    pub atime: u64,
    /// Modification time.
    pub mtime: u64,
    /// Change time.
    pub ctime: u64,
}

#[cfg(feature = "alloc")]
impl RamInode {
    /// Create a new file inode.
    #[must_use]
    pub fn new_file(id: InodeId, mode: u32) -> Self {
        Self {
            id,
            inode_type: RamInodeType::File,
            mode: mode & 0o7777,
            nlink: 1,
            uid: 0,
            gid: 0,
            data: Vec::new(),
            children: BTreeMap::new(),
            parent: InodeId::INVALID,
            atime: 0,
            mtime: 0,
            ctime: 0,
        }
    }

    /// Create a new directory inode.
    #[must_use]
    pub fn new_directory(id: InodeId, parent: InodeId, mode: u32) -> Self {
        let mut dir = Self {
            id,
            inode_type: RamInodeType::Directory,
            mode: mode & 0o7777,
            nlink: 2, // . and parent link
            uid: 0,
            gid: 0,
            data: Vec::new(),
            children: BTreeMap::new(),
            parent,
            atime: 0,
            mtime: 0,
            ctime: 0,
        };
        // Add . and .. entries conceptually (not stored, computed on read)
        dir.children.insert(String::from("."), id);
        if parent.is_valid() {
            dir.children.insert(String::from(".."), parent);
        } else {
            dir.children.insert(String::from(".."), id);
        }
        dir
    }

    /// Create a new symlink inode.
    #[must_use]
    pub fn new_symlink(id: InodeId, target: &str) -> Self {
        Self {
            id,
            inode_type: RamInodeType::Symlink,
            mode: 0o777,
            nlink: 1,
            uid: 0,
            gid: 0,
            data: target.as_bytes().to_vec(),
            children: BTreeMap::new(),
            parent: InodeId::INVALID,
            atime: 0,
            mtime: 0,
            ctime: 0,
        }
    }

    /// Get the size of the inode.
    #[must_use]
    pub fn size(&self) -> u64 {
        match self.inode_type {
            RamInodeType::File | RamInodeType::Symlink => self.data.len() as u64,
            RamInodeType::Directory => {
                (self.children.len() * core::mem::size_of::<DirEntry>()) as u64
            }
        }
    }

    /// Check if this is a directory.
    #[must_use]
    pub const fn is_dir(&self) -> bool {
        matches!(self.inode_type, RamInodeType::Directory)
    }

    /// Check if this is a file.
    #[must_use]
    pub const fn is_file(&self) -> bool {
        matches!(self.inode_type, RamInodeType::File)
    }

    /// Check if this is a symlink.
    #[must_use]
    pub const fn is_symlink(&self) -> bool {
        matches!(self.inode_type, RamInodeType::Symlink)
    }

    /// Read data from the inode.
    pub fn read(&self, offset: u64, buf: &mut [u8]) -> FsResult<usize> {
        if self.inode_type == RamInodeType::Directory {
            return Err(FsError::IsADirectory);
        }

        if offset >= self.data.len() as u64 {
            return Ok(0);
        }

        let start = offset as usize;
        let end = (start + buf.len()).min(self.data.len());
        let bytes = end - start;

        buf[..bytes].copy_from_slice(&self.data[start..end]);
        Ok(bytes)
    }

    /// Write data to the inode.
    pub fn write(&mut self, offset: u64, buf: &[u8]) -> FsResult<usize> {
        if self.inode_type == RamInodeType::Directory {
            return Err(FsError::IsADirectory);
        }

        let new_size = offset + buf.len() as u64;
        if new_size > MAX_FILE_SIZE {
            return Err(FsError::FileTooLarge);
        }

        // Extend if necessary
        if new_size > self.data.len() as u64 {
            self.data.resize(new_size as usize, 0);
        }

        let start = offset as usize;
        self.data[start..start + buf.len()].copy_from_slice(buf);

        Ok(buf.len())
    }

    /// Truncate the file to the specified size.
    pub fn truncate(&mut self, size: u64) -> FsResult<()> {
        if self.inode_type == RamInodeType::Directory {
            return Err(FsError::IsADirectory);
        }

        if size > MAX_FILE_SIZE {
            return Err(FsError::FileTooLarge);
        }

        self.data.resize(size as usize, 0);
        Ok(())
    }

    /// Get the symlink target.
    #[must_use]
    pub fn symlink_target(&self) -> Option<&str> {
        if self.inode_type == RamInodeType::Symlink {
            core::str::from_utf8(&self.data).ok()
        } else {
            None
        }
    }
}

#[cfg(feature = "alloc")]
impl InodeOps for RamInode {
    fn inode_id(&self) -> InodeId {
        self.id
    }

    fn file_type(&self) -> FileType {
        self.inode_type.into()
    }

    fn size(&self) -> u64 {
        RamInode::size(self)
    }

    fn mode(&self) -> u32 {
        FileType::from(self.inode_type).to_mode() | self.mode
    }

    fn nlink(&self) -> u32 {
        self.nlink
    }

    fn read(&self, offset: u64, buf: &mut [u8]) -> FsResult<usize> {
        RamInode::read(self, offset, buf)
    }

    fn write(&self, _offset: u64, _buf: &[u8]) -> FsResult<usize> {
        // InodeOps requires &self, but write needs &mut self
        // This is handled at the filesystem level
        Err(FsError::NotSupported)
    }

    fn truncate(&self, _size: u64) -> FsResult<()> {
        Err(FsError::NotSupported)
    }

    fn lookup(&self, name: &str) -> FsResult<InodeId> {
        if self.inode_type != RamInodeType::Directory {
            return Err(FsError::NotADirectory);
        }

        self.children.get(name).copied().ok_or(FsError::NotFound)
    }

    fn create(&self, _name: &str, _file_type: FileType, _mode: u32) -> FsResult<InodeId> {
        Err(FsError::NotSupported)
    }

    fn unlink(&self, _name: &str) -> FsResult<()> {
        Err(FsError::NotSupported)
    }

    fn link(&self, _name: &str, _inode_id: InodeId) -> FsResult<()> {
        Err(FsError::NotSupported)
    }

    fn readdir(&self, offset: usize, entries: &mut Vec<DirEntry>) -> FsResult<usize> {
        if self.inode_type != RamInodeType::Directory {
            return Err(FsError::NotADirectory);
        }

        let mut count = 0;
        for (i, (name, &inode_id)) in self.children.iter().enumerate() {
            if i < offset {
                continue;
            }

            // Determine file type (we'd need access to the inode table for accuracy)
            let file_type = if name == "." || name == ".." {
                FileType::Directory
            } else {
                FileType::Unknown
            };

            entries.push(DirEntry::new(name, inode_id, file_type));
            count += 1;
        }

        Ok(count)
    }

    fn readlink(&self) -> FsResult<String> {
        self.symlink_target()
            .map(String::from)
            .ok_or(FsError::InvalidArgument)
    }

    fn symlink(&self, _name: &str, _target: &str) -> FsResult<InodeId> {
        Err(FsError::NotSupported)
    }

    fn rename(&self, _old_name: &str, _new_dir: InodeId, _new_name: &str) -> FsResult<()> {
        Err(FsError::NotSupported)
    }
}

/// In-memory filesystem.
#[cfg(feature = "alloc")]
pub struct RamFs {
    /// Inode table.
    inodes: RefCell<BTreeMap<InodeId, RamInode>>,
    /// Next inode ID.
    next_inode: RefCell<u64>,
    /// Root inode ID.
    root_inode: InodeId,
    /// Is the filesystem mounted?
    mounted: RefCell<bool>,
    /// Total bytes used.
    bytes_used: RefCell<u64>,
    /// Maximum size.
    max_size: u64,
}

#[cfg(feature = "alloc")]
impl RamFs {
    /// Create a new RamFS with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::with_max_size(64 * 1024 * 1024) // 64 MB default
    }

    /// Create a new RamFS with a specified maximum size.
    #[must_use]
    pub fn with_max_size(max_size: u64) -> Self {
        let root_inode = InodeId(1);
        let mut inodes = BTreeMap::new();

        // Create root directory
        let root = RamInode::new_directory(root_inode, InodeId::INVALID, 0o755);
        inodes.insert(root_inode, root);

        Self {
            inodes: RefCell::new(inodes),
            next_inode: RefCell::new(2),
            root_inode,
            mounted: RefCell::new(false),
            bytes_used: RefCell::new(0),
            max_size,
        }
    }

    /// Get the inode for a given ID.
    fn get_inode(&self, id: InodeId) -> FsResult<RamInode> {
        self.inodes.borrow().get(&id).cloned().ok_or(FsError::InodeNotFound)
    }

    /// Get a mutable reference to an inode.
    fn with_inode_mut<F, R>(&self, id: InodeId, f: F) -> FsResult<R>
    where
        F: FnOnce(&mut RamInode) -> FsResult<R>,
    {
        let mut inodes = self.inodes.borrow_mut();
        let inode = inodes.get_mut(&id).ok_or(FsError::InodeNotFound)?;
        f(inode)
    }

    /// Allocate a new inode ID.
    fn alloc_inode(&self) -> FsResult<InodeId> {
        let mut next = self.next_inode.borrow_mut();
        if *next as usize >= MAX_INODES {
            return Err(FsError::OutOfInodes);
        }
        let id = InodeId(*next);
        *next += 1;
        Ok(id)
    }

    /// Insert an inode into the table.
    fn insert_inode(&self, inode: RamInode) {
        let size = inode.size();
        self.inodes.borrow_mut().insert(inode.id, inode);
        *self.bytes_used.borrow_mut() += size;
    }

    /// Remove an inode from the table.
    fn remove_inode(&self, id: InodeId) -> FsResult<RamInode> {
        let inode = self.inodes.borrow_mut().remove(&id).ok_or(FsError::InodeNotFound)?;
        let size = inode.size();
        let current_used = *self.bytes_used.borrow();
        *self.bytes_used.borrow_mut() = current_used.saturating_sub(size);
        Ok(inode)
    }

    /// Get the number of inodes.
    #[must_use]
    pub fn inode_count(&self) -> usize {
        self.inodes.borrow().len()
    }

    /// Get the bytes used.
    #[must_use]
    pub fn bytes_used(&self) -> u64 {
        *self.bytes_used.borrow()
    }
}

#[cfg(feature = "alloc")]
impl Default for RamFs {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "alloc")]
impl FileSystem for RamFs {
    fn mount(&mut self) -> FsResult<()> {
        let mut mounted = self.mounted.borrow_mut();
        if *mounted {
            return Err(FsError::Busy);
        }
        *mounted = true;
        Ok(())
    }

    fn unmount(&mut self) -> FsResult<()> {
        let mut mounted = self.mounted.borrow_mut();
        if !*mounted {
            return Err(FsError::NotSupported);
        }
        *mounted = false;
        Ok(())
    }

    fn root(&self) -> FsResult<InodeId> {
        Ok(self.root_inode)
    }

    fn name(&self) -> &str {
        "ramfs"
    }

    fn is_read_only(&self) -> bool {
        false
    }

    fn total_size(&self) -> u64 {
        self.max_size
    }

    fn free_space(&self) -> u64 {
        self.max_size.saturating_sub(*self.bytes_used.borrow())
    }

    fn block_size(&self) -> u32 {
        4096
    }

    fn sync(&self) -> FsResult<()> {
        Ok(()) // RamFS is always in sync
    }

    fn lookup_path(&self, path: &Path) -> FsResult<InodeId> {
        if !*self.mounted.borrow() {
            return Err(FsError::NotSupported);
        }

        let mut current = self.root_inode;

        for component in path.components() {
            match component {
                PathComponent::RootDir => {
                    current = self.root_inode;
                }
                PathComponent::CurDir => {}
                PathComponent::ParentDir => {
                    let inode = self.get_inode(current)?;
                    if inode.parent.is_valid() {
                        current = inode.parent;
                    }
                    // At root, parent is self
                }
                PathComponent::Normal(name) => {
                    let inode = self.get_inode(current)?;
                    if !inode.is_dir() {
                        return Err(FsError::NotADirectory);
                    }
                    current = inode.children.get(name).copied().ok_or(FsError::NotFound)?;
                }
            }
        }

        Ok(current)
    }

    fn stat(&self, inode_id: InodeId) -> FsResult<Inode> {
        if !*self.mounted.borrow() {
            return Err(FsError::NotSupported);
        }

        let ram_inode = self.get_inode(inode_id)?;

        Ok(Inode {
            id: ram_inode.id,
            file_type: ram_inode.inode_type.into(),
            size: ram_inode.size(),
            mode: ram_inode.mode,
            nlink: ram_inode.nlink,
            uid: ram_inode.uid,
            gid: ram_inode.gid,
            atime: ram_inode.atime,
            mtime: ram_inode.mtime,
            ctime: ram_inode.ctime,
            blksize: 4096,
            blocks: (ram_inode.size() + 511) / 512,
        })
    }

    fn read(&self, inode_id: InodeId, offset: u64, buf: &mut [u8]) -> FsResult<usize> {
        if !*self.mounted.borrow() {
            return Err(FsError::NotSupported);
        }

        let inode = self.get_inode(inode_id)?;
        inode.read(offset, buf)
    }

    fn write(&self, inode_id: InodeId, offset: u64, buf: &[u8]) -> FsResult<usize> {
        if !*self.mounted.borrow() {
            return Err(FsError::NotSupported);
        }

        // Check space
        let additional = buf.len() as u64;
        let used = *self.bytes_used.borrow();
        if used + additional > self.max_size {
            return Err(FsError::NoSpace);
        }

        self.with_inode_mut(inode_id, |inode| {
            let old_size = inode.size();
            let result = inode.write(offset, buf)?;
            let new_size = inode.size();
            Ok((result, new_size as i64 - old_size as i64))
        })
        .map(|(result, delta)| {
            if delta > 0 {
                *self.bytes_used.borrow_mut() += delta as u64;
            } else {
                let current = *self.bytes_used.borrow();
                *self.bytes_used.borrow_mut() = current.saturating_sub((-delta) as u64);
            }
            result
        })
    }

    fn truncate(&self, inode_id: InodeId, size: u64) -> FsResult<()> {
        if !*self.mounted.borrow() {
            return Err(FsError::NotSupported);
        }

        self.with_inode_mut(inode_id, |inode| {
            let old_size = inode.size();
            inode.truncate(size)?;
            let new_size = inode.size();
            Ok((old_size, new_size))
        })
        .map(|(old_size, new_size)| {
            if new_size > old_size {
                *self.bytes_used.borrow_mut() += new_size - old_size;
            } else {
                let current = *self.bytes_used.borrow();
                *self.bytes_used.borrow_mut() = current.saturating_sub(old_size - new_size);
            }
        })
    }

    fn lookup(&self, dir_inode: InodeId, name: &str) -> FsResult<InodeId> {
        if !*self.mounted.borrow() {
            return Err(FsError::NotSupported);
        }

        if name.len() > MAX_NAME_LEN {
            return Err(FsError::NameTooLong);
        }

        let inode = self.get_inode(dir_inode)?;
        if !inode.is_dir() {
            return Err(FsError::NotADirectory);
        }

        inode.children.get(name).copied().ok_or(FsError::NotFound)
    }

    fn create(
        &self,
        dir_inode: InodeId,
        name: &str,
        file_type: FileType,
        mode: u32,
    ) -> FsResult<InodeId> {
        if !*self.mounted.borrow() {
            return Err(FsError::NotSupported);
        }

        if name.len() > MAX_NAME_LEN {
            return Err(FsError::NameTooLong);
        }

        if name.is_empty() || name == "." || name == ".." {
            return Err(FsError::InvalidFilename);
        }

        // Check if name already exists
        {
            let parent = self.get_inode(dir_inode)?;
            if !parent.is_dir() {
                return Err(FsError::NotADirectory);
            }
            if parent.children.contains_key(name) {
                return Err(FsError::AlreadyExists);
            }
        }

        // Allocate new inode
        let new_id = self.alloc_inode()?;

        // Create the inode
        let new_inode = match file_type {
            FileType::Regular => RamInode::new_file(new_id, mode),
            FileType::Directory => RamInode::new_directory(new_id, dir_inode, mode),
            _ => return Err(FsError::NotSupported),
        };

        self.insert_inode(new_inode);

        // Add to parent's children
        self.with_inode_mut(dir_inode, |parent| {
            parent.children.insert(String::from(name), new_id);
            if file_type == FileType::Directory {
                parent.nlink += 1; // Subdirectory's .. points to us
            }
            Ok(())
        })?;

        Ok(new_id)
    }

    fn unlink(&self, dir_inode: InodeId, name: &str) -> FsResult<()> {
        if !*self.mounted.borrow() {
            return Err(FsError::NotSupported);
        }

        if name == "." || name == ".." {
            return Err(FsError::InvalidArgument);
        }

        // Get the target inode ID
        let target_id = self.lookup(dir_inode, name)?;
        let target = self.get_inode(target_id)?;

        // Check if directory is empty
        if target.is_dir() {
            // Only . and .. should exist
            if target.children.len() > 2 {
                return Err(FsError::DirectoryNotEmpty);
            }
        }

        // Remove from parent
        self.with_inode_mut(dir_inode, |parent| {
            parent.children.remove(name);
            if target.is_dir() {
                parent.nlink = parent.nlink.saturating_sub(1);
            }
            Ok(())
        })?;

        // Decrement nlink and potentially remove inode
        // For directories, we need to account for both the parent reference and self (.)
        let should_remove = self.with_inode_mut(target_id, |inode| {
            if inode.is_dir() {
                // Directory: decrement by 2 (parent's link and self's "." entry)
                inode.nlink = inode.nlink.saturating_sub(2);
            } else {
                // File/symlink: just one link
                inode.nlink = inode.nlink.saturating_sub(1);
            }
            Ok(inode.nlink == 0)
        })?;

        if should_remove {
            self.remove_inode(target_id)?;
        }

        Ok(())
    }

    fn readdir(&self, dir_inode: InodeId, offset: usize) -> FsResult<Vec<DirEntry>> {
        if !*self.mounted.borrow() {
            return Err(FsError::NotSupported);
        }

        let inode = self.get_inode(dir_inode)?;
        if !inode.is_dir() {
            return Err(FsError::NotADirectory);
        }

        let mut entries = Vec::new();

        for (i, (name, &child_id)) in inode.children.iter().enumerate() {
            if i < offset {
                continue;
            }

            let child = self.get_inode(child_id)?;
            entries.push(DirEntry::new(name, child_id, child.inode_type.into()));
        }

        Ok(entries)
    }
}

#[cfg(test)]
#[cfg(feature = "alloc")]
mod tests {
    use super::*;

    #[test]
    fn test_ramfs_create() {
        let mut fs = RamFs::new();
        fs.mount().unwrap();

        let root = fs.root().unwrap();
        assert_eq!(root, InodeId(1));

        let stat = fs.stat(root).unwrap();
        assert_eq!(stat.file_type, FileType::Directory);
    }

    #[test]
    fn test_ramfs_create_file() {
        let mut fs = RamFs::new();
        fs.mount().unwrap();

        let root = fs.root().unwrap();
        let file_id = fs.create(root, "test.txt", FileType::Regular, 0o644).unwrap();

        let stat = fs.stat(file_id).unwrap();
        assert_eq!(stat.file_type, FileType::Regular);
        assert_eq!(stat.size, 0);
    }

    #[test]
    fn test_ramfs_write_read() {
        let mut fs = RamFs::new();
        fs.mount().unwrap();

        let root = fs.root().unwrap();
        let file_id = fs.create(root, "test.txt", FileType::Regular, 0o644).unwrap();

        // Write data
        let data = b"Hello, World!";
        let written = fs.write(file_id, 0, data).unwrap();
        assert_eq!(written, data.len());

        // Read it back
        let mut buf = [0u8; 64];
        let read = fs.read(file_id, 0, &mut buf).unwrap();
        assert_eq!(read, data.len());
        assert_eq!(&buf[..read], data);
    }

    #[test]
    fn test_ramfs_create_directory() {
        let mut fs = RamFs::new();
        fs.mount().unwrap();

        let root = fs.root().unwrap();
        let dir_id = fs.create(root, "subdir", FileType::Directory, 0o755).unwrap();

        let stat = fs.stat(dir_id).unwrap();
        assert_eq!(stat.file_type, FileType::Directory);

        // Create file in subdirectory
        let file_id = fs.create(dir_id, "nested.txt", FileType::Regular, 0o644).unwrap();
        assert!(file_id.is_valid());
    }

    #[test]
    fn test_ramfs_lookup() {
        let mut fs = RamFs::new();
        fs.mount().unwrap();

        let root = fs.root().unwrap();
        let file_id = fs.create(root, "test.txt", FileType::Regular, 0o644).unwrap();

        let found = fs.lookup(root, "test.txt").unwrap();
        assert_eq!(found, file_id);

        assert_eq!(fs.lookup(root, "nonexistent.txt"), Err(FsError::NotFound));
    }

    #[test]
    fn test_ramfs_unlink() {
        let mut fs = RamFs::new();
        fs.mount().unwrap();

        let root = fs.root().unwrap();
        fs.create(root, "test.txt", FileType::Regular, 0o644).unwrap();

        fs.unlink(root, "test.txt").unwrap();

        assert_eq!(fs.lookup(root, "test.txt"), Err(FsError::NotFound));
    }

    #[test]
    fn test_ramfs_unlink_nonempty_dir() {
        let mut fs = RamFs::new();
        fs.mount().unwrap();

        let root = fs.root().unwrap();
        let dir_id = fs.create(root, "subdir", FileType::Directory, 0o755).unwrap();
        fs.create(dir_id, "file.txt", FileType::Regular, 0o644).unwrap();

        assert_eq!(fs.unlink(root, "subdir"), Err(FsError::DirectoryNotEmpty));
    }

    #[test]
    fn test_ramfs_readdir() {
        let mut fs = RamFs::new();
        fs.mount().unwrap();

        let root = fs.root().unwrap();
        fs.create(root, "a.txt", FileType::Regular, 0o644).unwrap();
        fs.create(root, "b.txt", FileType::Regular, 0o644).unwrap();
        fs.create(root, "c_dir", FileType::Directory, 0o755).unwrap();

        let entries = fs.readdir(root, 0).unwrap();

        // Should have . and .. plus our 3 entries
        assert!(entries.len() >= 5);

        let names: Vec<_> = entries.iter().map(|e| e.name()).collect();
        assert!(names.contains(&"a.txt"));
        assert!(names.contains(&"b.txt"));
        assert!(names.contains(&"c_dir"));
    }

    #[test]
    fn test_ramfs_truncate() {
        let mut fs = RamFs::new();
        fs.mount().unwrap();

        let root = fs.root().unwrap();
        let file_id = fs.create(root, "test.txt", FileType::Regular, 0o644).unwrap();

        fs.write(file_id, 0, b"Hello, World!").unwrap();
        assert_eq!(fs.stat(file_id).unwrap().size, 13);

        fs.truncate(file_id, 5).unwrap();
        assert_eq!(fs.stat(file_id).unwrap().size, 5);

        let mut buf = [0u8; 10];
        let read = fs.read(file_id, 0, &mut buf).unwrap();
        assert_eq!(read, 5);
        assert_eq!(&buf[..5], b"Hello");
    }

    #[test]
    fn test_ramfs_lookup_path() {
        let mut fs = RamFs::new();
        fs.mount().unwrap();

        let root = fs.root().unwrap();
        let dir_id = fs.create(root, "subdir", FileType::Directory, 0o755).unwrap();
        let file_id = fs.create(dir_id, "test.txt", FileType::Regular, 0o644).unwrap();

        let found = fs.lookup_path(Path::new("/subdir/test.txt")).unwrap();
        assert_eq!(found, file_id);

        let found = fs.lookup_path(Path::new("/")).unwrap();
        assert_eq!(found, root);
    }

    #[test]
    fn test_ramfs_space_tracking() {
        let fs = RamFs::with_max_size(1024);
        assert_eq!(fs.total_size(), 1024);
        assert!(fs.free_space() <= 1024);
    }

    #[test]
    fn test_ramfs_duplicate_name() {
        let mut fs = RamFs::new();
        fs.mount().unwrap();

        let root = fs.root().unwrap();
        fs.create(root, "test.txt", FileType::Regular, 0o644).unwrap();

        assert_eq!(
            fs.create(root, "test.txt", FileType::Regular, 0o644),
            Err(FsError::AlreadyExists)
        );
    }

    #[test]
    fn test_ram_inode_symlink() {
        let inode = RamInode::new_symlink(InodeId(5), "/path/to/target");
        assert!(inode.is_symlink());
        assert_eq!(inode.symlink_target(), Some("/path/to/target"));
    }
}
