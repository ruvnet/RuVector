//! Integration tests for the RuVix filesystem layer.

#![cfg(feature = "alloc")]

use ruvix_fs::{
    BlockDevice, DirEntry, Fat32BootSector, Fat32DirEntry, FileSystem, FileType, FsError, Inode, InodeId,
    MemoryBlockDevice, MemoryBlockDeviceMut, MountId, NullBlockDevice, OpenFile, OpenFileTable, OpenFlags,
    Path, PathBuf, PathComponent, RamFs, RamInode, RamInodeType, SeekFrom, VfsMountPoint,
    VfsMountTable, MAX_NAME_LEN, MAX_PATH_LEN, ROOT_INODE_ID,
};

// ============================================================================
// Path Tests
// ============================================================================

#[test]
fn test_path_absolute_relative() {
    assert!(Path::new("/").is_absolute());
    assert!(Path::new("/foo/bar").is_absolute());
    assert!(!Path::new("foo/bar").is_absolute());
    assert!(!Path::new("").is_absolute());

    assert!(!Path::new("/").is_relative());
    assert!(Path::new("foo").is_relative());
}

#[test]
fn test_path_components_iterator() {
    let path = Path::new("/foo/bar/baz");
    let components: Vec<_> = path.components().collect();

    assert_eq!(components.len(), 4);
    assert_eq!(components[0], PathComponent::RootDir);
    assert!(matches!(components[1], PathComponent::Normal("foo")));
    assert!(matches!(components[2], PathComponent::Normal("bar")));
    assert!(matches!(components[3], PathComponent::Normal("baz")));
}

#[test]
fn test_path_components_with_dots() {
    let path = Path::new("./foo/../bar");
    let components: Vec<_> = path.components().collect();

    assert_eq!(components.len(), 4);
    assert_eq!(components[0], PathComponent::CurDir);
    assert!(matches!(components[1], PathComponent::Normal("foo")));
    assert_eq!(components[2], PathComponent::ParentDir);
    assert!(matches!(components[3], PathComponent::Normal("bar")));
}

#[test]
fn test_path_parent_and_filename() {
    let path = Path::new("/foo/bar/baz.txt");

    assert_eq!(path.parent().map(|p| p.as_str()), Some("/foo/bar"));
    assert_eq!(path.file_name(), Some("baz.txt"));

    let path = Path::new("/");
    assert!(path.parent().is_none());
    assert!(path.file_name().is_none());

    let path = Path::new("/foo");
    assert_eq!(path.parent().map(|p| p.as_str()), Some("/"));
    assert_eq!(path.file_name(), Some("foo"));
}

#[test]
fn test_path_validation() {
    assert!(Path::new("/valid/path").validate().is_ok());
    assert!(Path::new("relative/path").validate().is_ok());

    // Path too long
    let long_path = "a".repeat(MAX_PATH_LEN + 1);
    assert_eq!(Path::new(&long_path).validate(), Err(FsError::PathTooLong));
}

#[test]
fn test_pathbuf_operations() {
    let mut path = PathBuf::from("/foo");
    path.push(Path::new("bar"));
    assert_eq!(path.as_str(), "/foo/bar");

    path.push(Path::new("baz.txt"));
    assert_eq!(path.as_str(), "/foo/bar/baz.txt");

    assert!(path.pop());
    assert_eq!(path.as_str(), "/foo/bar");

    path.set_file_name("new_name");
    assert_eq!(path.as_str(), "/foo/new_name");
}

#[test]
fn test_pathbuf_join() {
    let base = Path::new("/home/user");
    let joined = base.join(Path::new("documents/file.txt"));
    assert_eq!(joined.as_str(), "/home/user/documents/file.txt");

    // Absolute path replaces base
    let joined = base.join(Path::new("/etc/passwd"));
    assert_eq!(joined.as_str(), "/etc/passwd");
}

// ============================================================================
// VFS Tests
// ============================================================================

#[test]
fn test_inode_id() {
    assert!(InodeId(1).is_valid());
    assert!(!InodeId(0).is_valid());
    assert!(InodeId::ROOT.is_valid());
    assert!(!InodeId::INVALID.is_valid());
    assert_eq!(ROOT_INODE_ID, InodeId(1));
}

#[test]
fn test_file_type_mode_conversion() {
    assert_eq!(FileType::Regular.to_mode(), 0o100000);
    assert_eq!(FileType::Directory.to_mode(), 0o040000);
    assert_eq!(FileType::Symlink.to_mode(), 0o120000);
    assert_eq!(FileType::BlockDevice.to_mode(), 0o060000);
    assert_eq!(FileType::CharDevice.to_mode(), 0o020000);

    assert_eq!(FileType::from_mode(0o100644), FileType::Regular);
    assert_eq!(FileType::from_mode(0o040755), FileType::Directory);
    assert_eq!(FileType::from_mode(0o120777), FileType::Symlink);
}

#[test]
fn test_dir_entry() {
    let entry = DirEntry::new("test_file.txt", InodeId(42), FileType::Regular);

    assert_eq!(entry.name(), "test_file.txt");
    assert_eq!(entry.inode_id, InodeId(42));
    assert_eq!(entry.file_type, FileType::Regular);
}

#[test]
fn test_open_flags() {
    let flags = OpenFlags::READ | OpenFlags::WRITE;
    assert!(flags.can_read());
    assert!(flags.can_write());
    assert!(!flags.create());

    let flags = OpenFlags::READ | OpenFlags::CREATE | OpenFlags::EXCL;
    assert!(flags.can_read());
    assert!(flags.create());
    assert!(flags.exclusive());
    assert!(!flags.truncate());
}

#[test]
fn test_open_file_seek() {
    let mut file = OpenFile::new(MountId(0), InodeId(1), OpenFlags::READ);

    // Seek from start
    assert_eq!(file.seek(SeekFrom::Start(100), 1000).unwrap(), 100);
    assert_eq!(file.position, 100);

    // Seek from current (forward)
    assert_eq!(file.seek(SeekFrom::Current(50), 1000).unwrap(), 150);
    assert_eq!(file.position, 150);

    // Seek from current (backward)
    assert_eq!(file.seek(SeekFrom::Current(-30), 1000).unwrap(), 120);
    assert_eq!(file.position, 120);

    // Seek from end
    assert_eq!(file.seek(SeekFrom::End(-100), 1000).unwrap(), 900);
    assert_eq!(file.position, 900);

    // Seek from end (forward)
    assert_eq!(file.seek(SeekFrom::End(50), 1000).unwrap(), 1050);
    assert_eq!(file.position, 1050);
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
    assert!(!table.is_valid(999));

    // Get file
    let file = table.get(fd1).unwrap();
    assert_eq!(file.inode_id, InodeId(1));

    // Modify file
    let file = table.get_mut(fd1).unwrap();
    file.position = 500;
    assert_eq!(table.get(fd1).unwrap().position, 500);

    // Close file
    table.close(fd1).unwrap();
    assert!(!table.is_valid(fd1));
    assert_eq!(table.count(), 1);

    // Double close should fail
    assert_eq!(table.close(fd1), Err(FsError::InvalidFileDescriptor));
}

#[test]
fn test_mount_table() {
    let mut table = VfsMountTable::new();

    let id1 = table.mount("/", "rootfs", "ramfs").unwrap();
    let id2 = table.mount("/mnt/data", "disk0", "fat32").unwrap();
    let id3 = table.mount("/mnt/data/sub", "disk1", "fat32").unwrap();

    assert_eq!(table.count(), 3);

    // Find mount for path
    let mount = table.find(Path::new("/mnt/data/sub/file.txt")).unwrap();
    assert_eq!(mount.id, id3); // Most specific match

    let mount = table.find(Path::new("/mnt/data/other.txt")).unwrap();
    assert_eq!(mount.id, id2);

    let mount = table.find(Path::new("/etc/passwd")).unwrap();
    assert_eq!(mount.id, id1);

    // Find exact
    assert!(table.find_exact("/mnt/data").is_some());
    assert!(table.find_exact("/mnt/nonexistent").is_none());

    // Unmount
    table.unmount(id2).unwrap();
    assert_eq!(table.count(), 2);
    assert!(table.find_exact("/mnt/data").is_none());

    // Can't unmount twice
    assert_eq!(table.unmount(id2), Err(FsError::MountNotFound));
}

#[test]
fn test_mount_duplicate() {
    let mut table = VfsMountTable::new();
    table.mount("/mnt/data", "disk0", "fat32").unwrap();
    assert_eq!(
        table.mount("/mnt/data", "disk1", "fat32"),
        Err(FsError::MountAlreadyExists)
    );
}

#[test]
fn test_mount_point_relative_path() {
    let mount = VfsMountPoint::new(MountId(0), "/mnt/data", "disk0", "fat32");

    let path = Path::new("/mnt/data/subdir/file.txt");
    let relative = mount.relative_path(path).unwrap();
    assert_eq!(relative.as_str(), "/subdir/file.txt");

    let path = Path::new("/other/path");
    assert!(mount.relative_path(path).is_none());
}

// ============================================================================
// Block Device Tests
// ============================================================================

#[test]
fn test_null_block_device() {
    let dev = NullBlockDevice::new(512, 100);

    assert_eq!(dev.block_size(), 512);
    assert_eq!(dev.block_count(), 100);
    assert!(!dev.is_read_only());

    // Read returns zeros
    let mut buf = [0xFFu8; 512];
    dev.read_block(0, &mut buf).unwrap();
    assert!(buf.iter().all(|&b| b == 0));

    // Write is accepted (discarded)
    let buf = [0xAAu8; 512];
    dev.write_block(0, &buf).unwrap();

    // Invalid LBA
    let mut buf = [0u8; 512];
    assert_eq!(dev.read_block(100, &mut buf), Err(FsError::BlockDeviceError));
    assert_eq!(dev.read_block(101, &mut buf), Err(FsError::BlockDeviceError));
}

#[test]
fn test_null_block_device_read_only() {
    let dev = NullBlockDevice::new_read_only(512, 100);
    assert!(dev.is_read_only());

    let buf = [0xAAu8; 512];
    assert_eq!(dev.write_block(0, &buf), Err(FsError::ReadOnly));
}

#[test]
fn test_memory_block_device() {
    let dev = MemoryBlockDeviceMut::new(512, 10);

    assert_eq!(dev.block_size(), 512);
    assert_eq!(dev.block_count(), 10);
    assert!(!dev.is_read_only());

    // Write some data
    let write_data = [0xABu8; 512];
    dev.write_block(3, &write_data).unwrap();

    // Read it back
    let mut read_data = [0u8; 512];
    dev.read_block(3, &mut read_data).unwrap();
    assert_eq!(read_data, write_data);

    // Other blocks should still be zero
    dev.read_block(0, &mut read_data).unwrap();
    assert!(read_data.iter().all(|&b| b == 0));
}

#[test]
fn test_memory_block_device_from_data() {
    let mut data = vec![0u8; 512 * 4];
    data[512..1024].fill(0xBB); // Block 1

    let dev = MemoryBlockDevice::from_data(data, 512);
    assert_eq!(dev.block_count(), 4);

    let mut buf = [0u8; 512];
    dev.read_block(1, &mut buf).unwrap();
    assert!(buf.iter().all(|&b| b == 0xBB));
}

// ============================================================================
// RamFS Tests
// ============================================================================

#[test]
fn test_ramfs_mount_unmount() {
    let mut fs = RamFs::new();

    // Not mounted yet
    assert_eq!(fs.lookup(InodeId(1), "test"), Err(FsError::NotSupported));

    fs.mount().unwrap();

    // Can't mount twice
    assert_eq!(fs.mount(), Err(FsError::Busy));

    fs.unmount().unwrap();

    // Can't unmount twice
    assert_eq!(fs.unmount(), Err(FsError::NotSupported));
}

#[test]
fn test_ramfs_basic_operations() {
    let mut fs = RamFs::new();
    fs.mount().unwrap();

    let root = fs.root().unwrap();
    assert_eq!(root, InodeId(1));

    assert_eq!(fs.name(), "ramfs");
    assert!(!fs.is_read_only());
    assert!(fs.total_size() > 0);
    assert!(fs.free_space() <= fs.total_size());
    assert_eq!(fs.block_size(), 4096);
}

#[test]
fn test_ramfs_create_file() {
    let mut fs = RamFs::new();
    fs.mount().unwrap();

    let root = fs.root().unwrap();
    let file_id = fs.create(root, "test.txt", FileType::Regular, 0o644).unwrap();

    assert!(file_id.is_valid());
    assert_ne!(file_id, root);

    let stat = fs.stat(file_id).unwrap();
    assert_eq!(stat.file_type, FileType::Regular);
    assert_eq!(stat.size, 0);
    assert_eq!(stat.nlink, 1);
}

#[test]
fn test_ramfs_create_directory() {
    let mut fs = RamFs::new();
    fs.mount().unwrap();

    let root = fs.root().unwrap();
    let dir_id = fs.create(root, "subdir", FileType::Directory, 0o755).unwrap();

    let stat = fs.stat(dir_id).unwrap();
    assert_eq!(stat.file_type, FileType::Directory);
    assert_eq!(stat.nlink, 2); // . and parent

    // Create nested file
    let file_id = fs.create(dir_id, "nested.txt", FileType::Regular, 0o644).unwrap();
    assert!(file_id.is_valid());

    // Lookup should work
    assert_eq!(fs.lookup(dir_id, "nested.txt").unwrap(), file_id);
}

#[test]
fn test_ramfs_write_read() {
    let mut fs = RamFs::new();
    fs.mount().unwrap();

    let root = fs.root().unwrap();
    let file_id = fs.create(root, "data.bin", FileType::Regular, 0o644).unwrap();

    // Write data
    let data = b"Hello, RamFS!";
    let written = fs.write(file_id, 0, data).unwrap();
    assert_eq!(written, data.len());

    // Check size
    let stat = fs.stat(file_id).unwrap();
    assert_eq!(stat.size, data.len() as u64);

    // Read back
    let mut buf = [0u8; 64];
    let read = fs.read(file_id, 0, &mut buf).unwrap();
    assert_eq!(read, data.len());
    assert_eq!(&buf[..read], data);
}

#[test]
fn test_ramfs_write_at_offset() {
    let mut fs = RamFs::new();
    fs.mount().unwrap();

    let root = fs.root().unwrap();
    let file_id = fs.create(root, "offset.txt", FileType::Regular, 0o644).unwrap();

    // Write at offset
    fs.write(file_id, 10, b"World").unwrap();

    let stat = fs.stat(file_id).unwrap();
    assert_eq!(stat.size, 15); // 10 zeros + "World"

    let mut buf = [0u8; 20];
    let read = fs.read(file_id, 0, &mut buf).unwrap();
    assert_eq!(read, 15);
    assert_eq!(&buf[..10], &[0u8; 10]);
    assert_eq!(&buf[10..15], b"World");
}

#[test]
fn test_ramfs_truncate() {
    let mut fs = RamFs::new();
    fs.mount().unwrap();

    let root = fs.root().unwrap();
    let file_id = fs.create(root, "trunc.txt", FileType::Regular, 0o644).unwrap();

    fs.write(file_id, 0, b"Hello, World!").unwrap();
    assert_eq!(fs.stat(file_id).unwrap().size, 13);

    // Truncate to smaller
    fs.truncate(file_id, 5).unwrap();
    assert_eq!(fs.stat(file_id).unwrap().size, 5);

    let mut buf = [0u8; 10];
    let read = fs.read(file_id, 0, &mut buf).unwrap();
    assert_eq!(read, 5);
    assert_eq!(&buf[..5], b"Hello");

    // Truncate to larger (extends with zeros)
    fs.truncate(file_id, 10).unwrap();
    assert_eq!(fs.stat(file_id).unwrap().size, 10);
}

#[test]
fn test_ramfs_lookup() {
    let mut fs = RamFs::new();
    fs.mount().unwrap();

    let root = fs.root().unwrap();
    let file_id = fs.create(root, "lookup_test.txt", FileType::Regular, 0o644).unwrap();

    assert_eq!(fs.lookup(root, "lookup_test.txt").unwrap(), file_id);
    assert_eq!(fs.lookup(root, "nonexistent.txt"), Err(FsError::NotFound));

    // Lookup in non-directory
    assert_eq!(fs.lookup(file_id, "child"), Err(FsError::NotADirectory));
}

#[test]
fn test_ramfs_lookup_path() {
    let mut fs = RamFs::new();
    fs.mount().unwrap();

    let root = fs.root().unwrap();
    let dir_id = fs.create(root, "a", FileType::Directory, 0o755).unwrap();
    let subdir_id = fs.create(dir_id, "b", FileType::Directory, 0o755).unwrap();
    let file_id = fs.create(subdir_id, "c.txt", FileType::Regular, 0o644).unwrap();

    assert_eq!(fs.lookup_path(Path::new("/")).unwrap(), root);
    assert_eq!(fs.lookup_path(Path::new("/a")).unwrap(), dir_id);
    assert_eq!(fs.lookup_path(Path::new("/a/b")).unwrap(), subdir_id);
    assert_eq!(fs.lookup_path(Path::new("/a/b/c.txt")).unwrap(), file_id);

    // Nonexistent
    assert_eq!(fs.lookup_path(Path::new("/a/nonexistent")), Err(FsError::NotFound));

    // Through non-directory
    assert_eq!(fs.lookup_path(Path::new("/a/b/c.txt/invalid")), Err(FsError::NotADirectory));
}

#[test]
fn test_ramfs_unlink_file() {
    let mut fs = RamFs::new();
    fs.mount().unwrap();

    let root = fs.root().unwrap();
    let file_id = fs.create(root, "delete_me.txt", FileType::Regular, 0o644).unwrap();

    fs.unlink(root, "delete_me.txt").unwrap();

    // File should be gone
    assert_eq!(fs.lookup(root, "delete_me.txt"), Err(FsError::NotFound));

    // Inode should also be gone
    assert_eq!(fs.stat(file_id), Err(FsError::InodeNotFound));
}

#[test]
fn test_ramfs_unlink_empty_directory() {
    let mut fs = RamFs::new();
    fs.mount().unwrap();

    let root = fs.root().unwrap();
    let dir_id = fs.create(root, "empty_dir", FileType::Directory, 0o755).unwrap();

    fs.unlink(root, "empty_dir").unwrap();
    assert_eq!(fs.lookup(root, "empty_dir"), Err(FsError::NotFound));
    assert_eq!(fs.stat(dir_id), Err(FsError::InodeNotFound));
}

#[test]
fn test_ramfs_unlink_nonempty_directory() {
    let mut fs = RamFs::new();
    fs.mount().unwrap();

    let root = fs.root().unwrap();
    let dir_id = fs.create(root, "nonempty", FileType::Directory, 0o755).unwrap();
    fs.create(dir_id, "child.txt", FileType::Regular, 0o644).unwrap();

    assert_eq!(fs.unlink(root, "nonempty"), Err(FsError::DirectoryNotEmpty));
}

#[test]
fn test_ramfs_unlink_special() {
    let mut fs = RamFs::new();
    fs.mount().unwrap();

    let root = fs.root().unwrap();

    assert_eq!(fs.unlink(root, "."), Err(FsError::InvalidArgument));
    assert_eq!(fs.unlink(root, ".."), Err(FsError::InvalidArgument));
}

#[test]
fn test_ramfs_readdir() {
    let mut fs = RamFs::new();
    fs.mount().unwrap();

    let root = fs.root().unwrap();
    fs.create(root, "file1.txt", FileType::Regular, 0o644).unwrap();
    fs.create(root, "file2.txt", FileType::Regular, 0o644).unwrap();
    fs.create(root, "subdir", FileType::Directory, 0o755).unwrap();

    let entries = fs.readdir(root, 0).unwrap();

    // Should contain . .. file1.txt file2.txt subdir
    assert!(entries.len() >= 5);

    let names: Vec<_> = entries.iter().map(|e| e.name()).collect();
    assert!(names.contains(&"."));
    assert!(names.contains(&".."));
    assert!(names.contains(&"file1.txt"));
    assert!(names.contains(&"file2.txt"));
    assert!(names.contains(&"subdir"));
}

#[test]
fn test_ramfs_duplicate_name() {
    let mut fs = RamFs::new();
    fs.mount().unwrap();

    let root = fs.root().unwrap();
    fs.create(root, "exists.txt", FileType::Regular, 0o644).unwrap();

    assert_eq!(
        fs.create(root, "exists.txt", FileType::Regular, 0o644),
        Err(FsError::AlreadyExists)
    );

    assert_eq!(
        fs.create(root, "exists.txt", FileType::Directory, 0o755),
        Err(FsError::AlreadyExists)
    );
}

#[test]
fn test_ramfs_invalid_names() {
    let mut fs = RamFs::new();
    fs.mount().unwrap();

    let root = fs.root().unwrap();

    assert_eq!(
        fs.create(root, "", FileType::Regular, 0o644),
        Err(FsError::InvalidFilename)
    );

    assert_eq!(
        fs.create(root, ".", FileType::Regular, 0o644),
        Err(FsError::InvalidFilename)
    );

    assert_eq!(
        fs.create(root, "..", FileType::Regular, 0o644),
        Err(FsError::InvalidFilename)
    );

    // Name too long
    let long_name = "a".repeat(MAX_NAME_LEN + 1);
    assert_eq!(
        fs.create(root, &long_name, FileType::Regular, 0o644),
        Err(FsError::NameTooLong)
    );
}

#[test]
fn test_ramfs_space_tracking() {
    let fs = RamFs::with_max_size(1024 * 1024);

    assert_eq!(fs.total_size(), 1024 * 1024);
    assert!(fs.free_space() <= fs.total_size());
    assert_eq!(fs.inode_count(), 1); // Just root
}

#[test]
fn test_ram_inode_types() {
    let file = RamInode::new_file(InodeId(1), 0o644);
    assert!(file.is_file());
    assert!(!file.is_dir());
    assert!(!file.is_symlink());
    assert_eq!(file.inode_type, RamInodeType::File);

    let dir = RamInode::new_directory(InodeId(2), InodeId(1), 0o755);
    assert!(!dir.is_file());
    assert!(dir.is_dir());
    assert!(!dir.is_symlink());

    let symlink = RamInode::new_symlink(InodeId(3), "/path/to/target");
    assert!(!symlink.is_file());
    assert!(!symlink.is_dir());
    assert!(symlink.is_symlink());
    assert_eq!(symlink.symlink_target(), Some("/path/to/target"));
}

// ============================================================================
// FAT32 Tests
// ============================================================================

#[test]
fn test_fat32_dir_entry_parse() {
    let mut data = [0u8; 32];
    data[0..11].copy_from_slice(b"TEST    TXT");
    data[11] = 0x20; // Archive
    data[26..28].copy_from_slice(&200u16.to_le_bytes());
    data[28..32].copy_from_slice(&4096u32.to_le_bytes());

    let entry = Fat32DirEntry::parse(&data).unwrap();

    assert!(!entry.is_free());
    assert!(!entry.is_end());
    assert!(!entry.is_directory());
    assert!(!entry.is_lfn());
    assert!(!entry.is_volume_label());
    assert_eq!(entry.first_cluster(), 200);
    assert_eq!(entry.file_size, 4096);
    assert_eq!(entry.file_type(), FileType::Regular);
}

#[test]
fn test_fat32_dir_entry_directory() {
    let mut data = [0u8; 32];
    data[0..11].copy_from_slice(b"SUBDIR     ");
    data[11] = 0x10; // Directory

    let entry = Fat32DirEntry::parse(&data).unwrap();
    assert!(entry.is_directory());
    assert_eq!(entry.file_type(), FileType::Directory);
}

#[test]
fn test_fat32_dir_entry_free() {
    // Deleted entry
    let mut data = [0u8; 32];
    data[0] = 0xE5;
    let entry = Fat32DirEntry::parse(&data).unwrap();
    assert!(entry.is_free());
    assert!(!entry.is_end());

    // End marker
    let mut data = [0u8; 32];
    data[0] = 0x00;
    let entry = Fat32DirEntry::parse(&data).unwrap();
    assert!(entry.is_free());
    assert!(entry.is_end());
}

#[test]
fn test_fat32_dir_entry_name_matching() {
    let mut data = [0u8; 32];
    data[0..11].copy_from_slice(b"README  TXT");

    let entry = Fat32DirEntry::parse(&data).unwrap();

    assert!(entry.matches_name("README.TXT"));
    assert!(entry.matches_name("readme.txt"));
    assert!(entry.matches_name("ReAdMe.TxT"));
    assert!(!entry.matches_name("READ.TXT"));
    assert!(!entry.matches_name("README.DOC"));
}

#[test]
fn test_fat32_short_name() {
    let mut data = [0u8; 32];

    // Normal file
    data[0..11].copy_from_slice(b"TEST    TXT");
    let entry = Fat32DirEntry::parse(&data).unwrap();
    assert_eq!(entry.short_name(), "TEST.TXT");

    // No extension
    data[0..11].copy_from_slice(b"README     ");
    let entry = Fat32DirEntry::parse(&data).unwrap();
    assert_eq!(entry.short_name(), "README");

    // Full 8.3 name
    data[0..11].copy_from_slice(b"LONGNAMEXT1");
    let entry = Fat32DirEntry::parse(&data).unwrap();
    assert_eq!(entry.short_name(), "LONGNAME.XT1");
}

#[test]
fn test_fat32_boot_sector_calculations() {
    let boot = Fat32BootSector {
        bytes_per_sector: 512,
        sectors_per_cluster: 8,
        reserved_sectors: 32,
        num_fats: 2,
        total_sectors: 2097152,
        sectors_per_fat: 2048,
        root_cluster: 2,
        fsinfo_sector: 1,
        backup_boot_sector: 6,
        volume_serial: 0x12345678,
        volume_label: *b"TEST VOLUME",
        fs_type: *b"FAT32   ",
    };

    assert_eq!(boot.bytes_per_cluster(), 4096);
    assert_eq!(boot.first_data_sector(), 32 + 2 * 2048); // 4128
    assert_eq!(boot.cluster_to_sector(2), boot.first_data_sector());
    assert_eq!(boot.cluster_to_sector(3), boot.first_data_sector() + 8);

    assert_eq!(boot.fat_sector_for_cluster(0), 32);
    assert_eq!(boot.fat_offset_for_cluster(0), 0);
    assert_eq!(boot.fat_offset_for_cluster(128), 0); // 128 * 4 = 512 = new sector

    assert_eq!(boot.volume_label_str(), "TEST VOLUME");
}

// ============================================================================
// Error Tests
// ============================================================================

#[test]
fn test_error_to_errno() {
    assert_eq!(FsError::NotFound.to_errno(), 2);
    assert_eq!(FsError::PermissionDenied.to_errno(), 13);
    assert_eq!(FsError::AlreadyExists.to_errno(), 17);
    assert_eq!(FsError::NotADirectory.to_errno(), 20);
    assert_eq!(FsError::IsADirectory.to_errno(), 21);
    assert_eq!(FsError::InvalidArgument.to_errno(), 22);
    assert_eq!(FsError::TooManyOpenFiles.to_errno(), 24);
    assert_eq!(FsError::FileTooLarge.to_errno(), 27);
    assert_eq!(FsError::NoSpace.to_errno(), 28);
    assert_eq!(FsError::ReadOnly.to_errno(), 30);
    assert_eq!(FsError::PathTooLong.to_errno(), 36);
    assert_eq!(FsError::DirectoryNotEmpty.to_errno(), 39);
    assert_eq!(FsError::NotSupported.to_errno(), 95);
}

#[test]
fn test_error_display() {
    assert_eq!(FsError::NotFound.as_str(), "No such file or directory");
    assert_eq!(FsError::PermissionDenied.as_str(), "Permission denied");
    assert_eq!(FsError::ReadOnly.as_str(), "Read-only file system");
    assert_eq!(FsError::NoSpace.as_str(), "No space left on device");
}

// ============================================================================
// Inode Tests
// ============================================================================

#[test]
fn test_inode_creation() {
    let dir = Inode::directory(InodeId(1));
    assert!(dir.is_dir());
    assert!(!dir.is_file());
    assert!(!dir.is_symlink());
    assert_eq!(dir.mode, 0o755);
    assert_eq!(dir.nlink, 2);

    let file = Inode::file(InodeId(2));
    assert!(!file.is_dir());
    assert!(file.is_file());
    assert!(!file.is_symlink());
    assert_eq!(file.mode, 0o644);
    assert_eq!(file.nlink, 1);
}
