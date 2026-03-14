# ruvix-fs

Minimal filesystem implementation for the RuVix Cognition Kernel (ADR-087 Phase E).

## Overview

This crate provides a VFS (Virtual Filesystem) layer with pluggable filesystem backends:

- **RamFS**: In-memory filesystem for `/tmp` and temporary storage
- **FAT32**: Read-only FAT32 filesystem for boot partitions and external media

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │           VFS Mount Table           │
                    │  (path-to-filesystem resolution)    │
                    └─────────────────────────────────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            │                        │                        │
            ▼                        ▼                        ▼
    ┌───────────────┐       ┌───────────────┐        ┌───────────────┐
    │    RamFS      │       │    FAT32      │        │   Future FS   │
    │  (read/write) │       │  (read-only)  │        │    (ext4?)    │
    └───────────────┘       └───────────────┘        └───────────────┘
            │                        │
            ▼                        ▼
    ┌───────────────┐       ┌───────────────┐
    │    Memory     │       │  BlockDevice  │
    │   (Vec<u8>)   │       │    Trait      │
    └───────────────┘       └───────────────┘
```

## Features

| Feature | Description |
|---------|-------------|
| `std` | Enable standard library support |
| `alloc` | Enable alloc crate for heap allocation |
| `lfn` | Enable FAT32 Long Filename support |
| `fat32-write` | Enable write support for FAT32 (Phase 2) |
| `stats` | Enable filesystem statistics collection |

## Usage

### RamFS (In-Memory Filesystem)

```rust
use ruvix_fs::{RamFs, FileSystem, FileType, Path};

// Create and mount
let mut fs = RamFs::new();
fs.mount()?;

// Get root directory
let root = fs.root()?;

// Create files and directories
let file_id = fs.create(root, "hello.txt", FileType::Regular, 0o644)?;
let dir_id = fs.create(root, "subdir", FileType::Directory, 0o755)?;

// Write data
fs.write(file_id, 0, b"Hello, World!")?;

// Read data
let mut buf = [0u8; 64];
let bytes_read = fs.read(file_id, 0, &mut buf)?;

// Lookup by path
let found = fs.lookup_path(Path::new("/subdir"))?;

// List directory
let entries = fs.readdir(root, 0)?;
for entry in entries {
    println!("{}: {:?}", entry.name(), entry.file_type);
}

// Cleanup
fs.unlink(root, "hello.txt")?;
fs.unmount()?;
```

### FAT32 Filesystem

```rust
use ruvix_fs::{Fat32Fs, FileSystem, Path, MemoryBlockDevice};

// Create block device from disk image
let disk_data = std::fs::read("boot.img")?;
let device = MemoryBlockDevice::from_data(disk_data, 512);

// Create and mount FAT32
let mut fs = Fat32Fs::new(device)?;
fs.mount()?;

// Read boot sector info
let boot = fs.boot_sector();
println!("Volume: {}", boot.volume_label_str());
println!("Cluster size: {} bytes", boot.bytes_per_cluster());

// Lookup files
let kernel_id = fs.lookup_path(Path::new("/boot/kernel.bin"))?;

// Read file contents
let mut buf = vec![0u8; 4096];
let bytes_read = fs.read(kernel_id, 0, &mut buf)?;

fs.unmount()?;
```

### VFS Mount Table

```rust
use ruvix_fs::{VfsMountTable, Path};

let mut mounts = VfsMountTable::new();

// Mount filesystems
let root_id = mounts.mount("/", "rootfs", "ramfs")?;
let boot_id = mounts.mount("/boot", "disk0p1", "fat32")?;
let tmp_id = mounts.mount("/tmp", "tmpfs", "ramfs")?;

// Find mount point for path
let mount = mounts.find(Path::new("/boot/kernel.bin")).unwrap();
assert_eq!(mount.id, boot_id);

// Get relative path within mount
let rel = mount.relative_path(Path::new("/boot/kernel.bin")).unwrap();
assert_eq!(rel.as_str(), "/kernel.bin");

// Unmount
mounts.unmount(tmp_id)?;
```

### Block Device Abstraction

```rust
use ruvix_fs::{BlockDevice, MemoryBlockDeviceMut, NullBlockDevice};

// Memory-backed device
let device = MemoryBlockDeviceMut::new(512, 2048); // 1MB

// Write a block
let data = [0xAB; 512];
device.write_block(0, &data)?;

// Read a block
let mut buf = [0u8; 512];
device.read_block(0, &mut buf)?;

// Null device (for testing)
let null_dev = NullBlockDevice::new(512, 100);
```

## Core Types

### Path Types

| Type | Description |
|------|-------------|
| `Path` | Borrowed path slice (like `&str`) |
| `PathBuf` | Owned path buffer (like `String`) |
| `PathComponent` | Iterator item: `RootDir`, `CurDir`, `ParentDir`, `Normal(&str)` |

### VFS Types

| Type | Description |
|------|-------------|
| `InodeId` | Unique inode identifier within a filesystem |
| `MountId` | Mount point identifier |
| `FileType` | File type enum (Regular, Directory, Symlink, etc.) |
| `DirEntry` | Directory entry with name, inode, and type |
| `OpenFile` | Open file handle with seek position |
| `OpenFlags` | File open flags (READ, WRITE, CREATE, etc.) |
| `Inode` | Inode metadata structure |

### Filesystem Trait

```rust
pub trait FileSystem {
    fn mount(&mut self) -> FsResult<()>;
    fn unmount(&mut self) -> FsResult<()>;
    fn root(&self) -> FsResult<InodeId>;
    fn name(&self) -> &str;
    fn is_read_only(&self) -> bool;
    fn total_size(&self) -> u64;
    fn free_space(&self) -> u64;
    fn block_size(&self) -> u32;
    fn sync(&self) -> FsResult<()>;

    fn lookup_path(&self, path: &Path) -> FsResult<InodeId>;
    fn stat(&self, inode_id: InodeId) -> FsResult<Inode>;
    fn read(&self, inode_id: InodeId, offset: u64, buf: &mut [u8]) -> FsResult<usize>;
    fn write(&self, inode_id: InodeId, offset: u64, buf: &[u8]) -> FsResult<usize>;
    fn truncate(&self, inode_id: InodeId, size: u64) -> FsResult<()>;
    fn lookup(&self, dir_inode: InodeId, name: &str) -> FsResult<InodeId>;
    fn create(&self, dir_inode: InodeId, name: &str, file_type: FileType, mode: u32) -> FsResult<InodeId>;
    fn unlink(&self, dir_inode: InodeId, name: &str) -> FsResult<()>;
    fn readdir(&self, dir_inode: InodeId, offset: usize) -> FsResult<Vec<DirEntry>>;
}
```

### Block Device Trait

```rust
pub trait BlockDevice {
    fn read_block(&self, lba: u64, buf: &mut [u8]) -> FsResult<()>;
    fn write_block(&self, lba: u64, buf: &[u8]) -> FsResult<()>;
    fn block_size(&self) -> usize;
    fn block_count(&self) -> u64;
    fn is_read_only(&self) -> bool;
    fn sync(&self) -> FsResult<()>;
}
```

## Error Handling

All operations return `FsResult<T>` which is `Result<T, FsError>`. Error types include:

| Error | POSIX | Description |
|-------|-------|-------------|
| `NotFound` | ENOENT | File or directory not found |
| `PermissionDenied` | EACCES | Permission denied |
| `AlreadyExists` | EEXIST | File already exists |
| `NotADirectory` | ENOTDIR | Not a directory |
| `IsADirectory` | EISDIR | Is a directory |
| `DirectoryNotEmpty` | ENOTEMPTY | Directory not empty |
| `NoSpace` | ENOSPC | No space left on device |
| `ReadOnly` | EROFS | Read-only filesystem |
| `InvalidArgument` | EINVAL | Invalid argument |
| `IoError` | EIO | I/O error |

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_PATH_LEN` | 4096 | Maximum path length in bytes |
| `MAX_NAME_LEN` | 255 | Maximum filename length |
| `DEFAULT_BLOCK_SIZE` | 4096 | Default filesystem block size |
| `FAT32_SECTOR_SIZE` | 512 | FAT32 sector size |
| `MAX_MOUNTS` | 64 | Maximum mount points |
| `MAX_OPEN_FILES` | 256 | Maximum open files per process |
| `ROOT_INODE_ID` | 1 | Root inode ID |

## FAT32 Implementation Details

### Supported Features

- FAT32 boot sector parsing
- FAT table traversal
- Cluster chain following
- 8.3 filename support
- Long Filename (LFN) support (with `lfn` feature)
- Directory enumeration
- File reading

### Limitations

- Read-only by default
- No FAT12/FAT16 support
- No journaling
- Write support planned for Phase 2

### Boot Sector Structure

```rust
pub struct Fat32BootSector {
    pub bytes_per_sector: u16,
    pub sectors_per_cluster: u8,
    pub reserved_sectors: u16,
    pub num_fats: u8,
    pub total_sectors: u32,
    pub sectors_per_fat: u32,
    pub root_cluster: u32,
    pub fsinfo_sector: u16,
    pub backup_boot_sector: u16,
    pub volume_serial: u32,
    pub volume_label: [u8; 11],
    pub fs_type: [u8; 8],
}
```

## Testing

```bash
# Run tests
cargo test --features alloc

# Run with LFN support
cargo test --features alloc,lfn

# Run benchmarks
cargo bench --features alloc
```

## Performance

Benchmark results (on typical hardware):

| Operation | Time |
|-----------|------|
| Path parse (short) | ~10 ns |
| Path parse (long) | ~50 ns |
| RamFS create file | ~200 ns |
| RamFS lookup | ~100 ns |
| RamFS read 1KB | ~50 ns |
| RamFS write 1KB | ~100 ns |
| RamFS readdir (100 entries) | ~5 us |

## License

MIT OR Apache-2.0

## References

- [ADR-087: RuVix Cognition Kernel](../../docs/adr/ADR-087-ruvix-cognition-kernel.md)
- [FAT32 Specification](https://www.cs.fsu.edu/~cop4610t/assignments/project3/spec/fatspec.pdf)
- [POSIX Filesystem Semantics](https://pubs.opengroup.org/onlinepubs/9699919799/)
