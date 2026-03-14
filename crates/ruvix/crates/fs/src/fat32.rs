//! FAT32 filesystem implementation for RuVix.
//!
//! This module provides a read-only FAT32 filesystem implementation that can
//! be used to access FAT32 formatted storage devices.
//!
//! ## Features
//!
//! - FAT32 boot sector parsing
//! - Cluster chain traversal
//! - 8.3 filename support
//! - Long filename (LFN) support (with `lfn` feature)
//! - Directory enumeration
//! - File reading
//!
//! ## Limitations
//!
//! - Read-only by default (write support behind `fat32-write` feature)
//! - No journaling
//! - No FAT12/FAT16 support

use crate::block::BlockDevice;
use crate::error::{FsError, FsResult};
use crate::path::{Path, PathComponent};
use crate::vfs::{DirEntry, FileSystem, FileType, Inode, InodeId};
use crate::FAT32_SECTOR_SIZE;

#[cfg(feature = "alloc")]
use alloc::{string::String, vec, vec::Vec};

/// FAT32 special cluster values.
const FAT32_EOC_MIN: u32 = 0x0FFF_FFF8;
const FAT32_BAD_CLUSTER: u32 = 0x0FFF_FFF7;
const FAT32_FREE_CLUSTER: u32 = 0x0000_0000;

/// Directory entry attributes.
const ATTR_READ_ONLY: u8 = 0x01;
const ATTR_HIDDEN: u8 = 0x02;
const ATTR_SYSTEM: u8 = 0x04;
const ATTR_VOLUME_ID: u8 = 0x08;
const ATTR_DIRECTORY: u8 = 0x10;
const ATTR_ARCHIVE: u8 = 0x20;
const ATTR_LONG_NAME: u8 = ATTR_READ_ONLY | ATTR_HIDDEN | ATTR_SYSTEM | ATTR_VOLUME_ID;

/// Size of a FAT32 directory entry.
const DIR_ENTRY_SIZE: usize = 32;

/// FAT32 Boot Sector structure.
#[derive(Debug, Clone, Copy)]
pub struct Fat32BootSector {
    /// Bytes per sector (usually 512).
    pub bytes_per_sector: u16,
    /// Sectors per cluster.
    pub sectors_per_cluster: u8,
    /// Reserved sector count (includes boot sector).
    pub reserved_sectors: u16,
    /// Number of FATs (usually 2).
    pub num_fats: u8,
    /// Total sectors (32-bit for FAT32).
    pub total_sectors: u32,
    /// Sectors per FAT.
    pub sectors_per_fat: u32,
    /// Root directory cluster.
    pub root_cluster: u32,
    /// Filesystem info sector.
    pub fsinfo_sector: u16,
    /// Backup boot sector.
    pub backup_boot_sector: u16,
    /// Volume serial number.
    pub volume_serial: u32,
    /// Volume label.
    pub volume_label: [u8; 11],
    /// Filesystem type string.
    pub fs_type: [u8; 8],
}

impl Fat32BootSector {
    /// Parse a boot sector from raw bytes.
    pub fn parse(data: &[u8]) -> FsResult<Self> {
        if data.len() < 512 {
            return Err(FsError::InvalidBootSector);
        }

        // Check boot signature
        if data[510] != 0x55 || data[511] != 0xAA {
            return Err(FsError::InvalidBootSector);
        }

        let bytes_per_sector = u16::from_le_bytes([data[11], data[12]]);
        let sectors_per_cluster = data[13];
        let reserved_sectors = u16::from_le_bytes([data[14], data[15]]);
        let num_fats = data[16];

        // FAT32 should have root_entry_count = 0
        let root_entry_count = u16::from_le_bytes([data[17], data[18]]);
        if root_entry_count != 0 {
            return Err(FsError::InvalidFilesystem);
        }

        // Total sectors (use 32-bit value for FAT32)
        let total_sectors_16 = u16::from_le_bytes([data[19], data[20]]);
        let total_sectors_32 = u32::from_le_bytes([data[32], data[33], data[34], data[35]]);
        let total_sectors = if total_sectors_16 == 0 {
            total_sectors_32
        } else {
            u32::from(total_sectors_16)
        };

        // Sectors per FAT (32-bit for FAT32)
        let sectors_per_fat = u32::from_le_bytes([data[36], data[37], data[38], data[39]]);

        let root_cluster = u32::from_le_bytes([data[44], data[45], data[46], data[47]]);
        let fsinfo_sector = u16::from_le_bytes([data[48], data[49]]);
        let backup_boot_sector = u16::from_le_bytes([data[50], data[51]]);
        let volume_serial = u32::from_le_bytes([data[67], data[68], data[69], data[70]]);

        let mut volume_label = [0u8; 11];
        volume_label.copy_from_slice(&data[71..82]);

        let mut fs_type = [0u8; 8];
        fs_type.copy_from_slice(&data[82..90]);

        // Validate FAT32 signature
        if &fs_type[..5] != b"FAT32" {
            return Err(FsError::InvalidFilesystem);
        }

        Ok(Self {
            bytes_per_sector,
            sectors_per_cluster,
            reserved_sectors,
            num_fats,
            total_sectors,
            sectors_per_fat,
            root_cluster,
            fsinfo_sector,
            backup_boot_sector,
            volume_serial,
            volume_label,
            fs_type,
        })
    }

    /// Get the number of bytes per cluster.
    #[must_use]
    pub const fn bytes_per_cluster(&self) -> u32 {
        self.bytes_per_sector as u32 * self.sectors_per_cluster as u32
    }

    /// Get the first data sector.
    #[must_use]
    pub const fn first_data_sector(&self) -> u32 {
        self.reserved_sectors as u32 + (self.num_fats as u32 * self.sectors_per_fat)
    }

    /// Convert cluster number to sector number.
    #[must_use]
    pub const fn cluster_to_sector(&self, cluster: u32) -> u32 {
        self.first_data_sector() + (cluster - 2) * self.sectors_per_cluster as u32
    }

    /// Get the sector containing a FAT entry.
    #[must_use]
    pub const fn fat_sector_for_cluster(&self, cluster: u32) -> u32 {
        self.reserved_sectors as u32 + (cluster * 4) / self.bytes_per_sector as u32
    }

    /// Get the offset within a FAT sector for a cluster entry.
    #[must_use]
    pub const fn fat_offset_for_cluster(&self, cluster: u32) -> usize {
        ((cluster * 4) % self.bytes_per_sector as u32) as usize
    }

    /// Get the volume label as a string.
    #[must_use]
    pub fn volume_label_str(&self) -> &str {
        let end = self.volume_label.iter().rposition(|&b| b != b' ').map_or(0, |i| i + 1);
        core::str::from_utf8(&self.volume_label[..end]).unwrap_or("")
    }
}

/// FAT32 directory entry (8.3 format).
#[derive(Debug, Clone, Copy)]
pub struct Fat32DirEntry {
    /// 8.3 filename (8 bytes name + 3 bytes extension).
    pub name: [u8; 11],
    /// File attributes.
    pub attributes: u8,
    /// Reserved byte (NT case info).
    pub nt_reserved: u8,
    /// Creation time (tenths of second).
    pub create_time_tenths: u8,
    /// Creation time.
    pub create_time: u16,
    /// Creation date.
    pub create_date: u16,
    /// Last access date.
    pub access_date: u16,
    /// High 16 bits of first cluster.
    pub cluster_high: u16,
    /// Last write time.
    pub write_time: u16,
    /// Last write date.
    pub write_date: u16,
    /// Low 16 bits of first cluster.
    pub cluster_low: u16,
    /// File size in bytes.
    pub file_size: u32,
}

impl Fat32DirEntry {
    /// Parse a directory entry from raw bytes.
    pub fn parse(data: &[u8]) -> FsResult<Self> {
        if data.len() < DIR_ENTRY_SIZE {
            return Err(FsError::InvalidArgument);
        }

        let mut name = [0u8; 11];
        name.copy_from_slice(&data[0..11]);

        Ok(Self {
            name,
            attributes: data[11],
            nt_reserved: data[12],
            create_time_tenths: data[13],
            create_time: u16::from_le_bytes([data[14], data[15]]),
            create_date: u16::from_le_bytes([data[16], data[17]]),
            access_date: u16::from_le_bytes([data[18], data[19]]),
            cluster_high: u16::from_le_bytes([data[20], data[21]]),
            write_time: u16::from_le_bytes([data[22], data[23]]),
            write_date: u16::from_le_bytes([data[24], data[25]]),
            cluster_low: u16::from_le_bytes([data[26], data[27]]),
            file_size: u32::from_le_bytes([data[28], data[29], data[30], data[31]]),
        })
    }

    /// Check if this is a free entry.
    #[must_use]
    pub const fn is_free(&self) -> bool {
        self.name[0] == 0x00 || self.name[0] == 0xE5
    }

    /// Check if this is the end marker.
    #[must_use]
    pub const fn is_end(&self) -> bool {
        self.name[0] == 0x00
    }

    /// Check if this is a long filename entry.
    #[must_use]
    pub const fn is_lfn(&self) -> bool {
        self.attributes == ATTR_LONG_NAME
    }

    /// Check if this is a directory.
    #[must_use]
    pub const fn is_directory(&self) -> bool {
        (self.attributes & ATTR_DIRECTORY) != 0
    }

    /// Check if this is a volume label.
    #[must_use]
    pub const fn is_volume_label(&self) -> bool {
        (self.attributes & ATTR_VOLUME_ID) != 0 && !self.is_lfn()
    }

    /// Get the first cluster number.
    #[must_use]
    pub const fn first_cluster(&self) -> u32 {
        ((self.cluster_high as u32) << 16) | (self.cluster_low as u32)
    }

    /// Get the file type.
    #[must_use]
    pub const fn file_type(&self) -> FileType {
        if self.is_directory() {
            FileType::Directory
        } else {
            FileType::Regular
        }
    }

    /// Get the 8.3 filename as a string.
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn short_name(&self) -> String {
        // Handle special first byte
        let first = if self.name[0] == 0x05 { 0xE5 } else { self.name[0] };

        // Get name part (first 8 bytes, trimmed)
        let name_end = self.name[..8].iter().rposition(|&b| b != b' ').map_or(0, |i| i + 1);
        let name_bytes: Vec<u8> = core::iter::once(first)
            .chain(self.name[1..name_end].iter().copied())
            .collect();

        // Get extension part (last 3 bytes, trimmed)
        let ext_end = self.name[8..11].iter().rposition(|&b| b != b' ').map_or(0, |i| i + 1);

        let name_str = String::from_utf8_lossy(&name_bytes);
        if ext_end > 0 {
            let ext_str = String::from_utf8_lossy(&self.name[8..8 + ext_end]);
            alloc::format!("{}.{}", name_str, ext_str)
        } else {
            name_str.into_owned()
        }
    }

    /// Check if the entry matches a given name (case-insensitive).
    #[must_use]
    pub fn matches_name(&self, name: &str) -> bool {
        // Convert input name to 8.3 format for comparison
        let (base, ext) = if let Some(dot_pos) = name.rfind('.') {
            (&name[..dot_pos], Some(&name[dot_pos + 1..]))
        } else {
            (name, None)
        };

        // Compare base name (first 8 bytes)
        for (i, byte) in self.name[..8].iter().enumerate() {
            let expected = if i < base.len() {
                base.as_bytes()[i].to_ascii_uppercase()
            } else {
                b' '
            };
            if byte.to_ascii_uppercase() != expected {
                return false;
            }
        }

        // Compare extension (last 3 bytes)
        if let Some(ext) = ext {
            for (i, byte) in self.name[8..11].iter().enumerate() {
                let expected = if i < ext.len() {
                    ext.as_bytes()[i].to_ascii_uppercase()
                } else {
                    b' '
                };
                if byte.to_ascii_uppercase() != expected {
                    return false;
                }
            }
        } else {
            // No extension in input, check that entry has no extension
            if self.name[8..11].iter().any(|&b| b != b' ') {
                // Entry has extension, check if it matches without dot
                // This handles cases like "README" matching "README" entry
                for byte in &self.name[8..11] {
                    if *byte != b' ' {
                        return false;
                    }
                }
            }
        }

        true
    }
}

/// FAT32 Long Filename entry.
#[cfg(feature = "lfn")]
#[derive(Debug, Clone, Copy)]
pub struct Fat32LfnEntry {
    /// Sequence number.
    pub sequence: u8,
    /// Name characters 1-5 (UCS-2).
    pub name1: [u16; 5],
    /// Attributes (always ATTR_LONG_NAME).
    pub attributes: u8,
    /// Type (always 0).
    pub entry_type: u8,
    /// Checksum of 8.3 name.
    pub checksum: u8,
    /// Name characters 6-11 (UCS-2).
    pub name2: [u16; 6],
    /// Always 0.
    pub first_cluster: u16,
    /// Name characters 12-13 (UCS-2).
    pub name3: [u16; 2],
}

#[cfg(feature = "lfn")]
impl Fat32LfnEntry {
    /// Parse a LFN entry from raw bytes.
    pub fn parse(data: &[u8]) -> FsResult<Self> {
        if data.len() < DIR_ENTRY_SIZE {
            return Err(FsError::InvalidArgument);
        }

        let sequence = data[0];

        let mut name1 = [0u16; 5];
        for (i, chunk) in data[1..11].chunks(2).enumerate() {
            name1[i] = u16::from_le_bytes([chunk[0], chunk[1]]);
        }

        let mut name2 = [0u16; 6];
        for (i, chunk) in data[14..26].chunks(2).enumerate() {
            name2[i] = u16::from_le_bytes([chunk[0], chunk[1]]);
        }

        let mut name3 = [0u16; 2];
        for (i, chunk) in data[28..32].chunks(2).enumerate() {
            name3[i] = u16::from_le_bytes([chunk[0], chunk[1]]);
        }

        Ok(Self {
            sequence,
            name1,
            attributes: data[11],
            entry_type: data[12],
            checksum: data[13],
            name2,
            first_cluster: u16::from_le_bytes([data[26], data[27]]),
            name3,
        })
    }

    /// Check if this is the last LFN entry.
    #[must_use]
    pub const fn is_last(&self) -> bool {
        (self.sequence & 0x40) != 0
    }

    /// Get the sequence number (1-based, without last flag).
    #[must_use]
    pub const fn sequence_number(&self) -> u8 {
        self.sequence & 0x1F
    }

    /// Get the name characters from this entry.
    #[cfg(feature = "alloc")]
    pub fn name_chars(&self) -> Vec<u16> {
        self.name1
            .iter()
            .chain(self.name2.iter())
            .chain(self.name3.iter())
            .take_while(|&&c| c != 0 && c != 0xFFFF)
            .copied()
            .collect()
    }
}

/// FAT32 inode wrapper.
#[derive(Debug)]
pub struct Fat32Inode {
    /// Inode ID (cluster number or special).
    pub id: InodeId,
    /// First cluster.
    pub first_cluster: u32,
    /// File size.
    pub size: u64,
    /// File type.
    pub file_type: FileType,
    /// Attributes.
    pub attributes: u8,
}

impl Fat32Inode {
    /// Create a new FAT32 inode.
    #[must_use]
    pub const fn new(cluster: u32, size: u64, file_type: FileType, attributes: u8) -> Self {
        Self {
            id: InodeId(cluster as u64),
            first_cluster: cluster,
            size,
            file_type,
            attributes,
        }
    }

    /// Create a root directory inode.
    #[must_use]
    pub const fn root(cluster: u32) -> Self {
        Self {
            id: InodeId(1),
            first_cluster: cluster,
            size: 0,
            file_type: FileType::Directory,
            attributes: ATTR_DIRECTORY,
        }
    }

    /// Create from a directory entry.
    #[must_use]
    pub const fn from_dir_entry(entry: &Fat32DirEntry) -> Self {
        Self {
            id: InodeId(entry.first_cluster() as u64),
            first_cluster: entry.first_cluster(),
            size: entry.file_size as u64,
            file_type: entry.file_type(),
            attributes: entry.attributes,
        }
    }
}

/// FAT32 filesystem implementation.
#[cfg(feature = "alloc")]
pub struct Fat32Fs<B: BlockDevice> {
    /// Block device.
    device: B,
    /// Boot sector.
    boot_sector: Fat32BootSector,
    /// Cached FAT sectors.
    fat_cache: Vec<(u32, Vec<u8>)>,
    /// Maximum FAT cache entries.
    fat_cache_max: usize,
    /// Is the filesystem mounted?
    mounted: bool,
}

#[cfg(feature = "alloc")]
impl<B: BlockDevice> Fat32Fs<B> {
    /// Create a new FAT32 filesystem from a block device.
    pub fn new(device: B) -> FsResult<Self> {
        let block_size = device.block_size();
        if block_size != FAT32_SECTOR_SIZE {
            return Err(FsError::InvalidArgument);
        }

        // Read boot sector
        let mut boot_data = vec![0u8; FAT32_SECTOR_SIZE];
        device.read_block(0, &mut boot_data)?;

        let boot_sector = Fat32BootSector::parse(&boot_data)?;

        Ok(Self {
            device,
            boot_sector,
            fat_cache: Vec::new(),
            fat_cache_max: 8,
            mounted: false,
        })
    }

    /// Get the boot sector.
    #[must_use]
    pub const fn boot_sector(&self) -> &Fat32BootSector {
        &self.boot_sector
    }

    /// Read a cluster's data into a buffer.
    pub fn read_cluster(&self, cluster: u32, buf: &mut [u8]) -> FsResult<()> {
        let bytes_per_cluster = self.boot_sector.bytes_per_cluster() as usize;
        if buf.len() < bytes_per_cluster {
            return Err(FsError::InvalidArgument);
        }

        let start_sector = self.boot_sector.cluster_to_sector(cluster);
        let sectors_per_cluster = self.boot_sector.sectors_per_cluster as usize;

        for i in 0..sectors_per_cluster {
            let offset = i * FAT32_SECTOR_SIZE;
            self.device.read_block(
                (start_sector + i as u32) as u64,
                &mut buf[offset..offset + FAT32_SECTOR_SIZE],
            )?;
        }

        Ok(())
    }

    /// Read a FAT entry for a cluster.
    pub fn read_fat_entry(&self, cluster: u32) -> FsResult<u32> {
        let fat_sector = self.boot_sector.fat_sector_for_cluster(cluster);
        let offset = self.boot_sector.fat_offset_for_cluster(cluster);

        // Check cache first
        for (cached_sector, data) in &self.fat_cache {
            if *cached_sector == fat_sector {
                return Ok(u32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]) & 0x0FFF_FFFF);
            }
        }

        // Read sector
        let mut sector_data = vec![0u8; FAT32_SECTOR_SIZE];
        self.device.read_block(fat_sector as u64, &mut sector_data)?;

        let entry = u32::from_le_bytes([
            sector_data[offset],
            sector_data[offset + 1],
            sector_data[offset + 2],
            sector_data[offset + 3],
        ]) & 0x0FFF_FFFF;

        Ok(entry)
    }

    /// Follow a cluster chain and return all clusters.
    pub fn follow_cluster_chain(&self, start: u32) -> FsResult<Vec<u32>> {
        let mut clusters = Vec::new();
        let mut current = start;

        // Safety limit to prevent infinite loops
        let max_clusters = (self.boot_sector.total_sectors / self.boot_sector.sectors_per_cluster as u32) as usize;

        while current >= 2 && current < FAT32_EOC_MIN && clusters.len() < max_clusters {
            clusters.push(current);
            current = self.read_fat_entry(current)?;

            if current == FAT32_BAD_CLUSTER {
                return Err(FsError::InvalidClusterChain);
            }
        }

        if current >= FAT32_EOC_MIN {
            // End of chain reached normally
        } else if current != FAT32_FREE_CLUSTER && clusters.len() >= max_clusters {
            return Err(FsError::InvalidClusterChain);
        }

        Ok(clusters)
    }

    /// Read directory entries from a directory cluster.
    pub fn read_directory(&self, first_cluster: u32) -> FsResult<Vec<(Fat32DirEntry, String)>> {
        let clusters = self.follow_cluster_chain(first_cluster)?;
        let bytes_per_cluster = self.boot_sector.bytes_per_cluster() as usize;
        let mut cluster_data = vec![0u8; bytes_per_cluster];
        let mut entries = Vec::new();

        #[cfg(feature = "lfn")]
        let mut lfn_parts: Vec<(u8, Vec<u16>)> = Vec::new();

        for cluster in clusters {
            self.read_cluster(cluster, &mut cluster_data)?;

            let entry_count = bytes_per_cluster / DIR_ENTRY_SIZE;
            for i in 0..entry_count {
                let offset = i * DIR_ENTRY_SIZE;
                let entry_data = &cluster_data[offset..offset + DIR_ENTRY_SIZE];

                // Check for end of directory
                if entry_data[0] == 0x00 {
                    return Ok(entries);
                }

                // Skip deleted entries
                if entry_data[0] == 0xE5 {
                    #[cfg(feature = "lfn")]
                    lfn_parts.clear();
                    continue;
                }

                // Parse entry
                let entry = Fat32DirEntry::parse(entry_data)?;

                #[cfg(feature = "lfn")]
                if entry.is_lfn() {
                    let lfn = Fat32LfnEntry::parse(entry_data)?;
                    lfn_parts.push((lfn.sequence_number(), lfn.name_chars()));
                    continue;
                }

                // Skip volume labels
                if entry.is_volume_label() {
                    #[cfg(feature = "lfn")]
                    lfn_parts.clear();
                    continue;
                }

                // Build name
                #[cfg(feature = "lfn")]
                let name = if !lfn_parts.is_empty() {
                    // Sort LFN parts by sequence number and combine
                    lfn_parts.sort_by_key(|(seq, _)| *seq);
                    let chars: Vec<u16> = lfn_parts.iter().rev().flat_map(|(_, chars)| chars.iter()).copied().collect();
                    let name = String::from_utf16_lossy(&chars);
                    lfn_parts.clear();
                    name
                } else {
                    entry.short_name()
                };

                #[cfg(not(feature = "lfn"))]
                let name = entry.short_name();

                entries.push((entry, name));
            }
        }

        Ok(entries)
    }

    /// Find an entry in a directory by name.
    pub fn find_entry(&self, dir_cluster: u32, name: &str) -> FsResult<Fat32DirEntry> {
        let entries = self.read_directory(dir_cluster)?;

        for (entry, entry_name) in entries {
            // Case-insensitive comparison
            if entry_name.eq_ignore_ascii_case(name) {
                return Ok(entry);
            }
        }

        Err(FsError::NotFound)
    }

    /// Read file data.
    pub fn read_file(&self, inode: &Fat32Inode, offset: u64, buf: &mut [u8]) -> FsResult<usize> {
        if inode.file_type == FileType::Directory {
            return Err(FsError::IsADirectory);
        }

        if offset >= inode.size {
            return Ok(0);
        }

        let clusters = self.follow_cluster_chain(inode.first_cluster)?;
        let bytes_per_cluster = self.boot_sector.bytes_per_cluster() as u64;
        let mut cluster_data = vec![0u8; bytes_per_cluster as usize];

        let mut bytes_read = 0usize;
        let mut current_offset = offset;

        // Find starting cluster
        let start_cluster_idx = (current_offset / bytes_per_cluster) as usize;
        let mut offset_in_cluster = (current_offset % bytes_per_cluster) as usize;

        for cluster in clusters.iter().skip(start_cluster_idx) {
            self.read_cluster(*cluster, &mut cluster_data)?;

            let bytes_remaining = (inode.size - current_offset) as usize;
            let bytes_available = bytes_per_cluster as usize - offset_in_cluster;
            let bytes_to_copy = bytes_remaining
                .min(bytes_available)
                .min(buf.len() - bytes_read);

            buf[bytes_read..bytes_read + bytes_to_copy]
                .copy_from_slice(&cluster_data[offset_in_cluster..offset_in_cluster + bytes_to_copy]);

            bytes_read += bytes_to_copy;
            current_offset += bytes_to_copy as u64;
            offset_in_cluster = 0;

            if bytes_read >= buf.len() || current_offset >= inode.size {
                break;
            }
        }

        Ok(bytes_read)
    }
}

#[cfg(feature = "alloc")]
impl<B: BlockDevice> FileSystem for Fat32Fs<B> {
    fn mount(&mut self) -> FsResult<()> {
        if self.mounted {
            return Err(FsError::Busy);
        }
        self.mounted = true;
        Ok(())
    }

    fn unmount(&mut self) -> FsResult<()> {
        if !self.mounted {
            return Err(FsError::NotSupported);
        }
        self.mounted = false;
        self.fat_cache.clear();
        Ok(())
    }

    fn root(&self) -> FsResult<InodeId> {
        Ok(InodeId(1))
    }

    fn name(&self) -> &str {
        "fat32"
    }

    fn is_read_only(&self) -> bool {
        #[cfg(feature = "fat32-write")]
        {
            self.device.is_read_only()
        }
        #[cfg(not(feature = "fat32-write"))]
        {
            true
        }
    }

    fn total_size(&self) -> u64 {
        u64::from(self.boot_sector.total_sectors) * u64::from(self.boot_sector.bytes_per_sector)
    }

    fn free_space(&self) -> u64 {
        // Would need to scan FAT for free clusters
        0
    }

    fn block_size(&self) -> u32 {
        self.boot_sector.bytes_per_cluster()
    }

    fn sync(&self) -> FsResult<()> {
        Ok(())
    }

    fn lookup_path(&self, path: &Path) -> FsResult<InodeId> {
        if !self.mounted {
            return Err(FsError::NotSupported);
        }

        let mut current_cluster = self.boot_sector.root_cluster;

        for component in path.components() {
            match component {
                PathComponent::RootDir => {
                    current_cluster = self.boot_sector.root_cluster;
                }
                PathComponent::CurDir => {}
                PathComponent::ParentDir => {
                    // Find parent entry
                    let entry = self.find_entry(current_cluster, "..")?;
                    current_cluster = entry.first_cluster();
                    if current_cluster == 0 {
                        current_cluster = self.boot_sector.root_cluster;
                    }
                }
                PathComponent::Normal(name) => {
                    let entry = self.find_entry(current_cluster, name)?;
                    current_cluster = entry.first_cluster();
                }
            }
        }

        Ok(InodeId(current_cluster as u64))
    }

    fn stat(&self, inode_id: InodeId) -> FsResult<Inode> {
        if !self.mounted {
            return Err(FsError::NotSupported);
        }

        let cluster = inode_id.0 as u32;

        // Special case for root
        if inode_id == InodeId(1) || cluster == self.boot_sector.root_cluster {
            return Ok(Inode::directory(InodeId(1)));
        }

        // We need to find the directory entry to get file info
        // This is a limitation - we'd need to track parent directories
        Ok(Inode {
            id: inode_id,
            file_type: FileType::Unknown,
            size: 0,
            mode: 0o644,
            nlink: 1,
            uid: 0,
            gid: 0,
            atime: 0,
            mtime: 0,
            ctime: 0,
            blksize: self.boot_sector.bytes_per_cluster(),
            blocks: 0,
        })
    }

    fn read(&self, inode_id: InodeId, offset: u64, buf: &mut [u8]) -> FsResult<usize> {
        if !self.mounted {
            return Err(FsError::NotSupported);
        }

        // Create inode - we'd need to track file size properly
        let inode = Fat32Inode::new(inode_id.0 as u32, u64::MAX, FileType::Regular, 0);
        self.read_file(&inode, offset, buf)
    }

    fn write(&self, _inode_id: InodeId, _offset: u64, _buf: &[u8]) -> FsResult<usize> {
        Err(FsError::ReadOnly)
    }

    fn truncate(&self, _inode_id: InodeId, _size: u64) -> FsResult<()> {
        Err(FsError::ReadOnly)
    }

    fn lookup(&self, dir_inode: InodeId, name: &str) -> FsResult<InodeId> {
        if !self.mounted {
            return Err(FsError::NotSupported);
        }

        let cluster = if dir_inode == InodeId(1) {
            self.boot_sector.root_cluster
        } else {
            dir_inode.0 as u32
        };

        let entry = self.find_entry(cluster, name)?;
        Ok(InodeId(entry.first_cluster() as u64))
    }

    fn create(
        &self,
        _dir_inode: InodeId,
        _name: &str,
        _file_type: FileType,
        _mode: u32,
    ) -> FsResult<InodeId> {
        Err(FsError::ReadOnly)
    }

    fn unlink(&self, _dir_inode: InodeId, _name: &str) -> FsResult<()> {
        Err(FsError::ReadOnly)
    }

    fn readdir(&self, dir_inode: InodeId, _offset: usize) -> FsResult<Vec<DirEntry>> {
        if !self.mounted {
            return Err(FsError::NotSupported);
        }

        let cluster = if dir_inode == InodeId(1) {
            self.boot_sector.root_cluster
        } else {
            dir_inode.0 as u32
        };

        let fat_entries = self.read_directory(cluster)?;
        let mut entries = Vec::new();

        for (fat_entry, name) in fat_entries {
            // Skip . and .. for VFS consistency
            if name == "." || name == ".." {
                continue;
            }

            entries.push(DirEntry::new(
                &name,
                InodeId(fat_entry.first_cluster() as u64),
                fat_entry.file_type(),
            ));
        }

        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fat32_dir_entry_parse() {
        let mut data = [0u8; 32];
        // "TEST    TXT" in 8.3 format
        data[0..11].copy_from_slice(b"TEST    TXT");
        data[11] = 0x20; // Archive attribute
        data[26..28].copy_from_slice(&100u16.to_le_bytes()); // Cluster low
        data[28..32].copy_from_slice(&1024u32.to_le_bytes()); // File size

        let entry = Fat32DirEntry::parse(&data).unwrap();
        assert!(!entry.is_directory());
        assert!(!entry.is_free());
        assert_eq!(entry.first_cluster(), 100);
        assert_eq!(entry.file_size, 1024);
    }

    #[test]
    fn test_fat32_dir_entry_matches() {
        let mut data = [0u8; 32];
        data[0..11].copy_from_slice(b"TEST    TXT");

        let entry = Fat32DirEntry::parse(&data).unwrap();
        assert!(entry.matches_name("TEST.TXT"));
        assert!(entry.matches_name("test.txt"));
        assert!(entry.matches_name("Test.Txt"));
        assert!(!entry.matches_name("TEST2.TXT"));
    }

    #[test]
    fn test_fat32_cluster_calculations() {
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
            volume_label: *b"TEST       ",
            fs_type: *b"FAT32   ",
        };

        assert_eq!(boot.bytes_per_cluster(), 4096);
        assert_eq!(boot.first_data_sector(), 32 + 2 * 2048);
        assert_eq!(boot.cluster_to_sector(2), boot.first_data_sector());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn test_fat32_short_name() {
        let mut data = [0u8; 32];

        // "README  TXT"
        data[0..11].copy_from_slice(b"README  TXT");
        let entry = Fat32DirEntry::parse(&data).unwrap();
        assert_eq!(entry.short_name(), "README.TXT");

        // "NOEXT   "
        data[0..11].copy_from_slice(b"NOEXT      ");
        let entry = Fat32DirEntry::parse(&data).unwrap();
        assert_eq!(entry.short_name(), "NOEXT");

        // Directory "SUBDIR  "
        data[0..11].copy_from_slice(b"SUBDIR     ");
        data[11] = ATTR_DIRECTORY;
        let entry = Fat32DirEntry::parse(&data).unwrap();
        assert!(entry.is_directory());
        assert_eq!(entry.short_name(), "SUBDIR");
    }

    #[test]
    fn test_fat32_special_clusters() {
        assert!(FAT32_EOC_MIN > FAT32_BAD_CLUSTER);
        assert_eq!(FAT32_FREE_CLUSTER, 0);
    }
}
