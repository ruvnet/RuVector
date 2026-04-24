//! # Memory Management Unit Abstraction
//!
//! Provides traits for managing virtual memory and page tables, essential
//! for capability-aware memory isolation.
//!
//! ## Design
//!
//! - **4 KiB pages** - Standard ARM64 page size
//! - **Capability-tagged entries** - Store capability metadata in page tables
//! - **Multi-level page tables** - Support for 48-bit virtual addresses
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvix_hal::{Mmu, mmu::{PagePermissions, MmuError}};
//!
//! fn map_stack<M: Mmu>(mmu: &mut M, stack_top: u64, phys_base: u64) -> Result<(), MmuError> {
//!     const STACK_SIZE: usize = 16 * 4096; // 64 KiB
//!
//!     for i in 0..(STACK_SIZE / 4096) {
//!         let virt = stack_top - ((i + 1) * 4096) as u64;
//!         let phys = phys_base + (i * 4096) as u64;
//!
//!         mmu.map_page(
//!             virt,
//!             phys,
//!             PagePermissions::READ | PagePermissions::WRITE,
//!         )?;
//!     }
//!
//!     Ok(())
//! }
//! ```

use ruvix_types::Capability;

/// MMU error types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmuError {
    /// Address is not page-aligned
    NotPageAligned,
    /// Page is already mapped
    AlreadyMapped,
    /// Page is not mapped
    NotMapped,
    /// Invalid virtual address
    InvalidVirtualAddress,
    /// Invalid physical address
    InvalidPhysicalAddress,
    /// Invalid page permissions
    InvalidPermissions,
    /// Out of page table memory
    OutOfMemory,
    /// Capability check failed
    CapabilityDenied,
}

impl core::fmt::Display for MmuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotPageAligned => write!(f, "address not page-aligned"),
            Self::AlreadyMapped => write!(f, "page already mapped"),
            Self::NotMapped => write!(f, "page not mapped"),
            Self::InvalidVirtualAddress => write!(f, "invalid virtual address"),
            Self::InvalidPhysicalAddress => write!(f, "invalid physical address"),
            Self::InvalidPermissions => write!(f, "invalid page permissions"),
            Self::OutOfMemory => write!(f, "out of page table memory"),
            Self::CapabilityDenied => write!(f, "capability check failed"),
        }
    }
}

/// Page permissions (bitflags)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagePermissions(u8);

impl PagePermissions {
    /// No permissions
    pub const NONE: Self = Self(0);
    /// Read permission
    pub const READ: Self = Self(1 << 0);
    /// Write permission
    pub const WRITE: Self = Self(1 << 1);
    /// Execute permission
    pub const EXECUTE: Self = Self(1 << 2);
    /// User-accessible (vs kernel-only)
    pub const USER: Self = Self(1 << 3);
    /// Global mapping (not flushed on context switch)
    pub const GLOBAL: Self = Self(1 << 4);
    /// Non-cacheable
    pub const DEVICE: Self = Self(1 << 5);

    /// Read-only (kernel)
    pub const KERNEL_RO: Self = Self(Self::READ.0);
    /// Read-write (kernel)
    pub const KERNEL_RW: Self = Self(Self::READ.0 | Self::WRITE.0);
    /// Read-execute (kernel)
    pub const KERNEL_RX: Self = Self(Self::READ.0 | Self::EXECUTE.0);

    /// Read-only (user)
    pub const USER_RO: Self = Self(Self::READ.0 | Self::USER.0);
    /// Read-write (user)
    pub const USER_RW: Self = Self(Self::READ.0 | Self::WRITE.0 | Self::USER.0);
    /// Read-execute (user)
    pub const USER_RX: Self = Self(Self::READ.0 | Self::EXECUTE.0 | Self::USER.0);

    /// Check if a permission is set
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Get raw permission bits
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Create from raw bits
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }
}

impl core::ops::BitOr for PagePermissions {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl core::ops::BitAnd for PagePermissions {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

/// Page table entry metadata
#[derive(Debug, Clone, Copy)]
pub struct PageEntry {
    /// Physical address
    pub physical: u64,
    /// Permissions
    pub permissions: PagePermissions,
    /// Associated capability (optional)
    pub capability: Option<Capability>,
}

/// Memory Management Unit abstraction
///
/// This trait provides virtual memory management for ARM64 MMU and similar.
///
/// ## Page Size
///
/// All implementations MUST use 4 KiB pages (0x1000 bytes).
///
/// ## Thread Safety
///
/// MMU operations are NOT IRQ-safe. Disable interrupts when modifying
/// page tables.
///
/// ## Example Implementation (ARM64 MMU)
///
/// ```rust,ignore
/// use ruvix_hal::{Mmu, mmu::*};
/// use ruvix_types::Capability;
///
/// struct ArmMmu {
///     ttbr0: u64, // Translation table base register 0
///     ttbr1: u64, // Translation table base register 1
/// }
///
/// impl ArmMmu {
///     fn new() -> Self {
///         // Allocate page table roots
///         let ttbr0 = allocate_page_table();
///         let ttbr1 = allocate_page_table();
///
///         Self { ttbr0, ttbr1 }
///     }
///
///     fn activate(&self) {
///         // Write TTBR0_EL1 and TTBR1_EL1
///         unsafe {
///             core::arch::asm!(
///                 "msr ttbr0_el1, {ttbr0}",
///                 "msr ttbr1_el1, {ttbr1}",
///                 "isb",
///                 ttbr0 = in(reg) self.ttbr0,
///                 ttbr1 = in(reg) self.ttbr1,
///             );
///         }
///     }
///
///     fn walk_page_table(&self, vaddr: u64) -> Option<&mut PageTableEntry> {
///         // Implement 4-level page table walk
///         // Level 0: bits [47:39]
///         // Level 1: bits [38:30]
///         // Level 2: bits [29:21]
///         // Level 3: bits [20:12]
///         // Offset:  bits [11:0]
///         None // Simplified
///     }
/// }
///
/// impl Mmu for ArmMmu {
///     fn map_page(
///         &mut self,
///         virt: u64,
///         phys: u64,
///         perms: PagePermissions,
///     ) -> Result<(), MmuError> {
///         if virt & 0xFFF != 0 || phys & 0xFFF != 0 {
///             return Err(MmuError::NotPageAligned);
///         }
///
///         // Walk page table to get PTE
///         let pte = self.walk_page_table(virt)
///             .ok_or(MmuError::OutOfMemory)?;
///
///         // Check if already mapped
///         if pte.is_valid() {
///             return Err(MmuError::AlreadyMapped);
///         }
///
///         // Encode permissions in ARM64 format
///         let mut attr = phys & !0xFFF; // Physical address
///         attr |= 0b11; // Valid + table/block entry
///
///         if perms.contains(PagePermissions::READ) {
///             attr |= (0b00 << 6); // AP[2:1] = 00 (RW kernel)
///         }
///         if perms.contains(PagePermissions::USER) {
///             attr |= (0b01 << 6); // AP[2:1] = 01 (RW user)
///         }
///         if !perms.contains(PagePermissions::EXECUTE) {
///             attr |= (1 << 54); // UXN/PXN
///         }
///
///         // Write PTE
///         unsafe {
///             core::ptr::write_volatile(pte as *mut _ as *mut u64, attr);
///         }
///
///         // Invalidate TLB entry
///         unsafe {
///             core::arch::asm!(
///                 "tlbi vaae1is, {vaddr}",
///                 "dsb ish",
///                 "isb",
///                 vaddr = in(reg) virt >> 12,
///             );
///         }
///
///         Ok(())
///     }
///
///     fn unmap_page(&mut self, virt: u64) -> Result<(), MmuError> {
///         if virt & 0xFFF != 0 {
///             return Err(MmuError::NotPageAligned);
///         }
///
///         let pte = self.walk_page_table(virt)
///             .ok_or(MmuError::NotMapped)?;
///
///         if !pte.is_valid() {
///             return Err(MmuError::NotMapped);
///         }
///
///         // Clear PTE
///         unsafe {
///             core::ptr::write_volatile(pte as *mut _ as *mut u64, 0);
///         }
///
///         // Invalidate TLB
///         unsafe {
///             core::arch::asm!(
///                 "tlbi vaae1is, {vaddr}",
///                 "dsb ish",
///                 "isb",
///                 vaddr = in(reg) virt >> 12,
///             );
///         }
///
///         Ok(())
///     }
///
///     fn set_permissions(&mut self, virt: u64, perms: PagePermissions) -> Result<(), MmuError> {
///         // Similar to map_page but modify existing PTE
///         Ok(())
///     }
///
///     fn translate(&self, virt: u64) -> Result<u64, MmuError> {
///         let pte = self.walk_page_table(virt)
///             .ok_or(MmuError::NotMapped)?;
///
///         if !pte.is_valid() {
///             return Err(MmuError::NotMapped);
///         }
///
///         let offset = virt & 0xFFF;
///         let phys_base = pte.physical_address();
///         Ok(phys_base | offset)
///     }
///
///     fn get_page_entry(&self, virt: u64) -> Result<PageEntry, MmuError> {
///         let pte = self.walk_page_table(virt)
///             .ok_or(MmuError::NotMapped)?;
///
///         if !pte.is_valid() {
///             return Err(MmuError::NotMapped);
///         }
///
///         Ok(PageEntry {
///             physical: pte.physical_address(),
///             permissions: pte.decode_permissions(),
///             capability: pte.get_capability(),
///         })
///     }
///
///     fn flush_tlb(&mut self) {
///         unsafe {
///             core::arch::asm!(
///                 "tlbi vmalle1is",
///                 "dsb ish",
///                 "isb",
///             );
///         }
///     }
/// }
/// ```
pub trait Mmu {
    /// Map a virtual page to a physical page
    ///
    /// # Arguments
    ///
    /// * `virt` - Virtual address (must be page-aligned)
    /// * `phys` - Physical address (must be page-aligned)
    /// * `perms` - Page permissions
    ///
    /// # Errors
    ///
    /// - `NotPageAligned` if addresses are not 4 KiB aligned
    /// - `AlreadyMapped` if virtual page is already mapped
    /// - `OutOfMemory` if page table allocation fails
    fn map_page(&mut self, virt: u64, phys: u64, perms: PagePermissions) -> Result<(), MmuError>;

    /// Unmap a virtual page
    ///
    /// # Arguments
    ///
    /// * `virt` - Virtual address (must be page-aligned)
    ///
    /// # Errors
    ///
    /// - `NotPageAligned` if address is not 4 KiB aligned
    /// - `NotMapped` if page is not currently mapped
    fn unmap_page(&mut self, virt: u64) -> Result<(), MmuError>;

    /// Change permissions of an existing mapping
    ///
    /// # Arguments
    ///
    /// * `virt` - Virtual address (must be page-aligned)
    /// * `perms` - New permissions
    ///
    /// # Errors
    ///
    /// - `NotPageAligned` if address is not 4 KiB aligned
    /// - `NotMapped` if page is not currently mapped
    fn set_permissions(&mut self, virt: u64, perms: PagePermissions) -> Result<(), MmuError>;

    /// Translate virtual address to physical address
    ///
    /// # Arguments
    ///
    /// * `virt` - Virtual address (any alignment)
    ///
    /// # Returns
    ///
    /// Physical address corresponding to `virt`
    ///
    /// # Errors
    ///
    /// - `NotMapped` if page containing `virt` is not mapped
    fn translate(&self, virt: u64) -> Result<u64, MmuError>;

    /// Get page table entry for a virtual address
    ///
    /// # Arguments
    ///
    /// * `virt` - Virtual address (must be page-aligned)
    ///
    /// # Returns
    ///
    /// Page entry with physical address, permissions, and capability
    ///
    /// # Errors
    ///
    /// - `NotPageAligned` if address is not 4 KiB aligned
    /// - `NotMapped` if page is not currently mapped
    fn get_page_entry(&self, virt: u64) -> Result<PageEntry, MmuError>;

    /// Flush TLB (translation lookaside buffer)
    ///
    /// This MUST be called after modifying page tables to ensure
    /// consistency.
    fn flush_tlb(&mut self);

    /// Flush TLB entry for a specific virtual address
    ///
    /// More efficient than full `flush_tlb()` when modifying single page.
    ///
    /// # Arguments
    ///
    /// * `virt` - Virtual address (must be page-aligned)
    fn flush_tlb_entry(&mut self, virt: u64) {
        // Default: flush entire TLB
        let _ = virt;
        self.flush_tlb();
    }

    /// Map a page with an associated capability
    ///
    /// # Arguments
    ///
    /// * `virt` - Virtual address
    /// * `phys` - Physical address
    /// * `perms` - Page permissions
    /// * `cap` - Capability to associate with page
    ///
    /// # Errors
    ///
    /// Same as `map_page`, plus `CapabilityDenied` if capability is invalid.
    #[allow(unused_variables)]
    fn map_page_with_capability(
        &mut self,
        virt: u64,
        phys: u64,
        perms: PagePermissions,
        cap: Capability,
    ) -> Result<(), MmuError> {
        // Default: ignore capability
        self.map_page(virt, phys, perms)
    }

    /// Check if a virtual address is mapped
    ///
    /// # Arguments
    ///
    /// * `virt` - Virtual address (any alignment)
    fn is_mapped(&self, virt: u64) -> bool {
        self.translate(virt).is_ok()
    }

    /// Get page size in bytes
    ///
    /// Always returns 4096 (4 KiB) for ARM64.
    fn page_size(&self) -> usize {
        4096
    }

    /// Align address down to page boundary
    fn align_down(&self, addr: u64) -> u64 {
        addr & !(self.page_size() as u64 - 1)
    }

    /// Align address up to page boundary
    fn align_up(&self, addr: u64) -> u64 {
        (addr + (self.page_size() as u64 - 1)) & !(self.page_size() as u64 - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_permissions() {
        let perms = PagePermissions::READ | PagePermissions::WRITE;
        assert!(perms.contains(PagePermissions::READ));
        assert!(perms.contains(PagePermissions::WRITE));
        assert!(!perms.contains(PagePermissions::EXECUTE));

        let kernel_rw = PagePermissions::KERNEL_RW;
        assert!(kernel_rw.contains(PagePermissions::READ));
        assert!(kernel_rw.contains(PagePermissions::WRITE));
        assert!(!kernel_rw.contains(PagePermissions::USER));
    }

    #[test]
    fn test_permission_bits() {
        let perms = PagePermissions::USER_RX;
        let bits = perms.bits();
        let restored = PagePermissions::from_bits(bits);
        assert_eq!(perms, restored);
    }
}
