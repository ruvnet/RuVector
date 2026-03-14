//! AArch64 Memory Management Unit (MMU) implementation
//!
//! This module provides:
//! - 4KB page granule with 4-level page tables (48-bit VA)
//! - TTBR0_EL1 for user space (0x0000_0000_0000_0000 - 0x0000_FFFF_FFFF_FFFF)
//! - TTBR1_EL1 for kernel space (0xFFFF_0000_0000_0000 - 0xFFFF_FFFF_FFFF_FFFF)
//! - Memory attributes for Device and Normal memory
//! - Integration with RuVix capability-based permissions

use crate::registers::{set_mair_el1, set_tcr_el1, set_ttbr0_el1, set_ttbr1_el1};
use crate::PAGE_SIZE;
use ruvix_hal::mmu::{MmuError, PagePermissions};
use ruvix_hal::Mmu as MmuTrait;

/// Memory attribute index for Device-nGnRnE memory
const MAIR_DEVICE_nGnRnE: u8 = 0;

/// Memory attribute index for Normal memory (Write-Back)
const MAIR_NORMAL_WB: u8 = 1;

/// Memory attribute index for Normal memory (Write-Through)
const MAIR_NORMAL_WT: u8 = 2;

/// Page table entry flags
mod pte_flags {
    /// Valid entry
    pub const VALID: u64 = 1 << 0;
    /// Page (vs block)
    pub const PAGE: u64 = 1 << 1;
    /// User accessible
    pub const USER: u64 = 1 << 6;
    /// Read-only
    pub const RO: u64 = 1 << 7;
    /// Shareable
    pub const SHAREABLE: u64 = 2 << 8;
    /// Access flag
    pub const AF: u64 = 1 << 10;
    /// Not executable
    pub const XN: u64 = 1 << 54;
    /// Privileged execute never
    pub const PXN: u64 = 1 << 53;
}

/// AArch64 MMU implementation
pub struct Mmu {
    /// Level 0 page table for kernel (TTBR1_EL1)
    kernel_l0: [u64; 512],
    /// Level 0 page table for user (TTBR0_EL1)
    user_l0: [u64; 512],
}

impl Mmu {
    /// Create a new MMU instance
    pub const fn new() -> Self {
        Self {
            kernel_l0: [0; 512],
            user_l0: [0; 512],
        }
    }

    /// Initialize the MMU with default mappings
    ///
    /// # Safety
    ///
    /// Must be called during boot before MMU is enabled.
    pub unsafe fn init(&mut self) {
        // SAFETY: Called during boot before MMU is enabled, in correct order
        unsafe {
            // Configure Memory Attribute Indirection Register
            self.configure_mair();

            // Configure Translation Control Register
            self.configure_tcr();

            // Set up page tables
            self.setup_page_tables();

            // Install page table base registers
            self.install_page_tables();
        }
    }

    /// Configure MAIR_EL1 with memory attributes
    unsafe fn configure_mair(&self) {
        let mair: u64 = (0x00 << (MAIR_DEVICE_nGnRnE * 8))  // Device-nGnRnE
            | (0xFF << (MAIR_NORMAL_WB * 8))      // Normal, Write-Back
            | (0xBB << (MAIR_NORMAL_WT * 8)); // Normal, Write-Through

        // SAFETY: Configuring MAIR during boot
        unsafe {
            set_mair_el1(mair);
        }
    }

    /// Configure TCR_EL1 for 4KB granule, 48-bit VA
    unsafe fn configure_tcr(&self) {
        let tcr: u64 = (16 << 0) // T0SZ = 16 (48-bit VA for TTBR0)
            | (16 << 16)            // T1SZ = 16 (48-bit VA for TTBR1)
            | (0 << 14)             // TG0 = 4KB granule for TTBR0
            | (2 << 30)             // TG1 = 4KB granule for TTBR1
            | (1 << 8)              // IRGN0 = Normal, Inner Write-Back
            | (1 << 24)             // IRGN1 = Normal, Inner Write-Back
            | (1 << 10)             // ORGN0 = Normal, Outer Write-Back
            | (1 << 26)             // ORGN1 = Normal, Outer Write-Back
            | (3 << 12)             // SH0 = Inner Shareable
            | (3 << 28)             // SH1 = Inner Shareable
            | (2 << 32);            // IPS = 48-bit physical address

        // SAFETY: Configuring TCR during boot
        unsafe {
            set_tcr_el1(tcr);
        }
    }

    /// Set up identity mapping for kernel
    unsafe fn setup_page_tables(&mut self) {
        // For now, create identity mapping for first 1GB
        // This is sufficient for QEMU virt platform
        // TODO: Implement proper page table walk for dynamic mappings

        // Level 0 entry points to level 1 table
        // (In production, would allocate level 1 table dynamically)
    }

    /// Install page table base registers
    unsafe fn install_page_tables(&self) {
        let ttbr0 = self.user_l0.as_ptr() as u64;
        let ttbr1 = self.kernel_l0.as_ptr() as u64;

        // SAFETY: Setting page table base registers during boot
        unsafe {
            set_ttbr0_el1(ttbr0);
            set_ttbr1_el1(ttbr1);
        }
    }

    /// Convert page permissions to page table entry flags
    fn permissions_to_flags(perms: PagePermissions) -> u64 {
        let mut flags = pte_flags::VALID | pte_flags::AF;

        // Read/Write/Execute permissions
        if !perms.contains(PagePermissions::WRITE) {
            flags |= pte_flags::RO;
        }
        if !perms.contains(PagePermissions::EXECUTE) {
            flags |= pte_flags::XN;
        }

        // User vs kernel
        if perms.contains(PagePermissions::USER) {
            flags |= pte_flags::USER;
        } else {
            flags |= pte_flags::PXN; // Privileged execute never for kernel pages
        }

        // Memory attributes
        if perms.contains(PagePermissions::DEVICE) {
            flags |= (MAIR_DEVICE_nGnRnE as u64) << 2; // Device memory
        } else {
            flags |= (MAIR_NORMAL_WB as u64) << 2; // Normal WB by default
        }

        flags
    }
}

impl MmuTrait for Mmu {
    fn map_page(
        &mut self,
        virt: u64,
        phys: u64,
        perms: PagePermissions,
    ) -> Result<(), MmuError> {
        // Validate alignment
        if virt & (PAGE_SIZE as u64 - 1) != 0 {
            return Err(MmuError::NotPageAligned);
        }
        if phys & (PAGE_SIZE as u64 - 1) != 0 {
            return Err(MmuError::NotPageAligned);
        }

        // Convert permissions to PTE flags
        let _flags = Self::permissions_to_flags(perms);

        // TODO: Walk page tables and install mapping

        Ok(())
    }

    fn unmap_page(&mut self, virt: u64) -> Result<(), MmuError> {
        if virt & (PAGE_SIZE as u64 - 1) != 0 {
            return Err(MmuError::NotPageAligned);
        }

        // TODO: Walk page tables and remove mapping

        Ok(())
    }

    fn set_permissions(&mut self, _virt: u64, _perms: PagePermissions) -> Result<(), MmuError> {
        if _virt & (PAGE_SIZE as u64 - 1) != 0 {
            return Err(MmuError::NotPageAligned);
        }

        // TODO: Walk page tables and update PTE flags

        Ok(())
    }

    fn translate(&self, virt: u64) -> Result<u64, MmuError> {
        // TODO: Walk page tables to translate
        // For now, return error
        Err(MmuError::NotMapped)
    }

    fn get_page_entry(&self, virt: u64) -> Result<ruvix_hal::mmu::PageEntry, MmuError> {
        if virt & (PAGE_SIZE as u64 - 1) != 0 {
            return Err(MmuError::NotPageAligned);
        }

        // TODO: Walk page tables and get entry
        Err(MmuError::NotMapped)
    }

    fn flush_tlb(&mut self) {
        // SAFETY: Flushing TLB is safe and required after page table modifications
        unsafe {
            core::arch::asm!(
                "tlbi vmalle1is",  // Invalidate all TLB entries for EL1
                "dsb ish",         // Data synchronization barrier
                "isb",             // Instruction synchronization barrier
                options(nostack, nomem, preserves_flags)
            );
        }
    }
}

impl Default for Mmu {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mair_values() {
        assert_eq!(MAIR_DEVICE_nGnRnE, 0);
        assert_eq!(MAIR_NORMAL_WB, 1);
        assert_eq!(MAIR_NORMAL_WT, 2);
    }

    #[test]
    fn test_permissions_to_flags() {
        let read_only = PagePermissions::READ;
        let flags = Mmu::permissions_to_flags(read_only);
        assert!(flags & pte_flags::RO != 0);
        assert!(flags & pte_flags::XN != 0);

        let read_write_exec = PagePermissions::READ | PagePermissions::WRITE | PagePermissions::EXECUTE;
        let flags = Mmu::permissions_to_flags(read_write_exec);
        assert!(flags & pte_flags::RO == 0);
        assert!(flags & pte_flags::XN == 0);
    }
}
