//! AArch64 exception handling
//!
//! This module provides:
//! - Exception vector table (in asm/vectors.S)
//! - Exception handlers for EL1
//! - System call dispatch
//! - Page fault handling

use crate::registers::{esr_el1_read, far_el1_read};

/// Exception Syndrome Register (ESR_EL1) exception classes
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExceptionClass {
    /// Unknown exception
    Unknown = 0x00,
    /// Trapped WFI/WFE
    WfiWfe = 0x01,
    /// Trapped MCR/MRC (CP15)
    Cp15McrMrc = 0x03,
    /// Trapped MCRR/MRRC (CP15)
    Cp15McrrMrrc = 0x04,
    /// Trapped MCR/MRC (CP14)
    Cp14McrMrc = 0x05,
    /// Trapped LDC/STC (CP14)
    Cp14LdcStc = 0x06,
    /// Trapped access to SIMD/FP
    SimdFp = 0x07,
    /// Trapped VMRS (CP10)
    Cp10Vmrs = 0x08,
    /// Trapped MRRC (CP14)
    Cp14Mrrc = 0x0C,
    /// Illegal Execution State
    IllegalState = 0x0E,
    /// SVC instruction at AArch32
    Svc32 = 0x11,
    /// SVC instruction at AArch64
    Svc64 = 0x15,
    /// Trapped MSR/MRS/System instruction
    MsrMrs = 0x18,
    /// Instruction Abort from lower EL
    InsnAbortLower = 0x20,
    /// Instruction Abort from same EL
    InsnAbortSame = 0x21,
    /// PC alignment fault
    PcAlignment = 0x22,
    /// Data Abort from lower EL
    DataAbortLower = 0x24,
    /// Data Abort from same EL
    DataAbortSame = 0x25,
    /// SP alignment fault
    SpAlignment = 0x26,
    /// Trapped floating point exception (AArch32)
    Fp32 = 0x28,
    /// Trapped floating point exception (AArch64)
    Fp64 = 0x2C,
    /// SError interrupt
    SError = 0x2F,
    /// Breakpoint from lower EL
    BreakpointLower = 0x30,
    /// Breakpoint from same EL
    BreakpointSame = 0x31,
    /// Software Step from lower EL
    SoftwareStepLower = 0x32,
    /// Software Step from same EL
    SoftwareStepSame = 0x33,
    /// Watchpoint from lower EL
    WatchpointLower = 0x34,
    /// Watchpoint from same EL
    WatchpointSame = 0x35,
    /// BRK instruction (AArch64)
    Brk = 0x3C,
}

impl ExceptionClass {
    /// Extract exception class from ESR_EL1
    pub fn from_esr(esr: u64) -> Self {
        let ec = ((esr >> 26) & 0x3F) as u8;
        match ec {
            0x00 => ExceptionClass::Unknown,
            0x01 => ExceptionClass::WfiWfe,
            0x03 => ExceptionClass::Cp15McrMrc,
            0x04 => ExceptionClass::Cp15McrrMrrc,
            0x05 => ExceptionClass::Cp14McrMrc,
            0x06 => ExceptionClass::Cp14LdcStc,
            0x07 => ExceptionClass::SimdFp,
            0x08 => ExceptionClass::Cp10Vmrs,
            0x0C => ExceptionClass::Cp14Mrrc,
            0x0E => ExceptionClass::IllegalState,
            0x11 => ExceptionClass::Svc32,
            0x15 => ExceptionClass::Svc64,
            0x18 => ExceptionClass::MsrMrs,
            0x20 => ExceptionClass::InsnAbortLower,
            0x21 => ExceptionClass::InsnAbortSame,
            0x22 => ExceptionClass::PcAlignment,
            0x24 => ExceptionClass::DataAbortLower,
            0x25 => ExceptionClass::DataAbortSame,
            0x26 => ExceptionClass::SpAlignment,
            0x28 => ExceptionClass::Fp32,
            0x2C => ExceptionClass::Fp64,
            0x2F => ExceptionClass::SError,
            0x30 => ExceptionClass::BreakpointLower,
            0x31 => ExceptionClass::BreakpointSame,
            0x32 => ExceptionClass::SoftwareStepLower,
            0x33 => ExceptionClass::SoftwareStepSame,
            0x34 => ExceptionClass::WatchpointLower,
            0x35 => ExceptionClass::WatchpointSame,
            0x3C => ExceptionClass::Brk,
            _ => ExceptionClass::Unknown,
        }
    }
}

/// Exception context (saved by assembly)
#[repr(C)]
pub struct ExceptionContext {
    /// General purpose registers x0-x30
    pub gpr: [u64; 31],
    /// Stack pointer
    pub sp: u64,
    /// Exception link register (return address)
    pub elr: u64,
    /// Saved program status register
    pub spsr: u64,
}

/// Synchronous exception handler (called from assembly)
///
/// # Safety
///
/// Must only be called from exception vector with valid context.
#[no_mangle]
pub unsafe extern "C" fn handle_sync_exception(ctx: &mut ExceptionContext) {
    // SAFETY: Reading ESR_EL1 is safe in exception handler
    let esr = unsafe { esr_el1_read() };
    let ec = ExceptionClass::from_esr(esr);

    match ec {
        ExceptionClass::Svc64 => {
            // System call
            handle_syscall(ctx, esr);
        }
        ExceptionClass::DataAbortSame | ExceptionClass::DataAbortLower => {
            // Page fault
            handle_page_fault(ctx, esr);
        }
        ExceptionClass::InsnAbortSame | ExceptionClass::InsnAbortLower => {
            // Instruction abort
            handle_insn_abort(ctx, esr);
        }
        _ => {
            // Unhandled exception
            panic_exception(ctx, esr, "Unhandled synchronous exception");
        }
    }
}

/// IRQ handler (called from assembly)
///
/// # Safety
///
/// Must only be called from exception vector with valid context.
#[no_mangle]
pub unsafe extern "C" fn handle_irq(_ctx: &mut ExceptionContext) {
    // TODO: Implement interrupt handling
    // For now, just return
}

/// FIQ handler (called from assembly)
///
/// # Safety
///
/// Must only be called from exception vector with valid context.
#[no_mangle]
pub unsafe extern "C" fn handle_fiq(_ctx: &mut ExceptionContext) {
    // TODO: Implement fast interrupt handling
    // For now, just return
}

/// SError handler (called from assembly)
///
/// # Safety
///
/// Must only be called from exception vector with valid context.
#[no_mangle]
pub unsafe extern "C" fn handle_serror(ctx: &mut ExceptionContext) {
    // SAFETY: Reading ESR_EL1 is safe in exception handler
    let esr = unsafe { esr_el1_read() };
    panic_exception(ctx, esr, "SError interrupt");
}

/// Handle system call
fn handle_syscall(ctx: &mut ExceptionContext, esr: u64) {
    let _syscall_num = esr & 0xFFFF;

    // TODO: Implement syscall dispatch based on syscall_num
    // For now, just return error
    ctx.gpr[0] = u64::MAX; // Return -1
}

/// Handle page fault
fn handle_page_fault(ctx: &mut ExceptionContext, esr: u64) {
    // SAFETY: Reading FAR_EL1 is safe in exception handler
    let _fault_addr = unsafe { far_el1_read() };

    // Determine if read or write
    let is_write = (esr & (1 << 6)) != 0;

    // TODO: Handle page fault (allocate page, update mappings, etc.)
    // For now, panic
    panic_exception(
        ctx,
        esr,
        if is_write {
            "Write page fault"
        } else {
            "Read page fault"
        },
    );
}

/// Handle instruction abort
fn handle_insn_abort(ctx: &mut ExceptionContext, esr: u64) {
    // SAFETY: Reading FAR_EL1 is safe in exception handler
    let _fault_addr = unsafe { far_el1_read() };

    // TODO: Handle instruction abort
    // For now, panic
    panic_exception(ctx, esr, "Instruction abort");
}

/// Panic on unhandled exception
fn panic_exception(ctx: &ExceptionContext, esr: u64, msg: &str) -> ! {
    // SAFETY: Reading FAR_EL1 is safe in exception handler
    let far = unsafe { far_el1_read() };
    let ec = ExceptionClass::from_esr(esr);

    panic!(
        "{}\nEC: {:?}\nESR: 0x{:016x}\nFAR: 0x{:016x}\nELR: 0x{:016x}\nSPSR: 0x{:016x}",
        msg, ec, esr, far, ctx.elr, ctx.spsr
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exception_class_from_esr() {
        // SVC64 (EC = 0x15)
        let esr = 0x15 << 26;
        assert_eq!(ExceptionClass::from_esr(esr), ExceptionClass::Svc64);

        // Data abort same EL (EC = 0x25)
        let esr = 0x25 << 26;
        assert_eq!(ExceptionClass::from_esr(esr), ExceptionClass::DataAbortSame);
    }
}
