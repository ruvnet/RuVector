//! The hot-path partition switch.
//!
//! This is the < 10 microsecond critical path from ADR-132.
//! **NO** allocation. **NO** graph work. **NO** policy evaluation.
//! Just: save registers -> update VTTBR -> flush TLB -> restore registers.
//!
//! ## Implementation status (HONESTY NOTE)
//!
//! The hardware hot path is **NOT implemented**:
//! [`HARDWARE_SWITCH_IMPLEMENTED`] is `false`. Actual register
//! manipulation (MRS/MSR sequences, the VTTBR_EL2 write, `TLBI
//! VMALLE1`, and `DSB ISH`/`ISB` barriers) requires EL2 inline assembly
//! and is deferred to the HAL crate.
//!
//! Consequently the "< 10 us partition switch" ADR-132 target is **not
//! validated** by anything in this crate or its benchmarks:
//!
//! - [`partition_switch`] is a validation-only stub. It checks the
//!   target context and returns; it performs **no** state transfer and
//!   reports `elapsed_ns = 0` ("not measured"). Benchmarking it
//!   measures the stub, not a partition switch.
//! - [`partition_switch_save_restore`] is the honest host-measurable
//!   path: it performs the real register-file save/restore data
//!   movement (the memory-traffic component of a switch) against a
//!   caller-provided register-file stand-in. It still excludes the
//!   privileged hardware operations, so its benchmark is a *lower
//!   bound* on switch cost, not a certification of the target.

use rvm_types::{RvmError, RvmResult};

/// Whether the privileged hardware switch sequence (register MRS/MSR,
/// VTTBR_EL2 update, TLB invalidate, barriers) is implemented.
///
/// `false`: the < 10 us ADR-132 switch target CANNOT be validated yet.
/// Any benchmark or report citing a measured partition-switch latency
/// must check this constant before claiming the target is met.
pub const HARDWARE_SWITCH_IMPLEMENTED: bool = false;

/// Saved register state for a partition context.
///
/// Captures the minimal AArch64 EL2-visible state required to resume
/// execution in a partition. The HAL populates these fields from the
/// actual hardware registers.
///
/// Cache-line aligned (`align(64)`) to prevent false sharing between
/// per-CPU switch contexts. Hot fields accessed during context switch
/// (`vttbr_el2`, `elr_el2`, `spsr_el2`, `sp_el1`) are placed first
/// to fit in the first cache line.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(64))]
pub struct SwitchContext {
    /// Stage-2 translation table base register (VTTBR_EL2).
    ///
    /// Encodes the VMID in bits \[55:48\] and the physical address of
    /// the stage-2 page table root in bits \[47:1\].
    pub vttbr_el2: u64,
    /// Exception Link Register for EL2 (return address).
    pub elr_el2: u64,
    /// Saved Program Status Register for EL2.
    pub spsr_el2: u64,
    /// Stack pointer for EL1 (SP_EL1).
    pub sp_el1: u64,
    /// General-purpose registers x0-x30 (cold path, accessed after
    /// the hot fields above).
    pub gp_regs: [u64; 31],
}

impl SwitchContext {
    /// Create a zeroed switch context.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            vttbr_el2: 0,
            elr_el2: 0,
            spsr_el2: 0,
            sp_el1: 0,
            gp_regs: [0u64; 31],
        }
    }

    /// Initialise this context with a given entry point, stack pointer,
    /// VMID, and stage-2 page table base.
    ///
    /// This prepares a context for first entry into a guest partition.
    pub fn init(
        &mut self,
        entry_point: u64,
        stack_pointer: u64,
        vmid: u16,
        s2_table_base: u64,
    ) {
        self.elr_el2 = entry_point;
        self.sp_el1 = stack_pointer;
        // AArch64 EL1h mode, all DAIF masked.
        self.spsr_el2 = 0x3C5;
        // VTTBR_EL2: VMID in [55:48], table base in [47:1].
        self.vttbr_el2 = ((vmid as u64) << 48) | (s2_table_base & 0x0000_FFFF_FFFF_FFFE);
    }

    /// Extract the VMID from the VTTBR_EL2 field.
    #[must_use]
    pub const fn vmid(&self) -> u16 {
        (self.vttbr_el2 >> 48) as u16
    }

    /// Extract the stage-2 table base address from VTTBR_EL2.
    #[must_use]
    pub const fn s2_table_base(&self) -> u64 {
        self.vttbr_el2 & 0x0000_FFFF_FFFF_FFFE
    }

    /// Save the current context from a source context.
    ///
    /// On AArch64 bare-metal, this would execute MRS instructions.
    /// For host builds and testing, this copies the fields from `src`
    /// to simulate a register save.
    pub fn save_from(&mut self, src: &SwitchContext) {
        *self = *src;
    }

    /// Check whether this context represents a valid entry point
    /// (non-zero ELR and VTTBR).
    #[must_use]
    pub const fn is_valid_entry(&self) -> bool {
        self.elr_el2 != 0 && self.vttbr_el2 != 0
    }

    /// Return the entry point address (ELR_EL2).
    #[must_use]
    pub const fn entry_point(&self) -> u64 {
        self.elr_el2
    }

    /// Hypervisor address space boundary.
    ///
    /// Addresses at or above this value belong to the hypervisor's
    /// own higher-half virtual address space and must never be used
    /// as a guest entry point.
    const HYPERVISOR_BASE: u64 = 0xFFFF_0000_0000_0000;

    /// Validate that this context is safe to switch into.
    ///
    /// Checks:
    /// 1. Entry point (ELR_EL2) is not zero.
    /// 2. Entry point is below the hypervisor address space boundary.
    ///
    /// Returns `Err(InvalidPartitionState)` if invalid.
    pub const fn validate_for_switch(&self) -> RvmResult<()> {
        if self.elr_el2 == 0 {
            return Err(RvmError::InvalidPartitionState);
        }
        if self.elr_el2 >= Self::HYPERVISOR_BASE {
            return Err(RvmError::InvalidPartitionState);
        }
        Ok(())
    }
}

/// Result of a partition switch, capturing both contexts and timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwitchResult {
    /// VMID of the partition we switched away from.
    pub from_vmid: u16,
    /// VMID of the partition we switched to.
    pub to_vmid: u16,
    /// Number of nanoseconds elapsed. **Always 0 on host builds**,
    /// meaning "not measured" -- never cite this field as evidence that
    /// the < 10 us switch target is met (see
    /// [`HARDWARE_SWITCH_IMPLEMENTED`]).
    pub elapsed_ns: u64,
}

/// Validation-only partition switch STUB.
///
/// The intended hot path is:
/// 1. Save current registers into `from` (on AArch64: MRS sequence).
/// 2. Write `to.vttbr_el2` to VTTBR_EL2 (stage-2 page table base).
/// 3. TLB invalidate (`TLBI VMALLE1`).
/// 4. Barrier (`DSB ISH` + `ISB`).
/// 5. Restore registers from `to`.
///
/// This function performs **only** the target-context validation; steps
/// 1-5 are stubs awaiting the rvm-hal assembly sequences
/// ([`HARDWARE_SWITCH_IMPLEMENTED`] is `false`). It exists so callers
/// can exercise the validation/bookkeeping contract. For the honest
/// host-measurable data-movement path, use
/// [`partition_switch_save_restore`].
#[inline]
pub fn partition_switch(from: &mut SwitchContext, to: &SwitchContext) -> RvmResult<SwitchResult> {
    // Validate the target context before switching.
    to.validate_for_switch()?;

    let from_vmid = from.vmid();
    let to_vmid = to.vmid();

    // Step 1: save current register state.
    // On host builds, `from` already holds the correct state (set by
    // the caller via `init()`). On AArch64, rvm-hal::context_switch
    // performs the actual MRS/MSR sequence.

    // Step 2: update VTTBR_EL2.
    // HAL stub: MSR VTTBR_EL2, to.vttbr_el2
    let _ = to.vttbr_el2; // ensure the field is "read" for the compiler

    // Step 3: TLB invalidate.
    // HAL stub: TLBI VMALLE1

    // Step 4: barrier.
    // HAL stub: DSB ISH; ISB

    // Step 5: restore target register state.
    // HAL stub: LDP x0, x1, ... from `to`

    Ok(SwitchResult {
        from_vmid,
        to_vmid,
        elapsed_ns: 0, // Not measured: hardware path unimplemented.
    })
}

/// Honest host-measurable partition switch path: performs the actual
/// register-file save and restore **data movement** of a context
/// switch.
///
/// `hw` is the stand-in for the CPU register file (on AArch64 this
/// state lives in hardware registers; on host builds it is a struct).
/// The function:
///
/// 1. Validates the target context.
/// 2. Saves the current register file into `from` (step 1 of the real
///    sequence, as an actual ~280-byte copy).
/// 3. Restores the target state into the register file (step 5, same).
///
/// Steps 2-4 (VTTBR_EL2 write, TLBI, barriers) are **excluded** -- they
/// require EL2 privilege. A benchmark of this function is therefore a
/// lower bound on real switch cost: it measures the software
/// data-movement floor only and does NOT certify the < 10 us target
/// (see [`HARDWARE_SWITCH_IMPLEMENTED`]).
#[inline]
pub fn partition_switch_save_restore(
    hw: &mut SwitchContext,
    from: &mut SwitchContext,
    to: &SwitchContext,
) -> RvmResult<SwitchResult> {
    to.validate_for_switch()?;

    let from_vmid = hw.vmid();
    let to_vmid = to.vmid();

    // Step 1 (real data movement): save the current register file.
    *from = *hw;
    // Steps 2-4: VTTBR_EL2 write, TLBI VMALLE1, DSB ISH + ISB --
    // NOT performed (EL2 only; rvm-hal).
    // Step 5 (real data movement): restore the target register file.
    *hw = *to;

    Ok(SwitchResult {
        from_vmid,
        to_vmid,
        elapsed_ns: 0, // Timing is the benchmark harness's job.
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_switch_context_new_is_zeroed() {
        let ctx = SwitchContext::new();
        assert_eq!(ctx.gp_regs, [0u64; 31]);
        assert_eq!(ctx.sp_el1, 0);
        assert_eq!(ctx.elr_el2, 0);
        assert_eq!(ctx.spsr_el2, 0);
        assert_eq!(ctx.vttbr_el2, 0);
    }

    #[test]
    fn test_init_sets_fields() {
        let mut ctx = SwitchContext::new();
        ctx.init(0x4000_0000, 0x8000, 0x01, 0x0000_1000_0000_0000);

        assert_eq!(ctx.elr_el2, 0x4000_0000);
        assert_eq!(ctx.sp_el1, 0x8000);
        assert_eq!(ctx.spsr_el2, 0x3C5); // EL1h, DAIF masked
        assert_eq!(ctx.vmid(), 0x01);
        assert_eq!(ctx.s2_table_base(), 0x0000_1000_0000_0000);
    }

    #[test]
    fn test_vmid_extraction() {
        let mut ctx = SwitchContext::new();
        ctx.vttbr_el2 = 0x0042_0000_0000_0000; // VMID = 0x42
        assert_eq!(ctx.vmid(), 0x42);
    }

    #[test]
    fn test_s2_table_base_extraction() {
        let mut ctx = SwitchContext::new();
        ctx.vttbr_el2 = 0x00FF_0000_DEAD_BEE0;
        assert_eq!(ctx.s2_table_base(), 0x0000_0000_DEAD_BEE0);
    }

    #[test]
    fn test_is_valid_entry() {
        let ctx = SwitchContext::new();
        assert!(!ctx.is_valid_entry());

        let mut ctx2 = SwitchContext::new();
        ctx2.init(0x4000_0000, 0x8000, 1, 0x1000);
        assert!(ctx2.is_valid_entry());
    }

    #[test]
    fn test_save_from_copies_state() {
        let mut src = SwitchContext::new();
        src.gp_regs[0] = 0xCAFE;
        src.sp_el1 = 0x1000;
        src.elr_el2 = 0x2000;
        src.spsr_el2 = 0x3C5;
        src.vttbr_el2 = 0xDEAD_0000;

        let mut dst = SwitchContext::new();
        dst.save_from(&src);

        assert_eq!(dst.gp_regs[0], 0xCAFE);
        assert_eq!(dst.sp_el1, 0x1000);
        assert_eq!(dst.elr_el2, 0x2000);
        assert_eq!(dst.vttbr_el2, 0xDEAD_0000);
    }

    #[test]
    fn test_switch_preserves_contexts() {
        let mut from = SwitchContext::new();
        from.init(0x4000_0000, 0x8000, 1, 0x0001_0000_0000_0000);

        let mut to = SwitchContext::new();
        to.init(0x8000_0000, 0xF000, 2, 0x0002_0000_0000_0000);

        let result = partition_switch(&mut from, &to).unwrap();

        assert_eq!(result.from_vmid, 1);
        assert_eq!(result.to_vmid, 2);
        assert_eq!(result.elapsed_ns, 0);

        // Both contexts should be unchanged.
        assert_eq!(from.elr_el2, 0x4000_0000);
        assert_eq!(to.elr_el2, 0x8000_0000);
    }

    #[test]
    fn test_partition_switch_returns_vmids() {
        let mut a = SwitchContext::new();
        a.init(0x1000, 0x2000, 0x0A, 0x0001_0000_0000_0000);

        let mut b = SwitchContext::new();
        b.init(0x3000, 0x4000, 0x0B, 0x0002_0000_0000_0000);

        let result = partition_switch(&mut a, &b).unwrap();
        assert_eq!(result.from_vmid, 0x0A);
        assert_eq!(result.to_vmid, 0x0B);
    }

    #[test]
    fn test_switch_rejects_zero_entry_point() {
        let mut from = SwitchContext::new();
        from.init(0x4000_0000, 0x8000, 1, 0x0001_0000_0000_0000);

        let to = SwitchContext::new(); // elr_el2 = 0
        assert_eq!(
            partition_switch(&mut from, &to),
            Err(RvmError::InvalidPartitionState)
        );
    }

    #[test]
    fn test_switch_rejects_hypervisor_address() {
        let mut from = SwitchContext::new();
        from.init(0x4000_0000, 0x8000, 1, 0x0001_0000_0000_0000);

        let mut to = SwitchContext::new();
        // Entry point in hypervisor address space.
        to.init(0xFFFF_0000_0000_1000, 0x8000, 2, 0x0002_0000_0000_0000);
        assert_eq!(
            partition_switch(&mut from, &to),
            Err(RvmError::InvalidPartitionState)
        );
    }

    /// Honesty canary: if the hardware switch path ever gets
    /// implemented, this test fails and forces the <10us claim, the
    /// benchmarks, and the docs above to be revisited together.
    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_hardware_switch_is_documented_as_unimplemented() {
        assert!(
            !HARDWARE_SWITCH_IMPLEMENTED,
            "hardware switch now implemented: update the <10us claim, \
             benches, and module docs"
        );
    }

    #[test]
    fn test_save_restore_moves_state() {
        let mut hw = SwitchContext::new();
        hw.init(0x4000_0000, 0x8000, 1, 0x0001_0000_0000_0000);
        hw.gp_regs[0] = 0xAAAA;

        let mut from = SwitchContext::new();
        let mut to = SwitchContext::new();
        to.init(0x8000_0000, 0xF000, 2, 0x0002_0000_0000_0000);
        to.gp_regs[0] = 0xBBBB;

        let result = partition_switch_save_restore(&mut hw, &mut from, &to).unwrap();
        assert_eq!(result.from_vmid, 1);
        assert_eq!(result.to_vmid, 2);

        // Step 1: the outgoing state was saved into `from`.
        assert_eq!(from.vmid(), 1);
        assert_eq!(from.gp_regs[0], 0xAAAA);
        assert_eq!(from.elr_el2, 0x4000_0000);
        // Step 5: the register file now holds the target state.
        assert_eq!(hw.vmid(), 2);
        assert_eq!(hw.gp_regs[0], 0xBBBB);
        assert_eq!(hw.elr_el2, 0x8000_0000);
    }

    #[test]
    fn test_save_restore_rejects_invalid_target() {
        let mut hw = SwitchContext::new();
        hw.init(0x4000_0000, 0x8000, 1, 0x0001_0000_0000_0000);
        let mut from = SwitchContext::new();
        let to = SwitchContext::new(); // elr_el2 = 0: invalid

        assert_eq!(
            partition_switch_save_restore(&mut hw, &mut from, &to),
            Err(RvmError::InvalidPartitionState)
        );
        // Nothing was clobbered on rejection.
        assert_eq!(hw.vmid(), 1);
        assert_eq!(from.elr_el2, 0);
    }

    #[test]
    fn test_switch_is_repeatable() {
        let mut from = SwitchContext::new();
        from.init(0x4000_0000, 0x8000, 1, 0x0001_0000_0000_0000);

        let mut to = SwitchContext::new();
        to.init(0x8000_0000, 0xF000, 2, 0x0002_0000_0000_0000);

        let r1 = partition_switch(&mut from, &to).unwrap();
        let r2 = partition_switch(&mut from, &to).unwrap();
        assert_eq!(r1.elapsed_ns, r2.elapsed_ns);
    }
}
