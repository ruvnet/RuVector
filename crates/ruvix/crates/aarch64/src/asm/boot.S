/*
 * AArch64 boot assembly
 *
 * This is the first code executed when the kernel starts.
 * It sets up the execution environment and jumps to Rust code.
 */

.section .text.boot
.global _start

_start:
    /* Disable interrupts (DAIF: Debug, SError, IRQ, FIQ) */
    msr daifset, #0xf

    /* Check exception level */
    mrs x0, CurrentEL
    and x0, x0, #0xC  /* Mask to get EL bits */
    cmp x0, #8        /* Check if EL2 (0b10 << 2 = 8) */
    b.eq from_el2
    cmp x0, #12       /* Check if EL3 (0b11 << 2 = 12) */
    b.eq from_el3

from_el1:
    /* Already at EL1, continue boot */
    b setup_stack

from_el2:
    /* Drop from EL2 to EL1 */
    /* Configure HCR_EL2 for EL1 to use AArch64 */
    mov x0, #(1 << 31)  /* RW bit: EL1 is AArch64 */
    msr hcr_el2, x0

    /* Set SPSR_EL2 for EL1h (EL1 with SP_EL1) */
    mov x0, #0x5        /* 0b0101: EL1h, IRQ/FIQ masked */
    msr spsr_el2, x0

    /* Set ELR_EL2 to return address */
    adr x0, setup_stack
    msr elr_el2, x0

    /* Return to EL1 */
    eret

from_el3:
    /* Drop from EL3 to EL1 */
    /* Configure SCR_EL3 */
    mov x0, #(1 << 10)  /* RW bit: lower levels are AArch64 */
    orr x0, x0, #(1 << 0)  /* NS bit: Non-secure */
    msr scr_el3, x0

    /* Set SPSR_EL3 for EL1h */
    mov x0, #0x5        /* 0b0101: EL1h, IRQ/FIQ masked */
    msr spsr_el3, x0

    /* Set ELR_EL3 to return address */
    adr x0, setup_stack
    msr elr_el3, x0

    /* Return to EL1 */
    eret

setup_stack:
    /* Set up stack pointer */
    adr x0, __stack_top
    mov sp, x0

    /* Clear frame pointer */
    mov x29, xzr

    /* Clear BSS section */
    bl __clear_bss

    /* Jump to Rust early_init */
    bl early_init

    /* Should never return, but halt if it does */
1:
    wfi
    b 1b

/*
 * Clear BSS section
 * Called from setup_stack before jumping to Rust
 */
__clear_bss:
    /* Load BSS start and end */
    adr x0, __bss_start
    adr x1, __bss_end

    /* Calculate size */
    sub x2, x1, x0
    cbz x2, 2f  /* Skip if BSS is empty */

1:
    /* Clear 8 bytes at a time */
    str xzr, [x0], #8
    subs x2, x2, #8
    b.hi 1b

2:
    ret

/*
 * Stack definition
 * Defined in .bss section, 64KB stack
 */
.section .bss.stack
.align 16
__stack_bottom:
    .space 65536  /* 64KB stack */
__stack_top:
