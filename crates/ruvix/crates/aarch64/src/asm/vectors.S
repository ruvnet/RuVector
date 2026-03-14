/*
 * AArch64 exception vector table
 *
 * The vector table must be 2KB (0x800) aligned.
 * Each entry is 128 bytes (0x80), with 16 entries total:
 * - 4 exception types (Sync, IRQ, FIQ, SError)
 * - 4 sources (Current EL SP0, Current EL SPx, Lower EL AArch64, Lower EL AArch32)
 */

.section .text.vectors
.balign 0x800
.global exception_vectors

exception_vectors:

/*
 * Current EL with SP0
 */
.balign 0x80
current_el_sp0_sync:
    b handle_exception_wrapper
.balign 0x80
current_el_sp0_irq:
    b handle_irq_wrapper
.balign 0x80
current_el_sp0_fiq:
    b handle_fiq_wrapper
.balign 0x80
current_el_sp0_serror:
    b handle_serror_wrapper

/*
 * Current EL with SPx
 */
.balign 0x80
current_el_spx_sync:
    b handle_exception_wrapper
.balign 0x80
current_el_spx_irq:
    b handle_irq_wrapper
.balign 0x80
current_el_spx_fiq:
    b handle_fiq_wrapper
.balign 0x80
current_el_spx_serror:
    b handle_serror_wrapper

/*
 * Lower EL using AArch64
 */
.balign 0x80
lower_el_aarch64_sync:
    b handle_exception_wrapper
.balign 0x80
lower_el_aarch64_irq:
    b handle_irq_wrapper
.balign 0x80
lower_el_aarch64_fiq:
    b handle_fiq_wrapper
.balign 0x80
lower_el_aarch64_serror:
    b handle_serror_wrapper

/*
 * Lower EL using AArch32
 */
.balign 0x80
lower_el_aarch32_sync:
    b handle_exception_wrapper
.balign 0x80
lower_el_aarch32_irq:
    b handle_irq_wrapper
.balign 0x80
lower_el_aarch32_fiq:
    b handle_fiq_wrapper
.balign 0x80
lower_el_aarch32_serror:
    b handle_serror_wrapper

/*
 * Exception wrapper macros
 * Save context, call Rust handler, restore context
 */

.macro SAVE_CONTEXT
    /* Allocate space on stack for ExceptionContext (264 bytes) */
    sub sp, sp, #264

    /* Save general purpose registers x0-x30 */
    stp x0, x1, [sp, #0]
    stp x2, x3, [sp, #16]
    stp x4, x5, [sp, #32]
    stp x6, x7, [sp, #48]
    stp x8, x9, [sp, #64]
    stp x10, x11, [sp, #80]
    stp x12, x13, [sp, #96]
    stp x14, x15, [sp, #112]
    stp x16, x17, [sp, #128]
    stp x18, x19, [sp, #144]
    stp x20, x21, [sp, #160]
    stp x22, x23, [sp, #176]
    stp x24, x25, [sp, #192]
    stp x26, x27, [sp, #208]
    stp x28, x29, [sp, #224]
    str x30, [sp, #240]

    /* Save SP, ELR, SPSR */
    mrs x0, sp_el0
    str x0, [sp, #248]
    mrs x0, elr_el1
    str x0, [sp, #256]
    mrs x0, spsr_el1
    str x0, [sp, #264]
.endm

.macro RESTORE_CONTEXT
    /* Restore SP, ELR, SPSR */
    ldr x0, [sp, #248]
    msr sp_el0, x0
    ldr x0, [sp, #256]
    msr elr_el1, x0
    ldr x0, [sp, #264]
    msr spsr_el1, x0

    /* Restore general purpose registers x0-x30 */
    ldp x0, x1, [sp, #0]
    ldp x2, x3, [sp, #16]
    ldp x4, x5, [sp, #32]
    ldp x6, x7, [sp, #48]
    ldp x8, x9, [sp, #64]
    ldp x10, x11, [sp, #80]
    ldp x12, x13, [sp, #96]
    ldp x14, x15, [sp, #112]
    ldp x16, x17, [sp, #128]
    ldp x18, x19, [sp, #144]
    ldp x20, x21, [sp, #160]
    ldp x22, x23, [sp, #176]
    ldp x24, x25, [sp, #192]
    ldp x26, x27, [sp, #208]
    ldp x28, x29, [sp, #224]
    ldr x30, [sp, #240]

    /* Deallocate stack space */
    add sp, sp, #264
.endm

/*
 * Exception handlers that save/restore context
 */

handle_exception_wrapper:
    SAVE_CONTEXT
    mov x0, sp  /* Pass context pointer to Rust */
    bl handle_sync_exception
    RESTORE_CONTEXT
    eret

handle_irq_wrapper:
    SAVE_CONTEXT
    mov x0, sp
    bl handle_irq
    RESTORE_CONTEXT
    eret

handle_fiq_wrapper:
    SAVE_CONTEXT
    mov x0, sp
    bl handle_fiq
    RESTORE_CONTEXT
    eret

handle_serror_wrapper:
    SAVE_CONTEXT
    mov x0, sp
    bl handle_serror
    RESTORE_CONTEXT
    eret
