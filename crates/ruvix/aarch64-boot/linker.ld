ENTRY(_start)

MEMORY {
    RAM (rwx) : ORIGIN = 0x40000000, LENGTH = 128M
}

SECTIONS {
    . = 0x40000000;

    .text.boot : {
        KEEP(*(.text.boot))
    } > RAM

    .text : {
        *(.text .text.*)
    } > RAM

    .rodata : ALIGN(4K) {
        *(.rodata .rodata.*)
    } > RAM

    .data : ALIGN(4K) {
        *(.data .data.*)
    } > RAM

    .bss : ALIGN(4K) {
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        __bss_end = .;
    } > RAM

    . = ALIGN(4K);
    __stack_bottom = .;
    . = . + 64K;
    __stack_top = .;

    . = ALIGN(4K);
    __heap_start = .;
    __heap_end = ORIGIN(RAM) + LENGTH(RAM);
}
