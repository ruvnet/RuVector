TARGET = aarch64-ruvix
KERNEL = target/$(TARGET)/release/ruvix-kernel

.PHONY: all build run clean

all: build

build:
	cargo build --release

run: build
	qemu-system-aarch64 \
		-machine virt \
		-cpu cortex-a72 \
		-m 128M \
		-nographic \
		-kernel $(KERNEL)

debug: build
	qemu-system-aarch64 \
		-machine virt \
		-cpu cortex-a72 \
		-m 128M \
		-nographic \
		-s -S \
		-kernel $(KERNEL)

clean:
	cargo clean
