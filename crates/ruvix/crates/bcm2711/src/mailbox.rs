//! # VideoCore Mailbox Driver for BCM2711/BCM2712
//!
//! This module implements the VideoCore mailbox interface for communicating
//! with the GPU firmware on Raspberry Pi 4/5.
//!
//! ## Overview
//!
//! The mailbox is a bi-directional communication channel between the ARM CPU
//! and the VideoCore GPU. It is used for:
//!
//! - Querying hardware information (board revision, MAC address, etc.)
//! - Managing memory (GPU memory allocation, framebuffer)
//! - Controlling clocks and power domains
//! - Setting voltage and temperature limits
//!
//! ## Mailbox Channels
//!
//! | Channel | Description |
//! |---------|-------------|
//! | 0 | Power management |
//! | 1 | Framebuffer |
//! | 2 | Virtual UART |
//! | 3 | VCHIQ |
//! | 4 | LEDs |
//! | 5 | Buttons |
//! | 6 | Touch screen |
//! | 7 | (unused) |
//! | 8 | Property tags (ARM -> VC) |
//! | 9 | Property tags (VC -> ARM) |
//!
//! ## Property Tags
//!
//! The property tag interface (channel 8) allows querying and setting various
//! hardware properties. Each request/response uses a buffer with the format:
//!
//! ```text
//! +----------------+----------------+
//! | Buffer Size    | 4 bytes        |
//! +----------------+----------------+
//! | Request/Response Code | 4 bytes |
//! +----------------+----------------+
//! | Tag 0          | Variable       |
//! +----------------+----------------+
//! | ...            |                |
//! +----------------+----------------+
//! | End Tag (0)    | 4 bytes        |
//! +----------------+----------------+
//! | Padding        | To 16-byte align |
//! +----------------+----------------+
//! ```
//!
//! ## Example
//!
//! ```rust,no_run
//! use ruvix_bcm2711::mailbox::Mailbox;
//!
//! let mailbox = Mailbox::new();
//!
//! // Get board revision
//! let revision = mailbox.get_board_revision().unwrap();
//! println!("Board revision: 0x{:08x}", revision);
//!
//! // Get firmware version
//! let version = mailbox.get_firmware_revision().unwrap();
//! println!("Firmware: 0x{:08x}", version);
//!
//! // Get ARM memory range
//! let (base, size) = mailbox.get_arm_memory().unwrap();
//! println!("ARM memory: 0x{:08x} - 0x{:08x}", base, base + size);
//! ```

use crate::mmio::{dsb, MmioReg};
use crate::MAILBOX_BASE;

// =============================================================================
// Register Offsets
// =============================================================================

/// Mailbox 0 Read register (read from VC).
const MBOX_READ: usize = 0x00;

/// Mailbox 0 Poll register.
const MBOX_POLL: usize = 0x10;

/// Mailbox 0 Sender register.
const MBOX_SENDER: usize = 0x14;

/// Mailbox 0 Status register.
const MBOX_STATUS: usize = 0x18;

/// Mailbox 0 Configuration register.
const MBOX_CONFIG: usize = 0x1C;

/// Mailbox 1 Write register (write to VC).
const MBOX_WRITE: usize = 0x20;

// =============================================================================
// Status Register Bits
// =============================================================================

/// Mailbox full flag (cannot write).
const MBOX_FULL: u32 = 1 << 31;

/// Mailbox empty flag (cannot read).
const MBOX_EMPTY: u32 = 1 << 30;

// =============================================================================
// Mailbox Channels
// =============================================================================

/// Power management channel.
pub const CHANNEL_POWER: u8 = 0;

/// Framebuffer channel.
pub const CHANNEL_FRAMEBUFFER: u8 = 1;

/// Virtual UART channel.
pub const CHANNEL_VUART: u8 = 2;

/// VCHIQ channel.
pub const CHANNEL_VCHIQ: u8 = 3;

/// LEDs channel.
pub const CHANNEL_LEDS: u8 = 4;

/// Buttons channel.
pub const CHANNEL_BUTTONS: u8 = 5;

/// Touchscreen channel.
pub const CHANNEL_TOUCHSCREEN: u8 = 6;

/// Property tags (ARM to VC) channel.
pub const CHANNEL_PROPERTY: u8 = 8;

// =============================================================================
// Property Tags
// =============================================================================

/// Request code for property tag buffer.
const TAG_REQUEST: u32 = 0x0000_0000;

/// Response success code.
const TAG_RESPONSE_SUCCESS: u32 = 0x8000_0000;

/// Response error code.
const TAG_RESPONSE_ERROR: u32 = 0x8000_0001;

/// End tag (terminates tag list).
const TAG_END: u32 = 0x0000_0000;

// VideoCore property tags
/// Get firmware revision.
const TAG_GET_FIRMWARE_REV: u32 = 0x0000_0001;

/// Get board model.
const TAG_GET_BOARD_MODEL: u32 = 0x0001_0001;

/// Get board revision.
const TAG_GET_BOARD_REVISION: u32 = 0x0001_0002;

/// Get board MAC address.
const TAG_GET_BOARD_MAC: u32 = 0x0001_0003;

/// Get board serial.
const TAG_GET_BOARD_SERIAL: u32 = 0x0001_0004;

/// Get ARM memory.
const TAG_GET_ARM_MEMORY: u32 = 0x0001_0005;

/// Get VC memory.
const TAG_GET_VC_MEMORY: u32 = 0x0001_0006;

/// Get clocks.
const TAG_GET_CLOCKS: u32 = 0x0001_0007;

/// Get clock state.
const TAG_GET_CLOCK_STATE: u32 = 0x0003_0001;

/// Set clock state.
const TAG_SET_CLOCK_STATE: u32 = 0x0003_8001;

/// Get clock rate.
const TAG_GET_CLOCK_RATE: u32 = 0x0003_0002;

/// Set clock rate.
const TAG_SET_CLOCK_RATE: u32 = 0x0003_8002;

/// Get max clock rate.
const TAG_GET_MAX_CLOCK_RATE: u32 = 0x0003_0004;

/// Get min clock rate.
const TAG_GET_MIN_CLOCK_RATE: u32 = 0x0003_0007;

/// Get temperature.
const TAG_GET_TEMPERATURE: u32 = 0x0003_0006;

/// Get max temperature.
const TAG_GET_MAX_TEMPERATURE: u32 = 0x0003_000A;

/// Allocate framebuffer.
const TAG_ALLOCATE_BUFFER: u32 = 0x0004_0001;

/// Release framebuffer.
const TAG_RELEASE_BUFFER: u32 = 0x0004_8001;

/// Get physical display size.
const TAG_GET_PHYSICAL_SIZE: u32 = 0x0004_0003;

/// Set physical display size.
const TAG_SET_PHYSICAL_SIZE: u32 = 0x0004_8003;

/// Get virtual display size.
const TAG_GET_VIRTUAL_SIZE: u32 = 0x0004_0004;

/// Set virtual display size.
const TAG_SET_VIRTUAL_SIZE: u32 = 0x0004_8004;

/// Get depth (bits per pixel).
const TAG_GET_DEPTH: u32 = 0x0004_0005;

/// Set depth.
const TAG_SET_DEPTH: u32 = 0x0004_8005;

/// Get pixel order.
const TAG_GET_PIXEL_ORDER: u32 = 0x0004_0006;

/// Set pixel order.
const TAG_SET_PIXEL_ORDER: u32 = 0x0004_8006;

/// Get pitch (bytes per row).
const TAG_GET_PITCH: u32 = 0x0004_0008;

// =============================================================================
// Clock IDs
// =============================================================================

/// Clock ID for EMMC.
pub const CLOCK_EMMC: u32 = 1;

/// Clock ID for UART.
pub const CLOCK_UART: u32 = 2;

/// Clock ID for ARM.
pub const CLOCK_ARM: u32 = 3;

/// Clock ID for core.
pub const CLOCK_CORE: u32 = 4;

/// Clock ID for V3D.
pub const CLOCK_V3D: u32 = 5;

/// Clock ID for H264.
pub const CLOCK_H264: u32 = 6;

/// Clock ID for ISP.
pub const CLOCK_ISP: u32 = 7;

/// Clock ID for SDRAM.
pub const CLOCK_SDRAM: u32 = 8;

/// Clock ID for pixel.
pub const CLOCK_PIXEL: u32 = 9;

/// Clock ID for PWM.
pub const CLOCK_PWM: u32 = 10;

/// Clock ID for HEVC.
pub const CLOCK_HEVC: u32 = 11;

/// Clock ID for EMMC2.
pub const CLOCK_EMMC2: u32 = 12;

/// Clock ID for M2MC.
pub const CLOCK_M2MC: u32 = 13;

/// Clock ID for pixel BVB.
pub const CLOCK_PIXEL_BVB: u32 = 14;

// =============================================================================
// Types
// =============================================================================

/// Mailbox error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MailboxError {
    /// Mailbox is full, cannot write.
    Full,
    /// Mailbox is empty, cannot read.
    Empty,
    /// Response indicates an error.
    ResponseError,
    /// Timeout waiting for response.
    Timeout,
    /// Invalid buffer alignment.
    BufferAlignment,
    /// Tag response indicates failure.
    TagError,
    /// Unexpected response channel.
    ChannelMismatch,
}

/// Result type for mailbox operations.
pub type Result<T> = core::result::Result<T, MailboxError>;

// =============================================================================
// Mailbox Driver
// =============================================================================

/// VideoCore Mailbox driver.
///
/// Provides communication with the GPU firmware via the mailbox interface.
pub struct Mailbox {
    base: usize,
}

impl Mailbox {
    /// Create a new Mailbox driver instance.
    #[inline]
    pub const fn new() -> Self {
        Self { base: MAILBOX_BASE }
    }

    /// Create a new Mailbox driver with a custom base address.
    ///
    /// # Safety
    ///
    /// The provided base address must be a valid mailbox controller base.
    #[inline]
    pub const unsafe fn with_base(base: usize) -> Self {
        Self { base }
    }

    /// Send a message to the VideoCore on the specified channel.
    ///
    /// # Arguments
    ///
    /// * `channel` - Mailbox channel (0-15)
    /// * `data` - 28-bit data value (lower 4 bits must be 0, will be replaced by channel)
    ///
    /// # Errors
    ///
    /// Returns `Err(MailboxError::Timeout)` if the mailbox remains full.
    pub fn send(&self, channel: u8, data: u32) -> Result<()> {
        // Wait until mailbox is not full
        let mut timeout = 1_000_000_u32;
        unsafe {
            let status = MmioReg::<u32>::new(self.base + MBOX_STATUS);
            while (status.read() & MBOX_FULL) != 0 {
                timeout = timeout.saturating_sub(1);
                if timeout == 0 {
                    return Err(MailboxError::Timeout);
                }
                core::hint::spin_loop();
            }

            // Write data | channel to mailbox 1 (ARM -> VC)
            dsb();
            let mut write_reg = MmioReg::<u32>::new(self.base + MBOX_WRITE);
            write_reg.write((data & 0xFFFF_FFF0) | (channel as u32 & 0xF));
            dsb();
        }

        Ok(())
    }

    /// Receive a message from the VideoCore on the specified channel.
    ///
    /// # Arguments
    ///
    /// * `channel` - Mailbox channel (0-15)
    ///
    /// # Returns
    ///
    /// The 28-bit data value from the response.
    ///
    /// # Errors
    ///
    /// Returns `Err(MailboxError::Timeout)` if no response is received.
    /// Returns `Err(MailboxError::ChannelMismatch)` if response is on wrong channel.
    pub fn receive(&self, channel: u8) -> Result<u32> {
        loop {
            // Wait until mailbox is not empty
            let mut timeout = 1_000_000_u32;
            unsafe {
                let status = MmioReg::<u32>::new(self.base + MBOX_STATUS);
                while (status.read() & MBOX_EMPTY) != 0 {
                    timeout = timeout.saturating_sub(1);
                    if timeout == 0 {
                        return Err(MailboxError::Timeout);
                    }
                    core::hint::spin_loop();
                }

                dsb();
                let read_reg = MmioReg::<u32>::new(self.base + MBOX_READ);
                let response = read_reg.read();
                dsb();

                let resp_channel = (response & 0xF) as u8;
                let data = response & 0xFFFF_FFF0;

                if resp_channel == channel {
                    return Ok(data);
                }
                // Wrong channel, keep reading
            }
        }
    }

    /// Call the property interface with a buffer.
    ///
    /// The buffer must be 16-byte aligned and in the format expected by
    /// the VideoCore property interface.
    ///
    /// # Arguments
    ///
    /// * `buffer` - Pointer to the property buffer (must be 16-byte aligned)
    ///
    /// # Safety
    ///
    /// The buffer must be valid and properly formatted.
    ///
    /// # Errors
    ///
    /// Returns error if the call fails or response indicates error.
    pub unsafe fn property_call(&self, buffer: *mut u32) -> Result<()> {
        // Buffer address must be 16-byte aligned
        let addr = buffer as usize;
        if addr & 0xF != 0 {
            return Err(MailboxError::BufferAlignment);
        }

        // Convert ARM address to bus address (add 0xC000_0000 for L2 cached)
        // For simplicity, we use uncached addressing (0x4000_0000 prefix)
        let bus_addr = (addr | 0xC000_0000) as u32;

        // Send buffer address on property channel
        self.send(CHANNEL_PROPERTY, bus_addr)?;

        // Wait for response
        let response = self.receive(CHANNEL_PROPERTY)?;

        // Check response code in buffer
        dsb();
        let response_code = buffer.add(1).read_volatile();

        if response_code == TAG_RESPONSE_SUCCESS {
            Ok(())
        } else {
            Err(MailboxError::ResponseError)
        }
    }

    /// Get the firmware revision.
    ///
    /// # Returns
    ///
    /// The firmware revision as a 32-bit value.
    pub fn get_firmware_revision(&self) -> Result<u32> {
        #[repr(C, align(16))]
        struct Buffer {
            size: u32,
            code: u32,
            tag: u32,
            tag_size: u32,
            tag_code: u32,
            value: u32,
            end: u32,
        }

        let mut buffer = Buffer {
            size: core::mem::size_of::<Buffer>() as u32,
            code: TAG_REQUEST,
            tag: TAG_GET_FIRMWARE_REV,
            tag_size: 4,
            tag_code: 0,
            value: 0,
            end: TAG_END,
        };

        unsafe {
            self.property_call(&mut buffer as *mut Buffer as *mut u32)?;
            Ok(buffer.value)
        }
    }

    /// Get the board revision.
    ///
    /// # Returns
    ///
    /// The board revision code (see BoardRevision enum).
    pub fn get_board_revision(&self) -> Result<u32> {
        #[repr(C, align(16))]
        struct Buffer {
            size: u32,
            code: u32,
            tag: u32,
            tag_size: u32,
            tag_code: u32,
            value: u32,
            end: u32,
        }

        let mut buffer = Buffer {
            size: core::mem::size_of::<Buffer>() as u32,
            code: TAG_REQUEST,
            tag: TAG_GET_BOARD_REVISION,
            tag_size: 4,
            tag_code: 0,
            value: 0,
            end: TAG_END,
        };

        unsafe {
            self.property_call(&mut buffer as *mut Buffer as *mut u32)?;
            Ok(buffer.value)
        }
    }

    /// Get the board serial number.
    ///
    /// # Returns
    ///
    /// The 64-bit board serial number.
    pub fn get_board_serial(&self) -> Result<u64> {
        #[repr(C, align(16))]
        struct Buffer {
            size: u32,
            code: u32,
            tag: u32,
            tag_size: u32,
            tag_code: u32,
            serial_low: u32,
            serial_high: u32,
            end: u32,
        }

        let mut buffer = Buffer {
            size: core::mem::size_of::<Buffer>() as u32,
            code: TAG_REQUEST,
            tag: TAG_GET_BOARD_SERIAL,
            tag_size: 8,
            tag_code: 0,
            serial_low: 0,
            serial_high: 0,
            end: TAG_END,
        };

        unsafe {
            self.property_call(&mut buffer as *mut Buffer as *mut u32)?;
            let serial = ((buffer.serial_high as u64) << 32) | (buffer.serial_low as u64);
            Ok(serial)
        }
    }

    /// Get the ARM memory range.
    ///
    /// # Returns
    ///
    /// A tuple of (base_address, size) for the ARM memory.
    pub fn get_arm_memory(&self) -> Result<(u32, u32)> {
        #[repr(C, align(16))]
        struct Buffer {
            size: u32,
            code: u32,
            tag: u32,
            tag_size: u32,
            tag_code: u32,
            base: u32,
            mem_size: u32,
            end: u32,
        }

        let mut buffer = Buffer {
            size: core::mem::size_of::<Buffer>() as u32,
            code: TAG_REQUEST,
            tag: TAG_GET_ARM_MEMORY,
            tag_size: 8,
            tag_code: 0,
            base: 0,
            mem_size: 0,
            end: TAG_END,
        };

        unsafe {
            self.property_call(&mut buffer as *mut Buffer as *mut u32)?;
            Ok((buffer.base, buffer.mem_size))
        }
    }

    /// Get the VideoCore memory range.
    ///
    /// # Returns
    ///
    /// A tuple of (base_address, size) for the VC memory.
    pub fn get_vc_memory(&self) -> Result<(u32, u32)> {
        #[repr(C, align(16))]
        struct Buffer {
            size: u32,
            code: u32,
            tag: u32,
            tag_size: u32,
            tag_code: u32,
            base: u32,
            mem_size: u32,
            end: u32,
        }

        let mut buffer = Buffer {
            size: core::mem::size_of::<Buffer>() as u32,
            code: TAG_REQUEST,
            tag: TAG_GET_VC_MEMORY,
            tag_size: 8,
            tag_code: 0,
            base: 0,
            mem_size: 0,
            end: TAG_END,
        };

        unsafe {
            self.property_call(&mut buffer as *mut Buffer as *mut u32)?;
            Ok((buffer.base, buffer.mem_size))
        }
    }

    /// Get the current clock rate for a clock.
    ///
    /// # Arguments
    ///
    /// * `clock_id` - The clock identifier (see CLOCK_* constants)
    ///
    /// # Returns
    ///
    /// The clock rate in Hz.
    pub fn get_clock_rate(&self, clock_id: u32) -> Result<u32> {
        #[repr(C, align(16))]
        struct Buffer {
            size: u32,
            code: u32,
            tag: u32,
            tag_size: u32,
            tag_code: u32,
            clock_id: u32,
            rate: u32,
            end: u32,
        }

        let mut buffer = Buffer {
            size: core::mem::size_of::<Buffer>() as u32,
            code: TAG_REQUEST,
            tag: TAG_GET_CLOCK_RATE,
            tag_size: 8,
            tag_code: 0,
            clock_id,
            rate: 0,
            end: TAG_END,
        };

        unsafe {
            self.property_call(&mut buffer as *mut Buffer as *mut u32)?;
            Ok(buffer.rate)
        }
    }

    /// Set the clock rate for a clock.
    ///
    /// # Arguments
    ///
    /// * `clock_id` - The clock identifier (see CLOCK_* constants)
    /// * `rate` - The desired clock rate in Hz
    /// * `skip_turbo` - If true, don't apply turbo setting
    ///
    /// # Returns
    ///
    /// The actual clock rate set.
    pub fn set_clock_rate(&self, clock_id: u32, rate: u32, skip_turbo: bool) -> Result<u32> {
        #[repr(C, align(16))]
        struct Buffer {
            size: u32,
            code: u32,
            tag: u32,
            tag_size: u32,
            tag_code: u32,
            clock_id: u32,
            rate: u32,
            skip_turbo: u32,
            end: u32,
        }

        let mut buffer = Buffer {
            size: core::mem::size_of::<Buffer>() as u32,
            code: TAG_REQUEST,
            tag: TAG_SET_CLOCK_RATE,
            tag_size: 12,
            tag_code: 0,
            clock_id,
            rate,
            skip_turbo: if skip_turbo { 1 } else { 0 },
            end: TAG_END,
        };

        unsafe {
            self.property_call(&mut buffer as *mut Buffer as *mut u32)?;
            Ok(buffer.rate)
        }
    }

    /// Get the SoC temperature.
    ///
    /// # Returns
    ///
    /// The temperature in millidegrees Celsius.
    pub fn get_temperature(&self) -> Result<u32> {
        #[repr(C, align(16))]
        struct Buffer {
            size: u32,
            code: u32,
            tag: u32,
            tag_size: u32,
            tag_code: u32,
            sensor_id: u32,
            temperature: u32,
            end: u32,
        }

        let mut buffer = Buffer {
            size: core::mem::size_of::<Buffer>() as u32,
            code: TAG_REQUEST,
            tag: TAG_GET_TEMPERATURE,
            tag_size: 8,
            tag_code: 0,
            sensor_id: 0, // 0 = SoC temperature
            temperature: 0,
            end: TAG_END,
        };

        unsafe {
            self.property_call(&mut buffer as *mut Buffer as *mut u32)?;
            Ok(buffer.temperature)
        }
    }

    /// Get the maximum SoC temperature before throttling.
    ///
    /// # Returns
    ///
    /// The maximum temperature in millidegrees Celsius.
    pub fn get_max_temperature(&self) -> Result<u32> {
        #[repr(C, align(16))]
        struct Buffer {
            size: u32,
            code: u32,
            tag: u32,
            tag_size: u32,
            tag_code: u32,
            sensor_id: u32,
            temperature: u32,
            end: u32,
        }

        let mut buffer = Buffer {
            size: core::mem::size_of::<Buffer>() as u32,
            code: TAG_REQUEST,
            tag: TAG_GET_MAX_TEMPERATURE,
            tag_size: 8,
            tag_code: 0,
            sensor_id: 0,
            temperature: 0,
            end: TAG_END,
        };

        unsafe {
            self.property_call(&mut buffer as *mut Buffer as *mut u32)?;
            Ok(buffer.temperature)
        }
    }
}

impl Default for Mailbox {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_constants() {
        assert_eq!(CHANNEL_POWER, 0);
        assert_eq!(CHANNEL_PROPERTY, 8);
    }

    #[test]
    fn test_clock_constants() {
        assert_eq!(CLOCK_ARM, 3);
        assert_eq!(CLOCK_UART, 2);
    }
}
