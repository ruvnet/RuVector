//! # BCM2711 GPIO Controller Driver
//!
//! This module implements a driver for the BCM2711 GPIO controller found in
//! Raspberry Pi 4/5. The controller provides 58 GPIO pins with configurable
//! functions, pull-up/pull-down resistors, and interrupt capabilities.
//!
//! ## GPIO Pin Functions
//!
//! Each GPIO pin can be configured for one of 8 alternate functions:
//!
//! | Function | Description |
//! |----------|-------------|
//! | Input | Digital input |
//! | Output | Digital output |
//! | Alt0 | Alternate function 0 (device-specific) |
//! | Alt1 | Alternate function 1 |
//! | Alt2 | Alternate function 2 |
//! | Alt3 | Alternate function 3 |
//! | Alt4 | Alternate function 4 |
//! | Alt5 | Alternate function 5 |
//!
//! ## Common Pin Assignments (RPi 4)
//!
//! | Pin | Alt0 | Alt3 | Alt4 | Alt5 |
//! |-----|------|------|------|------|
//! | 14 | UART0 TXD | - | - | UART1 TXD |
//! | 15 | UART0 RXD | - | - | UART1 RXD |
//! | 2 | SDA1 | - | - | - |
//! | 3 | SCL1 | - | - | - |
//! | 7 | SPI0 CE1 | - | - | - |
//! | 8 | SPI0 CE0 | - | - | - |
//! | 9 | SPI0 MISO | - | - | - |
//! | 10 | SPI0 MOSI | - | - | - |
//! | 11 | SPI0 SCLK | - | - | - |
//!
//! ## Register Map
//!
//! | Offset | Register | Description |
//! |--------|----------|-------------|
//! | 0x00 | GPFSEL0 | GPIO Function Select 0 (pins 0-9) |
//! | 0x04 | GPFSEL1 | GPIO Function Select 1 (pins 10-19) |
//! | ... | ... | ... |
//! | 0x1C | GPSET0 | GPIO Pin Output Set 0 (pins 0-31) |
//! | 0x20 | GPSET1 | GPIO Pin Output Set 1 (pins 32-57) |
//! | 0x28 | GPCLR0 | GPIO Pin Output Clear 0 |
//! | 0x2C | GPCLR1 | GPIO Pin Output Clear 1 |
//! | 0x34 | GPLEV0 | GPIO Pin Level 0 |
//! | 0x38 | GPLEV1 | GPIO Pin Level 1 |
//! | 0xE4 | GPIO_PUP_PDN_CNTRL_REG0 | Pull-up/down control (BCM2711) |
//!
//! ## Example
//!
//! ```rust,no_run
//! use ruvix_bcm2711::gpio::{Gpio, Function, Pull};
//!
//! let mut gpio = Gpio::new();
//!
//! // Configure GPIO 14 for mini UART TX
//! gpio.set_function(14, Function::Alt5).unwrap();
//! gpio.set_pull(14, Pull::None).unwrap();
//!
//! // Configure GPIO 21 as output LED
//! gpio.set_function(21, Function::Output).unwrap();
//! gpio.write(21, true);  // Turn on
//! gpio.write(21, false); // Turn off
//!
//! // Read GPIO 17 input
//! gpio.set_function(17, Function::Input).unwrap();
//! gpio.set_pull(17, Pull::Up).unwrap();
//! let value = gpio.read(17);
//! ```

use crate::mmio::{delay_cycles, dsb, MmioReg};
use crate::GPIO_BASE;

// =============================================================================
// Register Offsets
// =============================================================================

/// GPIO Function Select registers (GPFSEL0-5), 3 bits per pin.
const GPFSEL0: usize = 0x00;
const GPFSEL1: usize = 0x04;
const GPFSEL2: usize = 0x08;
const GPFSEL3: usize = 0x0C;
const GPFSEL4: usize = 0x10;
const GPFSEL5: usize = 0x14;

/// GPIO Pin Output Set registers.
const GPSET0: usize = 0x1C;
const GPSET1: usize = 0x20;

/// GPIO Pin Output Clear registers.
const GPCLR0: usize = 0x28;
const GPCLR1: usize = 0x2C;

/// GPIO Pin Level registers (read-only).
const GPLEV0: usize = 0x34;
const GPLEV1: usize = 0x38;

/// GPIO Event Detect Status registers.
const GPEDS0: usize = 0x40;
const GPEDS1: usize = 0x44;

/// GPIO Rising Edge Detect Enable registers.
const GPREN0: usize = 0x4C;
const GPREN1: usize = 0x50;

/// GPIO Falling Edge Detect Enable registers.
const GPFEN0: usize = 0x58;
const GPFEN1: usize = 0x5C;

/// GPIO High Detect Enable registers.
const GPHEN0: usize = 0x64;
const GPHEN1: usize = 0x68;

/// GPIO Low Detect Enable registers.
const GPLEN0: usize = 0x70;
const GPLEN1: usize = 0x74;

/// GPIO Async Rising Edge Detect registers.
const GPAREN0: usize = 0x7C;
const GPAREN1: usize = 0x80;

/// GPIO Async Falling Edge Detect registers.
const GPAFEN0: usize = 0x88;
const GPAFEN1: usize = 0x8C;

/// BCM2711 Pull-up/Pull-down registers (replaces GPPUD/GPPUDCLK).
const GPIO_PUP_PDN_CNTRL_REG0: usize = 0xE4;
const GPIO_PUP_PDN_CNTRL_REG1: usize = 0xE8;
const GPIO_PUP_PDN_CNTRL_REG2: usize = 0xEC;
const GPIO_PUP_PDN_CNTRL_REG3: usize = 0xF0;

// =============================================================================
// Constants
// =============================================================================

/// Maximum GPIO pin number (0-57 on BCM2711).
const MAX_PIN: u8 = 57;

/// Number of pins per GPFSEL register.
const PINS_PER_FSEL: u8 = 10;

/// Bits per pin in GPFSEL registers.
const BITS_PER_FSEL: u8 = 3;

/// Number of pins per pull control register.
const PINS_PER_PULL: u8 = 16;

/// Bits per pin in pull control registers.
const BITS_PER_PULL: u8 = 2;

// =============================================================================
// Types
// =============================================================================

/// GPIO pin function selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Function {
    /// Digital input.
    Input = 0b000,
    /// Digital output.
    Output = 0b001,
    /// Alternate function 0 (e.g., UART0, I2C0, SPI0).
    Alt0 = 0b100,
    /// Alternate function 1.
    Alt1 = 0b101,
    /// Alternate function 2.
    Alt2 = 0b110,
    /// Alternate function 3.
    Alt3 = 0b111,
    /// Alternate function 4.
    Alt4 = 0b011,
    /// Alternate function 5 (e.g., UART1/Mini UART).
    Alt5 = 0b010,
}

/// GPIO pull-up/pull-down resistor configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Pull {
    /// No pull-up or pull-down.
    None = 0b00,
    /// Pull-up resistor enabled.
    Up = 0b01,
    /// Pull-down resistor enabled.
    Down = 0b10,
    /// Reserved (do not use).
    Reserved = 0b11,
}

/// GPIO edge detection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDetect {
    /// No edge detection.
    None,
    /// Rising edge detection.
    Rising,
    /// Falling edge detection.
    Falling,
    /// Both edges detection.
    Both,
}

/// GPIO driver error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpioError {
    /// Pin number out of range (0-57).
    InvalidPin,
}

// =============================================================================
// GPIO Controller
// =============================================================================

/// BCM2711 GPIO Controller driver.
///
/// Provides low-level access to GPIO pins on Raspberry Pi 4/5.
pub struct Gpio {
    base: usize,
}

impl Gpio {
    /// Create a new GPIO controller instance.
    ///
    /// Uses the default GPIO base address for the platform.
    #[inline]
    pub const fn new() -> Self {
        Self { base: GPIO_BASE }
    }

    /// Create a new GPIO controller with a custom base address.
    ///
    /// # Safety
    ///
    /// The provided base address must be a valid GPIO controller base.
    #[inline]
    pub const unsafe fn with_base(base: usize) -> Self {
        Self { base }
    }

    /// Set the function of a GPIO pin.
    ///
    /// # Arguments
    ///
    /// * `pin` - GPIO pin number (0-57)
    /// * `function` - Desired pin function
    ///
    /// # Errors
    ///
    /// Returns `Err(GpioError::InvalidPin)` if pin number is out of range.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use ruvix_bcm2711::gpio::{Gpio, Function};
    ///
    /// let mut gpio = Gpio::new();
    /// gpio.set_function(14, Function::Alt5).unwrap(); // Mini UART TX
    /// ```
    pub fn set_function(&mut self, pin: u8, function: Function) -> Result<(), GpioError> {
        if pin > MAX_PIN {
            return Err(GpioError::InvalidPin);
        }

        let reg_offset = (pin / PINS_PER_FSEL) as usize * 4;
        let bit_offset = ((pin % PINS_PER_FSEL) * BITS_PER_FSEL) as usize;

        // SAFETY: Base is guaranteed to be valid GPIO controller address
        unsafe {
            let mut reg = MmioReg::<u32>::new(self.base + GPFSEL0 + reg_offset);
            reg.modify(|val| {
                // Clear the 3-bit field for this pin
                let mask = !(0b111_u32 << bit_offset);
                let new_val = (val & mask) | ((function as u32) << bit_offset);
                new_val
            });
        }

        Ok(())
    }

    /// Get the current function of a GPIO pin.
    ///
    /// # Arguments
    ///
    /// * `pin` - GPIO pin number (0-57)
    ///
    /// # Errors
    ///
    /// Returns `Err(GpioError::InvalidPin)` if pin number is out of range.
    pub fn get_function(&self, pin: u8) -> Result<Function, GpioError> {
        if pin > MAX_PIN {
            return Err(GpioError::InvalidPin);
        }

        let reg_offset = (pin / PINS_PER_FSEL) as usize * 4;
        let bit_offset = ((pin % PINS_PER_FSEL) * BITS_PER_FSEL) as usize;

        // SAFETY: Base is guaranteed to be valid GPIO controller address
        let val = unsafe {
            let reg = MmioReg::<u32>::new(self.base + GPFSEL0 + reg_offset);
            (reg.read() >> bit_offset) & 0b111
        };

        Ok(match val {
            0b000 => Function::Input,
            0b001 => Function::Output,
            0b100 => Function::Alt0,
            0b101 => Function::Alt1,
            0b110 => Function::Alt2,
            0b111 => Function::Alt3,
            0b011 => Function::Alt4,
            0b010 => Function::Alt5,
            _ => Function::Input, // Should never happen
        })
    }

    /// Set the pull-up/pull-down resistor for a GPIO pin.
    ///
    /// BCM2711 uses a new register layout for pull control (different from BCM2835).
    ///
    /// # Arguments
    ///
    /// * `pin` - GPIO pin number (0-57)
    /// * `pull` - Pull-up/pull-down configuration
    ///
    /// # Errors
    ///
    /// Returns `Err(GpioError::InvalidPin)` if pin number is out of range.
    pub fn set_pull(&mut self, pin: u8, pull: Pull) -> Result<(), GpioError> {
        if pin > MAX_PIN {
            return Err(GpioError::InvalidPin);
        }

        // BCM2711 has 4 pull control registers, 16 pins per register, 2 bits per pin
        let reg_offset = (pin / PINS_PER_PULL) as usize * 4;
        let bit_offset = ((pin % PINS_PER_PULL) * BITS_PER_PULL) as usize;

        // SAFETY: Base is guaranteed to be valid GPIO controller address
        unsafe {
            let mut reg = MmioReg::<u32>::new(self.base + GPIO_PUP_PDN_CNTRL_REG0 + reg_offset);
            reg.modify(|val| {
                // Clear the 2-bit field for this pin
                let mask = !(0b11_u32 << bit_offset);
                let new_val = (val & mask) | ((pull as u32) << bit_offset);
                new_val
            });
        }

        Ok(())
    }

    /// Get the current pull-up/pull-down configuration for a GPIO pin.
    ///
    /// # Arguments
    ///
    /// * `pin` - GPIO pin number (0-57)
    ///
    /// # Errors
    ///
    /// Returns `Err(GpioError::InvalidPin)` if pin number is out of range.
    pub fn get_pull(&self, pin: u8) -> Result<Pull, GpioError> {
        if pin > MAX_PIN {
            return Err(GpioError::InvalidPin);
        }

        let reg_offset = (pin / PINS_PER_PULL) as usize * 4;
        let bit_offset = ((pin % PINS_PER_PULL) * BITS_PER_PULL) as usize;

        // SAFETY: Base is guaranteed to be valid GPIO controller address
        let val = unsafe {
            let reg = MmioReg::<u32>::new(self.base + GPIO_PUP_PDN_CNTRL_REG0 + reg_offset);
            (reg.read() >> bit_offset) & 0b11
        };

        Ok(match val {
            0b00 => Pull::None,
            0b01 => Pull::Up,
            0b10 => Pull::Down,
            _ => Pull::Reserved,
        })
    }

    /// Set the output level of a GPIO pin (high/low).
    ///
    /// The pin must be configured as output first.
    ///
    /// # Arguments
    ///
    /// * `pin` - GPIO pin number (0-57)
    /// * `high` - `true` for high (3.3V), `false` for low (0V)
    #[inline]
    pub fn write(&mut self, pin: u8, high: bool) {
        if pin > MAX_PIN {
            return;
        }

        let (set_offset, clr_offset) = if pin < 32 {
            (GPSET0, GPCLR0)
        } else {
            (GPSET1, GPCLR1)
        };

        let bit = 1_u32 << (pin % 32);

        // SAFETY: Base is guaranteed to be valid GPIO controller address
        unsafe {
            if high {
                let mut reg = MmioReg::<u32>::new(self.base + set_offset);
                reg.write(bit);
            } else {
                let mut reg = MmioReg::<u32>::new(self.base + clr_offset);
                reg.write(bit);
            }
        }
    }

    /// Read the current level of a GPIO pin.
    ///
    /// # Arguments
    ///
    /// * `pin` - GPIO pin number (0-57)
    ///
    /// # Returns
    ///
    /// `true` if pin is high, `false` if low or invalid pin.
    #[inline]
    pub fn read(&self, pin: u8) -> bool {
        if pin > MAX_PIN {
            return false;
        }

        let offset = if pin < 32 { GPLEV0 } else { GPLEV1 };
        let bit = 1_u32 << (pin % 32);

        // SAFETY: Base is guaranteed to be valid GPIO controller address
        unsafe {
            let reg = MmioReg::<u32>::new(self.base + offset);
            (reg.read() & bit) != 0
        }
    }

    /// Configure edge detection for a GPIO pin.
    ///
    /// # Arguments
    ///
    /// * `pin` - GPIO pin number (0-57)
    /// * `edge` - Edge detection mode
    ///
    /// # Errors
    ///
    /// Returns `Err(GpioError::InvalidPin)` if pin number is out of range.
    pub fn set_edge_detect(&mut self, pin: u8, edge: EdgeDetect) -> Result<(), GpioError> {
        if pin > MAX_PIN {
            return Err(GpioError::InvalidPin);
        }

        let (ren_offset, fen_offset) = if pin < 32 {
            (GPREN0, GPFEN0)
        } else {
            (GPREN1, GPFEN1)
        };

        let bit = 1_u32 << (pin % 32);

        // SAFETY: Base is guaranteed to be valid GPIO controller address
        unsafe {
            let mut ren = MmioReg::<u32>::new(self.base + ren_offset);
            let mut fen = MmioReg::<u32>::new(self.base + fen_offset);

            match edge {
                EdgeDetect::None => {
                    ren.modify(|v| v & !bit);
                    fen.modify(|v| v & !bit);
                }
                EdgeDetect::Rising => {
                    ren.modify(|v| v | bit);
                    fen.modify(|v| v & !bit);
                }
                EdgeDetect::Falling => {
                    ren.modify(|v| v & !bit);
                    fen.modify(|v| v | bit);
                }
                EdgeDetect::Both => {
                    ren.modify(|v| v | bit);
                    fen.modify(|v| v | bit);
                }
            }
        }

        Ok(())
    }

    /// Check if an event was detected on a GPIO pin.
    ///
    /// # Arguments
    ///
    /// * `pin` - GPIO pin number (0-57)
    ///
    /// # Returns
    ///
    /// `true` if an event was detected (edge or level as configured).
    #[inline]
    pub fn event_detected(&self, pin: u8) -> bool {
        if pin > MAX_PIN {
            return false;
        }

        let offset = if pin < 32 { GPEDS0 } else { GPEDS1 };
        let bit = 1_u32 << (pin % 32);

        // SAFETY: Base is guaranteed to be valid GPIO controller address
        unsafe {
            let reg = MmioReg::<u32>::new(self.base + offset);
            (reg.read() & bit) != 0
        }
    }

    /// Clear an event detection status for a GPIO pin.
    ///
    /// # Arguments
    ///
    /// * `pin` - GPIO pin number (0-57)
    #[inline]
    pub fn clear_event(&mut self, pin: u8) {
        if pin > MAX_PIN {
            return;
        }

        let offset = if pin < 32 { GPEDS0 } else { GPEDS1 };
        let bit = 1_u32 << (pin % 32);

        // SAFETY: Base is guaranteed to be valid GPIO controller address
        unsafe {
            let mut reg = MmioReg::<u32>::new(self.base + offset);
            reg.write(bit); // Write 1 to clear
        }
    }

    /// Toggle a GPIO output pin.
    ///
    /// # Arguments
    ///
    /// * `pin` - GPIO pin number (0-57)
    #[inline]
    pub fn toggle(&mut self, pin: u8) {
        let current = self.read(pin);
        self.write(pin, !current);
    }
}

impl Default for Gpio {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_values() {
        assert_eq!(Function::Input as u32, 0b000);
        assert_eq!(Function::Output as u32, 0b001);
        assert_eq!(Function::Alt0 as u32, 0b100);
        assert_eq!(Function::Alt5 as u32, 0b010);
    }

    #[test]
    fn test_pull_values() {
        assert_eq!(Pull::None as u32, 0b00);
        assert_eq!(Pull::Up as u32, 0b01);
        assert_eq!(Pull::Down as u32, 0b10);
    }

    #[test]
    fn test_invalid_pin() {
        let mut gpio = Gpio::new();
        assert_eq!(gpio.set_function(58, Function::Output), Err(GpioError::InvalidPin));
        assert_eq!(gpio.set_function(255, Function::Output), Err(GpioError::InvalidPin));
    }

    #[test]
    fn test_pin_boundaries() {
        let gpio = Gpio::new();

        // Pin 0 should be valid
        assert!(gpio.get_function(0).is_ok());

        // Pin 57 should be valid
        assert!(gpio.get_function(57).is_ok());

        // Pin 58 should be invalid
        assert!(gpio.get_function(58).is_err());
    }
}
