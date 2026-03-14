//! Sensor types for RuView perception integration.
//!
//! RuView sits outside the kernel but provides the perception plane.
//! Sensors produce typed, coherence-scored events delivered via queues.

use crate::handle::Handle;

/// Handle to a sensor subscription.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct SubscriptionHandle(pub Handle);

impl SubscriptionHandle {
    /// Creates a new subscription handle.
    #[inline]
    #[must_use]
    pub const fn new(id: u32, generation: u32) -> Self {
        Self(Handle::new(id, generation))
    }

    /// Creates a null (invalid) subscription handle.
    #[inline]
    #[must_use]
    pub const fn null() -> Self {
        Self(Handle::null())
    }

    /// Checks if this handle is null.
    #[inline]
    #[must_use]
    pub const fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Returns the raw handle.
    #[inline]
    #[must_use]
    pub const fn raw(&self) -> Handle {
        self.0
    }
}

impl Default for SubscriptionHandle {
    fn default() -> Self {
        Self::null()
    }
}

/// Type of sensor data source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SensorType {
    /// Camera/video sensor.
    Camera = 0,

    /// Microphone/audio sensor.
    Microphone = 1,

    /// Network traffic tap.
    NetworkTap = 2,

    /// Financial market data feed.
    MarketFeed = 3,

    /// Git repository event stream.
    GitStream = 4,

    /// File system change events.
    FileSystem = 5,

    /// System metrics (CPU, memory, etc.).
    SystemMetrics = 6,

    /// Custom/user-defined sensor type.
    Custom = 255,
}

impl SensorType {
    /// Returns the sensor type as a string.
    #[inline]
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Camera => "Camera",
            Self::Microphone => "Microphone",
            Self::NetworkTap => "NetworkTap",
            Self::MarketFeed => "MarketFeed",
            Self::GitStream => "GitStream",
            Self::FileSystem => "FileSystem",
            Self::SystemMetrics => "SystemMetrics",
            Self::Custom => "Custom",
        }
    }

    /// Converts from a raw u8 value.
    #[inline]
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Camera),
            1 => Some(Self::Microphone),
            2 => Some(Self::NetworkTap),
            3 => Some(Self::MarketFeed),
            4 => Some(Self::GitStream),
            5 => Some(Self::FileSystem),
            6 => Some(Self::SystemMetrics),
            255 => Some(Self::Custom),
            _ => None,
        }
    }
}

impl Default for SensorType {
    fn default() -> Self {
        Self::Custom
    }
}

/// Sensor descriptor identifying a data source.
///
/// Used in `sensor_subscribe` to specify which sensor to monitor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct SensorDescriptor {
    /// Type of sensor.
    pub sensor_type: SensorType,

    /// Device identifier (hardware address, URL hash, stream ID, etc.).
    /// Interpretation depends on sensor_type.
    pub device_id: u64,

    /// Filter expression hash (0 = no filter).
    /// The actual filter is stored in a region and referenced by hash.
    pub filter_hash: u64,

    /// Requested sampling rate (events per second).
    /// 0 = all events (no downsampling).
    pub sample_rate: u32,
}

impl SensorDescriptor {
    /// Creates a new sensor descriptor.
    #[inline]
    #[must_use]
    pub const fn new(sensor_type: SensorType, device_id: u64) -> Self {
        Self {
            sensor_type,
            device_id,
            filter_hash: 0,
            sample_rate: 0,
        }
    }

    /// Creates a descriptor with sampling rate.
    #[inline]
    #[must_use]
    pub const fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Creates a descriptor with a filter.
    #[inline]
    #[must_use]
    pub const fn with_filter(mut self, filter_hash: u64) -> Self {
        self.filter_hash = filter_hash;
        self
    }

    /// Returns true if downsampling is enabled.
    #[inline]
    #[must_use]
    pub const fn is_downsampled(&self) -> bool {
        self.sample_rate > 0
    }

    /// Returns true if a filter is applied.
    #[inline]
    #[must_use]
    pub const fn has_filter(&self) -> bool {
        self.filter_hash != 0
    }
}

impl Default for SensorDescriptor {
    fn default() -> Self {
        Self {
            sensor_type: SensorType::Custom,
            device_id: 0,
            filter_hash: 0,
            sample_rate: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subscription_handle() {
        let h = SubscriptionHandle::new(3, 5);
        assert!(!h.is_null());
        assert_eq!(h.raw().id, 3);
    }

    #[test]
    fn test_sensor_type_roundtrip() {
        for i in 0..=6 {
            let st = SensorType::from_u8(i).unwrap();
            assert_eq!(st as u8, i);
        }
        assert!(SensorType::from_u8(100).is_none());
        assert_eq!(SensorType::from_u8(255), Some(SensorType::Custom));
    }

    #[test]
    fn test_sensor_descriptor() {
        let desc = SensorDescriptor::new(SensorType::Camera, 12345)
            .with_sample_rate(30)
            .with_filter(0xABCD);

        assert!(desc.is_downsampled());
        assert!(desc.has_filter());
        assert_eq!(desc.sample_rate, 30);
    }
}
