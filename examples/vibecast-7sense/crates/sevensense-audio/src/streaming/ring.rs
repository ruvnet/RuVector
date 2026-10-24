//! Lock-free single-producer single-consumer ring buffer for live audio.
//!
//! The producer is a real-time audio callback: it must never allocate, lock, or
//! block. When the consumer falls behind, the producer overwrites the oldest
//! samples and the consumer detects the lapse from the position counters. ADR-010
//! records why lossy overwrite is preferred to back-pressure here -- stalling an
//! audio callback causes device-level glitches, and in a monitoring deployment
//! stale audio is worth less than current audio.
//!
//! Each slot carries a monotonically increasing publication stamp around its
//! [`AtomicU32`] sample. The stamp acts as a sequence lock: a consumer accepts a
//! sample only when the slot still contains the exact logical position it asked
//! for before and after the value load. That prevents a concurrent overwrite
//! from being mistaken for an older sample.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

#[derive(Debug)]
struct Slot {
    sample: AtomicU32,
    /// Even values identify a stable logical position; odd values mean that
    /// the producer is replacing this slot.
    stamp: AtomicU64,
}

impl Slot {
    const fn empty() -> Self {
        Self {
            sample: AtomicU32::new(0),
            stamp: AtomicU64::new(0),
        }
    }
}

/// A bounded SPSC ring buffer that overwrites its oldest samples when full.
///
/// Capacity is rounded up to a power of two so index wrapping is a mask.
///
/// # Example
/// ```
/// use sevensense_audio::streaming::RingBuffer;
///
/// let ring = RingBuffer::new(8);
/// ring.write(&[1.0, 2.0, 3.0]);
///
/// let mut out = [0.0f32; 3];
/// let (read, dropped) = ring.read(&mut out);
/// assert_eq!(read, 3);
/// assert_eq!(dropped, 0);
/// assert_eq!(out, [1.0, 2.0, 3.0]);
/// ```
#[derive(Debug)]
pub struct RingBuffer {
    slots: Box<[Slot]>,
    mask: usize,
    /// Total samples ever written. Only the producer stores to this.
    write_pos: AtomicU64,
    /// Total samples ever consumed. Only the consumer stores to this.
    read_pos: AtomicU64,
    /// Total samples lost to overwrite, accumulated by the consumer.
    dropped: AtomicU64,
}

impl RingBuffer {
    /// Maximum supported value of `write_pos`.
    ///
    /// Publication stamps reserve zero for empty slots and encode a logical
    /// position as `(position + 1) * 2`, so positions must remain below this
    /// bound to keep every stable stamp non-zero and unique.
    const MAX_WRITE_POS: u64 = (1u64 << 63) - 1;

    /// Creates a ring buffer holding at least `capacity` samples.
    ///
    /// # Panics
    /// Panics if `capacity` is zero.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "ring buffer capacity must be non-zero");
        let capacity = capacity.next_power_of_two();
        let slots = (0..capacity)
            .map(|_| Slot::empty())
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            slots,
            mask: capacity - 1,
            write_pos: AtomicU64::new(0),
            read_pos: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
        }
    }

    /// Creates a ring buffer sized to hold `seconds` of audio at `sample_rate`.
    #[must_use]
    pub fn with_duration(seconds: f32, sample_rate: u32) -> Self {
        let samples = (seconds * sample_rate as f32).ceil() as usize;
        Self::new(samples.max(1))
    }

    /// Number of samples the buffer can hold.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Samples currently available to read, saturating at capacity.
    #[must_use]
    pub fn available(&self) -> usize {
        let write = self.write_pos.load(Ordering::Acquire);
        let read = self.read_pos.load(Ordering::Relaxed);
        ((write - read) as usize).min(self.capacity())
    }

    /// Total samples lost to overwrite since creation.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Total samples ever written.
    #[must_use]
    pub fn written(&self) -> u64 {
        self.write_pos.load(Ordering::Acquire)
    }

    /// Writes samples, overwriting the oldest when full. Never blocks.
    ///
    /// This is the only method the producer may call. It performs no allocation,
    /// so it is safe to invoke from a real-time audio callback.
    ///
    /// # Panics
    ///
    /// Panics before modifying any slot if this batch would advance the
    /// publication counter beyond its supported `2^63 - 1` sample horizon.
    pub fn write(&self, samples: &[f32]) {
        let mut pos = self.write_pos.load(Ordering::Relaxed);
        let sample_count = u64::try_from(samples.len()).expect("sample count must fit in u64");
        let end = pos
            .checked_add(sample_count)
            .filter(|&end| end <= Self::MAX_WRITE_POS)
            .expect("ring buffer publication counter exhausted");

        for &sample in samples {
            let slot = &self.slots[(pos as usize) & self.mask];
            let published = Self::published_stamp(pos);

            // All three slot operations are sequentially consistent. In
            // particular, a consumer that observes the new sample must also
            // observe either the in-progress marker or the new publication
            // stamp on its validation load.
            slot.stamp.store(published | 1, Ordering::SeqCst);
            slot.sample.store(sample.to_bits(), Ordering::SeqCst);
            slot.stamp.store(published, Ordering::SeqCst);
            pos += 1;

            // Publish each completed sample independently. If a large callback
            // batch laps the consumer, it can account for the loss while that
            // callback is still running rather than seeing changed slots behind
            // a stale batch-level position.
            self.write_pos.store(pos, Ordering::Release);
        }
        debug_assert_eq!(pos, end);
    }

    /// Reads up to `out.len()` samples.
    ///
    /// Returns `(samples_read, samples_dropped)`, where `samples_dropped` counts
    /// samples overwritten before this read could reach them. A non-zero drop
    /// count means the consumer was lapped and the stream has a gap.
    ///
    /// This is the only method the consumer may call.
    pub fn read(&self, out: &mut [f32]) -> (usize, u64) {
        self.read_inner(out, |_| {}, |_| {})
    }

    fn read_inner<F, G>(
        &self,
        out: &mut [f32],
        mut after_stamp: F,
        mut after_sample: G,
    ) -> (usize, u64)
    where
        F: FnMut(u64),
        G: FnMut(u64),
    {
        let mut write = self.write_pos.load(Ordering::Acquire);
        let mut read = self.read_pos.load(Ordering::Relaxed);
        let capacity = self.capacity() as u64;

        // If the producer has lapped us, skip forward to the oldest sample that
        // still exists. Reporting the gap is the point: a silent skip would look
        // like clean audio with a discontinuity in it.
        let mut dropped = 0u64;
        if write.saturating_sub(read) > capacity {
            let oldest = write - capacity;
            dropped += oldest - read;
            read = oldest;
        }

        let mut count = 0;
        while count < out.len() {
            if read >= write {
                write = self.write_pos.load(Ordering::Acquire);
                if read >= write {
                    break;
                }
            }

            let slot = &self.slots[(read as usize) & self.mask];
            let expected = Self::published_stamp(read);
            let before = slot.stamp.load(Ordering::SeqCst);
            after_stamp(read);

            if before == expected {
                let bits = slot.sample.load(Ordering::SeqCst);
                after_sample(read);
                let after = slot.stamp.load(Ordering::SeqCst);
                if after == expected {
                    out[count] = f32::from_bits(bits);
                    count += 1;
                    read += 1;
                    continue;
                }
            }

            // The producer touched this slot while it was being inspected. A
            // newly published position tells us exactly how far the consumer
            // was lapped. If publication is still in progress, return without
            // consuming the ambiguous slot; the next call will retry it or
            // account for it once the producer publishes.
            write = self.write_pos.load(Ordering::Acquire);
            let oldest = write.saturating_sub(capacity);
            if read < oldest {
                if count > 0 {
                    // Account for the known gap now, but return the contiguous
                    // prefix already copied. The next read resumes at `oldest`.
                    dropped += oldest - read;
                    read = oldest;
                    break;
                }
                dropped += oldest - read;
                read = oldest;
            } else {
                break;
            }
        }

        self.read_pos.store(read, Ordering::Release);
        self.dropped.fetch_add(dropped, Ordering::Relaxed);
        (count, dropped)
    }

    #[inline]
    const fn published_stamp(position: u64) -> u64 {
        debug_assert!(position < Self::MAX_WRITE_POS);
        (position + 1) * 2
    }

    #[cfg(test)]
    fn read_after_stamp<F>(&self, out: &mut [f32], after_stamp: F) -> (usize, u64)
    where
        F: FnMut(u64),
    {
        self.read_inner(out, after_stamp, |_| {})
    }

    #[cfg(test)]
    fn read_after_sample<F>(&self, out: &mut [f32], after_sample: F) -> (usize, u64)
    where
        F: FnMut(u64),
    {
        self.read_inner(out, |_| {}, after_sample)
    }

    /// Discards all buffered samples without reading them.
    pub fn clear(&self) {
        let write = self.write_pos.load(Ordering::Acquire);
        self.read_pos.store(write, Ordering::Release);
    }
}

// `Send` and `Sync` are derived automatically: every field is an atomic, so no
// `unsafe impl` is needed. Correctness under concurrent use rests on the SPSC
// contract documented on `read` and `write`, not on a manual marker.
const _: fn() = || {
    fn assert_shareable<T: Send + Sync>() {}
    assert_shareable::<RingBuffer>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_rounds_up_to_a_power_of_two() {
        assert_eq!(RingBuffer::new(3).capacity(), 4);
        assert_eq!(RingBuffer::new(1000).capacity(), 1024);
        assert_eq!(RingBuffer::new(1024).capacity(), 1024);
    }

    #[test]
    fn duration_constructor_sizes_for_the_sample_rate() {
        let ring = RingBuffer::with_duration(1.0, 32_000);
        assert!(ring.capacity() >= 32_000);
    }

    #[test]
    fn write_then_read_round_trips() {
        let ring = RingBuffer::new(16);
        ring.write(&[1.0, -2.5, 3.25]);

        let mut out = [0.0f32; 4];
        let (count, dropped) = ring.read(&mut out);

        assert_eq!(count, 3);
        assert_eq!(dropped, 0);
        assert_eq!(&out[..3], &[1.0, -2.5, 3.25]);
    }

    #[test]
    fn reading_an_empty_buffer_yields_nothing() {
        let ring = RingBuffer::new(8);
        let mut out = [0.0f32; 4];
        assert_eq!(ring.read(&mut out), (0, 0));
    }

    #[test]
    fn overwrite_reports_the_gap_and_keeps_the_newest_samples() {
        let ring = RingBuffer::new(4);
        // Write ten samples into a four-slot buffer: six are lost.
        let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
        ring.write(&input);

        let mut out = [0.0f32; 4];
        let (count, dropped) = ring.read(&mut out);

        assert_eq!(count, 4);
        assert_eq!(dropped, 6, "six samples were overwritten before being read");
        assert_eq!(out, [6.0, 7.0, 8.0, 9.0], "the newest samples survive");
        assert_eq!(ring.dropped(), 6);
    }

    #[test]
    fn overwrite_between_slot_validation_and_copy_is_detected() {
        let ring = RingBuffer::new(4);
        ring.write(&[0.0, 1.0, 2.0, 3.0]);

        let mut overwrote = false;
        let mut out = [f32::NAN; 4];
        let (count, dropped) = ring.read_after_stamp(&mut out, |position| {
            if position == 0 && !overwrote {
                overwrote = true;
                // This is the precise race that position-only validation
                // misses: the consumer already captured write_pos == 4 when
                // the producer replaces logical position 0 with position 4.
                ring.write(&[4.0]);
            }
        });

        assert_eq!(count, 4);
        assert_eq!(dropped, 1);
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(ring.dropped(), 1);
    }

    #[test]
    fn overwrite_between_copy_and_second_validation_is_detected() {
        let ring = RingBuffer::new(4);
        ring.write(&[0.0, 1.0, 2.0, 3.0]);

        let mut overwrote = false;
        let mut out = [f32::NAN; 4];
        let (count, dropped) = ring.read_after_sample(&mut out, |position| {
            if position == 0 && !overwrote {
                overwrote = true;
                ring.write(&[4.0]);
            }
        });

        assert_eq!(count, 4);
        assert_eq!(dropped, 1);
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(ring.dropped(), 1);
    }

    #[test]
    fn overwrite_after_a_prefix_reports_gap_without_mixing_slices() {
        let ring = RingBuffer::new(4);
        ring.write(&[0.0, 1.0, 2.0, 3.0]);

        let mut overwrote = false;
        let mut prefix = [f32::NAN; 4];
        let (count, dropped) = ring.read_after_stamp(&mut prefix, |position| {
            if position == 2 && !overwrote {
                overwrote = true;
                ring.write(&[4.0, 5.0, 6.0]);
            }
        });

        assert_eq!(count, 2);
        assert_eq!(dropped, 1);
        assert_eq!(&prefix[..count], &[0.0, 1.0]);
        assert_eq!(ring.dropped(), 1);

        let mut suffix = [f32::NAN; 4];
        let (count, dropped) = ring.read(&mut suffix);
        assert_eq!(count, 4);
        assert_eq!(dropped, 0, "the gap was already reported by the prior read");
        assert_eq!(suffix, [3.0, 4.0, 5.0, 6.0]);
        assert_eq!(ring.dropped(), 1, "the gap must not be counted twice");
    }

    #[test]
    fn final_supported_publication_remains_readable() {
        let ring = RingBuffer::new(4);
        let final_position = RingBuffer::MAX_WRITE_POS - 1;
        ring.write_pos.store(final_position, Ordering::Relaxed);
        ring.read_pos.store(final_position, Ordering::Relaxed);

        ring.write(&[42.0]);

        let mut out = [f32::NAN; 1];
        assert_eq!(ring.read(&mut out), (1, 0));
        assert_eq!(out, [42.0]);
        assert_eq!(ring.written(), RingBuffer::MAX_WRITE_POS);
    }

    #[test]
    #[should_panic(expected = "ring buffer publication counter exhausted")]
    fn publication_past_the_supported_horizon_is_rejected() {
        let ring = RingBuffer::new(4);
        ring.write_pos
            .store(RingBuffer::MAX_WRITE_POS, Ordering::Relaxed);
        ring.write(&[1.0]);
    }

    #[test]
    fn available_saturates_at_capacity() {
        let ring = RingBuffer::new(4);
        ring.write(&[0.0; 100]);
        assert_eq!(ring.available(), 4);
    }

    #[test]
    fn interleaved_writes_and_reads_preserve_order() {
        let ring = RingBuffer::new(8);
        let mut out = [0.0f32; 2];
        let mut received = Vec::new();

        for chunk in 0..10 {
            ring.write(&[chunk as f32 * 2.0, chunk as f32 * 2.0 + 1.0]);
            let (count, dropped) = ring.read(&mut out);
            assert_eq!(dropped, 0, "consumer keeps up, so nothing should drop");
            received.extend_from_slice(&out[..count]);
        }

        let expected: Vec<f32> = (0..20).map(|i| i as f32).collect();
        assert_eq!(received, expected);
    }

    #[test]
    fn clear_discards_pending_samples() {
        let ring = RingBuffer::new(8);
        ring.write(&[1.0, 2.0, 3.0]);
        ring.clear();

        let mut out = [0.0f32; 4];
        assert_eq!(ring.read(&mut out).0, 0);
    }

    #[test]
    fn survives_a_concurrent_producer_and_consumer() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        let ring = Arc::new(RingBuffer::new(1024));
        let done = Arc::new(AtomicBool::new(false));
        const TOTAL: usize = 100_000;

        let producer = {
            let ring = Arc::clone(&ring);
            let done = Arc::clone(&done);
            std::thread::spawn(move || {
                for i in 0..TOTAL {
                    ring.write(&[i as f32]);
                }
                done.store(true, Ordering::Release);
            })
        };

        let consumer = {
            let ring = Arc::clone(&ring);
            let done = Arc::clone(&done);
            std::thread::spawn(move || {
                let mut out = vec![0.0f32; 256];
                let mut seen = 0u64;
                let mut lost = 0u64;
                let mut next = 0u64;
                loop {
                    let (count, dropped) = ring.read(&mut out);
                    seen += count as u64;
                    lost += dropped;
                    next += dropped;
                    for &sample in &out[..count] {
                        assert_eq!(sample, next as f32, "samples must retain stream order");
                        next += 1;
                    }
                    if count == 0 && done.load(Ordering::Acquire) && ring.available() == 0 {
                        break;
                    }
                }
                (seen, lost, next)
            })
        };

        producer.join().unwrap();
        let (seen, lost, next) = consumer.join().unwrap();

        // Every sample is either consumed or accounted for as dropped. Nothing
        // may vanish silently, and nothing may be counted twice.
        assert_eq!(
            seen + lost,
            TOTAL as u64,
            "consumed {seen} + dropped {lost} should equal {TOTAL}"
        );
        assert_eq!(next, TOTAL as u64);
    }
}
