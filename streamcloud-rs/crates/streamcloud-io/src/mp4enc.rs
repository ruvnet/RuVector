//! Streaming H.264 → MP4 encoding (feature `mp4`).
//!
//! Frames are encoded with [`openh264`] (bundled C source, built via `cc`) and
//! muxed into an MP4 with the pure-Rust [`mp4`] crate as they arrive. The MP4
//! container needs `Seek` to back-patch box sizes, so output goes to a file that
//! is finalized in [`Mp4Sink::finish`]; encoding itself is incremental.

use crate::{FrameSink, IoError};
use mp4::{AvcConfig, MediaConfig, Mp4Config, Mp4Sample, Mp4Writer, TrackConfig, TrackType};
use openh264::encoder::{Encoder, EncoderConfig};
use openh264::formats::{RgbSliceU8, YUVBuffer};
use openh264::OpenH264API;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

const TIMESCALE: u32 = 1000;

/// A [`FrameSink`] that writes a streaming H.264 MP4.
pub struct Mp4Sink {
    encoder: Encoder,
    writer: Option<Mp4Writer<BufWriter<File>>>,
    path: PathBuf,
    fps: u32,
    width: u16,
    height: u16,
    frame_index: u64,
    rgb_scratch: Vec<u8>,
}

impl Mp4Sink {
    /// Create an MP4 sink. Width and height must be even (YUV 4:2:0) and fixed
    /// for the whole stream.
    pub fn new(path: impl Into<PathBuf>, fps: u32) -> Result<Self, IoError> {
        let config = EncoderConfig::new().max_frame_rate(fps as f32);
        let encoder = Encoder::with_api_config(OpenH264API::from_source(), config)
            .map_err(|e| IoError::Encoder(format!("openh264 init: {e}")))?;
        Ok(Self {
            encoder,
            writer: None,
            path: path.into(),
            fps: fps.max(1),
            width: 0,
            height: 0,
            frame_index: 0,
            rgb_scratch: Vec::new(),
        })
    }

    fn frame_duration(&self) -> u32 {
        (TIMESCALE / self.fps).max(1)
    }
}

/// Iterate annex-B NAL units (payload after each start code) in `data`.
fn nal_units(data: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::with_capacity(starts.len());
    for (idx, &s) in starts.iter().enumerate() {
        // End at the byte before the next start code's leading zeros.
        let end = if idx + 1 < starts.len() {
            let next = starts[idx + 1] - 3;
            // trim a trailing zero that belonged to a 4-byte start code
            if next > s && data[next - 1] == 0 {
                next - 1
            } else {
                next
            }
        } else {
            data.len()
        };
        if end > s {
            out.push(&data[s..end]);
        }
    }
    out
}

#[inline]
fn nal_type(nal: &[u8]) -> u8 {
    nal.first().map(|b| b & 0x1f).unwrap_or(0)
}

impl FrameSink for Mp4Sink {
    fn push(&mut self, rgba: &[u8], width: usize, height: usize) -> Result<(), IoError> {
        // RGBA -> RGB scratch.
        let px = width * height;
        self.rgb_scratch.clear();
        self.rgb_scratch.reserve(px * 3);
        for p in 0..px {
            let i = p * 4;
            self.rgb_scratch.extend_from_slice(&rgba[i..i + 3]);
        }

        let yuv = YUVBuffer::from_rgb_source(RgbSliceU8::new(&self.rgb_scratch, (width, height)));
        let bitstream = self
            .encoder
            .encode(&yuv)
            .map_err(|e| IoError::Encoder(format!("encode: {e}")))?;
        let annexb = bitstream.to_vec();
        let nals = nal_units(&annexb);

        // Lazily create the writer once we have SPS/PPS + dimensions.
        if self.writer.is_none() {
            if width % 2 != 0 || height % 2 != 0 {
                return Err(IoError::Encoder(format!(
                    "dimensions must be even for H.264, got {width}x{height}"
                )));
            }
            let sps = nals
                .iter()
                .find(|n| nal_type(n) == 7)
                .ok_or_else(|| IoError::Encoder("no SPS in first frame".into()))?
                .to_vec();
            let pps = nals
                .iter()
                .find(|n| nal_type(n) == 8)
                .ok_or_else(|| IoError::Encoder("no PPS in first frame".into()))?
                .to_vec();
            self.width = width as u16;
            self.height = height as u16;

            let file = File::create(&self.path)?;
            let mp4_cfg = Mp4Config {
                major_brand: (*b"isom").into(),
                minor_version: 512,
                compatible_brands: vec![(*b"isom").into(), (*b"avc1").into()],
                timescale: TIMESCALE,
            };
            let mut writer = Mp4Writer::write_start(BufWriter::new(file), &mp4_cfg)
                .map_err(|e| IoError::Encoder(format!("mp4 start: {e}")))?;
            let track = TrackConfig {
                track_type: TrackType::Video,
                timescale: TIMESCALE,
                language: "und".to_string(),
                media_conf: MediaConfig::AvcConfig(AvcConfig {
                    width: self.width,
                    height: self.height,
                    seq_param_set: sps,
                    pic_param_set: pps,
                }),
            };
            writer
                .add_track(&track)
                .map_err(|e| IoError::Encoder(format!("add_track: {e}")))?;
            self.writer = Some(writer);
        } else if width as u16 != self.width || height as u16 != self.height {
            return Err(IoError::SizeChanged {
                expected: (self.width as usize, self.height as usize),
                got: (width, height),
            });
        }

        // Build the sample: length-prefixed VCL NALs (types 1 and 5).
        let mut sample_bytes = Vec::new();
        let mut is_sync = false;
        for nal in &nals {
            let t = nal_type(nal);
            if t == 5 {
                is_sync = true;
            }
            if t == 1 || t == 5 {
                sample_bytes.extend_from_slice(&(nal.len() as u32).to_be_bytes());
                sample_bytes.extend_from_slice(nal);
            }
        }
        // Some frames (rare) may carry only param-set refresh; skip empties.
        if !sample_bytes.is_empty() {
            let dur = self.frame_duration();
            let sample = Mp4Sample {
                start_time: self.frame_index * dur as u64,
                duration: dur,
                rendering_offset: 0,
                is_sync,
                bytes: mp4::Bytes::from(sample_bytes),
            };
            self.writer
                .as_mut()
                .unwrap()
                .write_sample(1, &sample)
                .map_err(|e| IoError::Encoder(format!("write_sample: {e}")))?;
            self.frame_index += 1;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), IoError> {
        if let Some(mut writer) = self.writer.take() {
            writer
                .write_end()
                .map_err(|e| IoError::Encoder(format!("mp4 end: {e}")))?;
        }
        Ok(())
    }
}

/// Convenience: number of frames committed to the MP4.
impl Mp4Sink {
    /// Frames written so far.
    pub fn frame_count(&self) -> u64 {
        self.frame_index
    }
    /// Output path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: usize, h: usize, t: u8) -> Vec<u8> {
        let mut b = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let r = ((x + t as usize) % 256) as u8;
                let g = ((y + t as usize) % 256) as u8;
                b.extend_from_slice(&[r, g, 128, 255]);
            }
        }
        b
    }

    #[test]
    fn encodes_a_playable_mp4() {
        let path = std::env::temp_dir().join(format!("streamcloud-{}.mp4", std::process::id()));
        let (w, h) = (64, 48);
        let mut sink = Mp4Sink::new(&path, 20).unwrap();
        for t in 0..8u8 {
            sink.push(&frame(w, h, t * 16), w, h).unwrap();
        }
        sink.finish().unwrap();
        assert!(sink.frame_count() >= 1);

        // File exists and is non-trivial.
        let meta = std::fs::metadata(&path).unwrap();
        assert!(meta.len() > 256, "mp4 too small: {} bytes", meta.len());

        // Read it back with the same crate to confirm a valid AVC video track.
        let f = std::fs::File::open(&path).unwrap();
        let size = f.metadata().unwrap().len();
        let reader = mp4::Mp4Reader::read_header(std::io::BufReader::new(f), size).unwrap();
        assert!(reader.tracks().values().any(|t| t.media_type().is_ok()));
        assert_eq!(reader.tracks().len(), 1);
        let _ = std::fs::remove_file(&path);
    }
}
