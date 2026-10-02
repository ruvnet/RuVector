//! # streamcloud-io
//!
//! Frame output for StreamCloud: save rendered frames as **PNG**
//! and mux a **streaming H.264 MP4** as frames are produced (~20 FPS, matching
//! the model's streaming nature).
//!
//! Everything is driven through the [`FrameSink`] trait so the pipeline is
//! encoder-agnostic: point it at a [`PngSequenceSink`], an
//! [`mp4enc::Mp4Sink`] (feature `mp4`), a [`MultiSink`], or [`NullSink`] in
//! tests.

use std::path::{Path, PathBuf};

#[cfg(feature = "mp4")]
pub mod mp4enc;

/// Errors from frame I/O.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    /// Underlying I/O failure.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// Image encode failure.
    #[error("image: {0}")]
    Image(#[from] image::ImageError),
    /// Encoder error (MP4/H.264).
    #[error("encoder: {0}")]
    Encoder(String),
    /// A frame's dimensions changed mid-stream (not allowed for video).
    #[error("frame size changed: expected {expected:?}, got {got:?}")]
    SizeChanged {
        /// First frame size.
        expected: (usize, usize),
        /// Offending frame size.
        got: (usize, usize),
    },
}

/// A consumer of rendered RGBA frames.
pub trait FrameSink {
    /// Push one row-major RGBA8 frame (`len == width * height * 4`).
    fn push(&mut self, rgba: &[u8], width: usize, height: usize) -> Result<(), IoError>;
    /// Flush/finalize. Call exactly once when the stream ends.
    fn finish(&mut self) -> Result<(), IoError>;
}

/// Save a single RGBA8 buffer as a PNG.
pub fn save_png(
    path: impl AsRef<Path>,
    rgba: &[u8],
    width: usize,
    height: usize,
) -> Result<(), IoError> {
    image::save_buffer(
        path.as_ref(),
        rgba,
        width as u32,
        height as u32,
        image::ExtendedColorType::Rgba8,
    )?;
    Ok(())
}

/// Writes each frame as `{dir}/{prefix}{index:05}.png`.
pub struct PngSequenceSink {
    dir: PathBuf,
    prefix: String,
    index: u64,
}

impl PngSequenceSink {
    /// Create, ensuring `dir` exists.
    pub fn new(dir: impl Into<PathBuf>, prefix: impl Into<String>) -> Result<Self, IoError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            prefix: prefix.into(),
            index: 0,
        })
    }

    /// Number of frames written so far.
    pub fn count(&self) -> u64 {
        self.index
    }
}

impl FrameSink for PngSequenceSink {
    fn push(&mut self, rgba: &[u8], width: usize, height: usize) -> Result<(), IoError> {
        let path = self
            .dir
            .join(format!("{}{:05}.png", self.prefix, self.index));
        save_png(path, rgba, width, height)?;
        self.index += 1;
        Ok(())
    }
    fn finish(&mut self) -> Result<(), IoError> {
        Ok(())
    }
}

/// Fan out frames to several sinks (e.g. PNG snapshots + MP4 stream).
pub struct MultiSink {
    sinks: Vec<Box<dyn FrameSink>>,
}

impl MultiSink {
    /// New fan-out sink.
    pub fn new(sinks: Vec<Box<dyn FrameSink>>) -> Self {
        Self { sinks }
    }
}

impl FrameSink for MultiSink {
    fn push(&mut self, rgba: &[u8], width: usize, height: usize) -> Result<(), IoError> {
        for s in &mut self.sinks {
            s.push(rgba, width, height)?;
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<(), IoError> {
        for s in &mut self.sinks {
            s.finish()?;
        }
        Ok(())
    }
}

/// Discards frames but counts them (tests / benchmarks).
#[derive(Default)]
pub struct NullSink {
    /// Frames seen.
    pub count: u64,
}

impl FrameSink for NullSink {
    fn push(&mut self, _rgba: &[u8], _width: usize, _height: usize) -> Result<(), IoError> {
        self.count += 1;
        Ok(())
    }
    fn finish(&mut self) -> Result<(), IoError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: usize, h: usize, v: u8) -> Vec<u8> {
        let mut b = Vec::with_capacity(w * h * 4);
        for _ in 0..w * h {
            b.extend_from_slice(&[v, v / 2, 255 - v, 255]);
        }
        b
    }

    #[test]
    fn png_sequence_writes_files() {
        let dir = std::env::temp_dir().join(format!("streamcloud-io-png-{}", std::process::id()));
        let mut sink = PngSequenceSink::new(&dir, "f").unwrap();
        for i in 0..3 {
            sink.push(&frame(8, 8, (i * 80) as u8), 8, 8).unwrap();
        }
        sink.finish().unwrap();
        assert_eq!(sink.count(), 3);
        let written = std::fs::read_dir(&dir).unwrap().count();
        assert_eq!(written, 3);
        let first = std::fs::metadata(dir.join("f00000.png")).unwrap();
        assert!(first.len() > 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn null_and_multi_sink() {
        let mut s = MultiSink::new(vec![Box::new(NullSink::default())]);
        s.push(&frame(4, 4, 100), 4, 4).unwrap();
        s.finish().unwrap();
    }
}
