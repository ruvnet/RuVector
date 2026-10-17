//! Deterministic synthetic video source.
//!
//! Generates a streaming scene (panning camera over a textured ground plane with
//! a moving highlight) so the pipeline, memory retrieval, and demos run without
//! a real video file. Replace with a real frame source (camera / image
//! sequence) to drive the pipeline on actual footage.

use crate::Frame;

/// Generate frame `index` of a deterministic synthetic scene.
pub fn synthetic_frame(index: u64, width: usize, height: usize) -> Frame {
    let mut px = Vec::with_capacity(width * height * 4);
    let t = index as f32;
    // Horizontal pan.
    let pan = t * 6.0;
    // Moving highlight center.
    let hx = width as f32 * (0.5 + 0.35 * (t * 0.12).sin());
    let hy = height as f32 * (0.45 + 0.15 * (t * 0.09).cos());

    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 + pan;
            let fy = y as f32;
            let horizon = height as f32 * 0.45;

            let (mut r, mut g, mut b);
            if (y as f32) < horizon {
                // Sky: vertical gradient.
                let v = y as f32 / horizon;
                r = (70.0 + 60.0 * v) as i32;
                g = (110.0 + 80.0 * v) as i32;
                b = (180.0 + 60.0 * v) as i32;
            } else {
                // Ground: checkerboard texture, perspective-ish row scaling.
                let depth_row = (y as f32 - horizon) / (height as f32 - horizon);
                let scale = 6.0 + depth_row * 18.0;
                let cxic = ((fx / scale) as i32 + (fy / scale) as i32) % 2;
                let base = if cxic == 0 { 90 } else { 150 };
                r = (base as f32 * (0.6 + 0.4 * depth_row)) as i32;
                g = (base as f32 * (0.55 + 0.35 * depth_row)) as i32;
                b = (base as f32 * (0.4 + 0.3 * depth_row)) as i32;
            }

            // Moving highlight (a bright blob), adds structure that memory can
            // match across frames.
            let dx = x as f32 - hx;
            let dy = y as f32 - hy;
            let d2 = dx * dx + dy * dy;
            let glow = (-d2 / (2.0 * 40.0 * 40.0)).exp();
            r += (glow * 120.0) as i32;
            g += (glow * 110.0) as i32;
            b += (glow * 60.0) as i32;

            px.push(r.clamp(0, 255) as u8);
            px.push(g.clamp(0, 255) as u8);
            px.push(b.clamp(0, 255) as u8);
            px.push(255);
        }
    }
    Frame::new(width, height, px)
}

/// An iterator over `count` synthetic frames.
pub fn synthetic_stream(count: u64, width: usize, height: usize) -> impl Iterator<Item = Frame> {
    (0..count).map(move |i| synthetic_frame(i, width, height))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_have_correct_size_and_vary() {
        let f0 = synthetic_frame(0, 64, 48);
        let f5 = synthetic_frame(5, 64, 48);
        assert_eq!(f0.width, 64);
        assert_eq!(f0.height, 48);
        assert_eq!(f0.pixels.len(), 64 * 48 * 4);
        assert_ne!(f0.pixels, f5.pixels, "scene should animate");
    }
}
