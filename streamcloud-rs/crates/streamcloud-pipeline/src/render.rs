//! CPU software renderer for point clouds.
//!
//! Rasterizes a [`PointCloud`] to an RGBA8 buffer with a z-buffer and an
//! orbiting camera, so the output video shows real 3D parallax (rotating the
//! cloud reveals depth structure the input 2D frame can't). This is the
//! hermetic default path; the wgpu/WebGPU renderers (native + browser) consume
//! the same point clouds.

use streamcloud_model::PointCloud;

/// Software point-cloud rasterizer.
pub struct SoftwareRenderer {
    /// Output width.
    pub width: usize,
    /// Output height.
    pub height: usize,
    /// Background RGBA.
    pub background: [u8; 4],
    /// Half-size of the square splat (0 = single pixel).
    pub point_radius: i32,
}

impl SoftwareRenderer {
    /// New renderer with a dark background and 1px-radius splats.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            background: [12, 14, 20, 255],
            point_radius: 1,
        }
    }

    /// Render `pc` from an orbit angle (radians) about the scene's vertical axis.
    pub fn render(&self, pc: &PointCloud, view_angle: f32) -> Vec<u8> {
        let w = self.width;
        let h = self.height;
        let mut color = Vec::with_capacity(w * h * 4);
        for _ in 0..w * h {
            color.extend_from_slice(&self.background);
        }
        let mut zbuf = vec![f32::INFINITY; w * h];

        if pc.is_empty() {
            return color;
        }

        // Orbit about the cloud centroid.
        let (min, max) = pc.bounds().unwrap();
        let cx3 = 0.5 * (min[0] + max[0]);
        let cy3 = 0.5 * (min[1] + max[1]);
        let cz3 = 0.5 * (min[2] + max[2]);
        let (s, c) = view_angle.sin_cos();

        let focal = w as f32;
        let cx = w as f32 * 0.5;
        let cy = h as f32 * 0.5;

        for p in &pc.points {
            // Translate to centroid, rotate about Y, translate back.
            let dx = p.pos[0] - cx3;
            let dz = p.pos[2] - cz3;
            let rx = c * dx + s * dz;
            let rz = -s * dx + c * dz;
            let x = cx3 + rx;
            let y = p.pos[1] - cy3 + cy3; // unchanged (Y axis)
            let z = cz3 + rz;
            if z <= 0.05 {
                continue;
            }

            let sx = (focal * x / z + cx).round() as i32;
            let sy = (cy - focal * (y - cy3) / z).round() as i32;

            for oy in -self.point_radius..=self.point_radius {
                for ox in -self.point_radius..=self.point_radius {
                    let px = sx + ox;
                    let py = sy + oy;
                    if px < 0 || py < 0 || px as usize >= w || py as usize >= h {
                        continue;
                    }
                    let idx = py as usize * w + px as usize;
                    if z < zbuf[idx] {
                        zbuf[idx] = z;
                        let o = idx * 4;
                        color[o] = p.color[0];
                        color[o + 1] = p.color[1];
                        color[o + 2] = p.color[2];
                        color[o + 3] = 255;
                    }
                }
            }
        }
        color
    }

    /// Fraction of pixels that are not background (a cheap "did we draw" check).
    pub fn coverage(&self, rgba: &[u8]) -> f32 {
        let total = self.width * self.height;
        if total == 0 {
            return 0.0;
        }
        let mut drawn = 0usize;
        for i in 0..total {
            let o = i * 4;
            if rgba[o..o + 3] != self.background[..3] {
                drawn += 1;
            }
        }
        drawn as f32 / total as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use streamcloud_model::Point;

    #[test]
    fn renders_points_onto_canvas() {
        let pc = PointCloud {
            points: (0..200)
                .map(|i| Point {
                    pos: [(i % 20) as f32 * 0.1 - 1.0, (i / 20) as f32 * 0.1, 3.0],
                    color: [255, 200, 50],
                })
                .collect(),
        };
        let r = SoftwareRenderer::new(128, 96);
        let img = r.render(&pc, 0.0);
        assert_eq!(img.len(), 128 * 96 * 4);
        assert!(r.coverage(&img) > 0.0, "nothing was drawn");
    }

    #[test]
    fn orbit_changes_image() {
        let pc = PointCloud {
            points: (0..300)
                .map(|i| Point {
                    pos: [
                        (i % 30) as f32 * 0.1 - 1.5,
                        ((i / 30) % 10) as f32 * 0.1,
                        2.0 + (i % 7) as f32 * 0.2,
                    ],
                    color: [100, 180, 255],
                })
                .collect(),
        };
        let r = SoftwareRenderer::new(96, 96);
        let a = r.render(&pc, 0.0);
        let b = r.render(&pc, 0.6);
        assert_ne!(a, b, "orbit should change the rendered image");
    }
}
