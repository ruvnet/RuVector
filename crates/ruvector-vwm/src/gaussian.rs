//! Core 4D Gaussian primitive for volumetric scene representation.
//!
//! Each [`Gaussian4D`] represents a single volumetric element with spatial position,
//! anisotropic covariance, view-dependent color (spherical harmonics), opacity,
//! and temporal deformation parameters. The linear motion model allows evaluating
//! Gaussian positions at arbitrary time values without re-fitting.
//!
//! [`ScreenGaussian`] is the result of projecting a 4D Gaussian into screen space
//! for rasterization.

/// A single 4D Gaussian primitive.
///
/// Represents a volumetric element with position, covariance, color, opacity,
/// and temporal parameters.
#[derive(Clone, Debug)]
pub struct Gaussian4D {
    /// Center position [x, y, z]
    pub center: [f32; 3],
    /// Covariance matrix (upper triangle, 6 elements: xx, xy, xz, yy, yz, zz)
    pub covariance: [f32; 6],
    /// Spherical harmonics coefficients for view-dependent color (degree 0 = 3 floats RGB)
    pub sh_coeffs: [f32; 3],
    /// Opacity in [0, 1]
    pub opacity: f32,
    /// Scale factors [sx, sy, sz]
    pub scale: [f32; 3],
    /// Rotation quaternion [w, x, y, z]
    pub rotation: [f32; 4],
    /// Temporal deformation parameters: time range [start, end]
    pub time_range: [f32; 2],
    /// Per-axis velocity for linear motion model
    pub velocity: [f32; 3],
    /// Unique ID within tile
    pub id: u32,
}

/// Screen-space projected Gaussian for rasterization.
#[derive(Clone, Debug)]
pub struct ScreenGaussian {
    /// Projected center in screen coordinates [x, y]
    pub center_screen: [f32; 2],
    /// Depth value for sorting
    pub depth: f32,
    /// Inverse 2D covariance (upper triangle: a, b, c where matrix = [[a,b],[b,c]])
    pub conic: [f32; 3],
    /// Evaluated color [r, g, b]
    pub color: [f32; 3],
    /// Opacity after depth attenuation
    pub opacity: f32,
    /// Screen-space radius for tile binning
    pub radius: f32,
    /// ID from the source Gaussian
    pub id: u32,
}

impl Gaussian4D {
    /// Create a new Gaussian at the given center with default parameters.
    pub fn new(center: [f32; 3], id: u32) -> Self {
        Self {
            center,
            covariance: [1.0, 0.0, 0.0, 1.0, 0.0, 1.0], // identity-like
            sh_coeffs: [0.5, 0.5, 0.5],                   // neutral gray
            opacity: 1.0,
            scale: [1.0, 1.0, 1.0],
            rotation: [1.0, 0.0, 0.0, 0.0], // identity quaternion
            time_range: [f32::NEG_INFINITY, f32::INFINITY],
            velocity: [0.0, 0.0, 0.0],
            id,
        }
    }

    /// Evaluate position at time `t` using the linear motion model.
    ///
    /// The temporal midpoint of the Gaussian's time range is used as the
    /// reference time. Position is: `center + velocity * (t - t_mid)`.
    /// If the time range is unbounded (midpoint is infinite or NaN), the
    /// center position is returned directly.
    pub fn position_at(&self, t: f32) -> [f32; 3] {
        let t_mid = (self.time_range[0] + self.time_range[1]) * 0.5;
        if !t_mid.is_finite() {
            return self.center;
        }
        let dt = t - t_mid;
        [
            self.center[0] + self.velocity[0] * dt,
            self.center[1] + self.velocity[1] * dt,
            self.center[2] + self.velocity[2] * dt,
        ]
    }

    /// Check if this Gaussian is active at time `t`.
    pub fn is_active_at(&self, t: f32) -> bool {
        t >= self.time_range[0] && t <= self.time_range[1]
    }

    /// Project to screen space given a 4x4 column-major view-projection matrix.
    ///
    /// Returns `None` if the Gaussian is behind the camera (w <= 0) or is not
    /// active at the given time.
    ///
    /// The projection uses the Jacobian of the projective transform to map the
    /// 3D covariance into a 2D screen-space conic. The radius is estimated from
    /// the eigenvalues of the 2D covariance.
    pub fn project(&self, view_proj: &[f32; 16], t: f32) -> Option<ScreenGaussian> {
        if !self.is_active_at(t) {
            return None;
        }

        let pos = self.position_at(t);

        // Apply view-projection (column-major multiplication)
        let x = view_proj[0] * pos[0]
            + view_proj[4] * pos[1]
            + view_proj[8] * pos[2]
            + view_proj[12];
        let y = view_proj[1] * pos[0]
            + view_proj[5] * pos[1]
            + view_proj[9] * pos[2]
            + view_proj[13];
        let z = view_proj[2] * pos[0]
            + view_proj[6] * pos[1]
            + view_proj[10] * pos[2]
            + view_proj[14];
        let w = view_proj[3] * pos[0]
            + view_proj[7] * pos[1]
            + view_proj[11] * pos[2]
            + view_proj[15];

        if w <= 1e-7 {
            return None;
        }

        let inv_w = 1.0 / w;
        let ndc_x = x * inv_w;
        let ndc_y = y * inv_w;
        let depth = z * inv_w;

        // Build 3x3 covariance from upper triangle
        // [cov_xx, cov_xy, cov_xz]
        // [cov_xy, cov_yy, cov_yz]
        // [cov_xz, cov_yz, cov_zz]
        let cov3d = [
            self.covariance[0] * self.scale[0] * self.scale[0],
            self.covariance[1] * self.scale[0] * self.scale[1],
            self.covariance[2] * self.scale[0] * self.scale[2],
            self.covariance[3] * self.scale[1] * self.scale[1],
            self.covariance[4] * self.scale[1] * self.scale[2],
            self.covariance[5] * self.scale[2] * self.scale[2],
        ];

        // Approximate Jacobian of projection at this point:
        // J = [ 1/w, 0, -x/w^2 ]
        //     [ 0, 1/w, -y/w^2 ]
        let j00 = inv_w;
        let j02 = -x * inv_w * inv_w;
        let j11 = inv_w;
        let j12 = -y * inv_w * inv_w;

        // Sigma_2d = J * Sigma_3d * J^T  (only need upper triangle)
        // Row 0 of J * Sigma_3d:
        let t0 = j00 * cov3d[0] + j02 * cov3d[2];
        let t1 = j00 * cov3d[1] + j02 * cov3d[4];
        let t2 = j00 * cov3d[2] + j02 * cov3d[5];
        // Row 1 of J * Sigma_3d:
        let t3 = j11 * cov3d[1] + j12 * cov3d[2];
        let t4 = j11 * cov3d[3] + j12 * cov3d[4];
        let t5 = j11 * cov3d[4] + j12 * cov3d[5];

        // 2D covariance upper triangle: [a, b, c] where matrix = [[a,b],[b,c]]
        let cov2d_a = t0 * j00 + t2 * j02;
        let cov2d_b = t0 * 0.0 + t1 * j11 + t2 * j12; // cross term
        let _ = t3; // used via the symmetric property
        let cov2d_c = t4 * j11 + t5 * j12;

        // Correct cross term: using J rows explicitly
        let cov2d_b_correct = t3 * j00 + t5 * j02;
        let _ = cov2d_b; // shadow with correct value
        let cov2d_b = (cov2d_b_correct + (t0 * 0.0 + t1 * j11 + t2 * j12)) * 0.5;

        // Add a small regularization to avoid singularity
        let cov2d_a = cov2d_a + 0.3;
        let cov2d_c = cov2d_c + 0.3;

        // Invert 2D covariance to get conic
        let det = cov2d_a * cov2d_c - cov2d_b * cov2d_b;
        if det.abs() < 1e-10 {
            return None;
        }
        let inv_det = 1.0 / det;
        let conic = [
            cov2d_c * inv_det,
            -cov2d_b * inv_det,
            cov2d_a * inv_det,
        ];

        // Estimate screen-space radius from eigenvalues of 2D covariance
        let mid = (cov2d_a + cov2d_c) * 0.5;
        let disc = ((cov2d_a - cov2d_c) * 0.5).powi(2) + cov2d_b * cov2d_b;
        let lambda_max = mid + disc.max(0.0).sqrt();
        // 3-sigma radius
        let radius = (3.0 * lambda_max.max(0.0).sqrt()).max(0.1);

        Some(ScreenGaussian {
            center_screen: [ndc_x, ndc_y],
            depth,
            conic,
            color: self.sh_coeffs,
            opacity: self.opacity,
            radius,
            id: self.id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_gaussian() {
        let g = Gaussian4D::new([1.0, 2.0, 3.0], 42);
        assert_eq!(g.center, [1.0, 2.0, 3.0]);
        assert_eq!(g.id, 42);
        assert_eq!(g.opacity, 1.0);
        assert_eq!(g.rotation, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_position_at_stationary() {
        let g = Gaussian4D::new([1.0, 2.0, 3.0], 0);
        let pos = g.position_at(5.0);
        // velocity is zero, so position should be center regardless of t
        assert_eq!(pos, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_position_at_with_velocity() {
        let mut g = Gaussian4D::new([0.0, 0.0, 0.0], 0);
        g.velocity = [1.0, 2.0, 3.0];
        g.time_range = [0.0, 10.0];
        // t_mid = 5.0, so at t=7.0, dt=2.0
        let pos = g.position_at(7.0);
        assert!((pos[0] - 2.0).abs() < 1e-6);
        assert!((pos[1] - 4.0).abs() < 1e-6);
        assert!((pos[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_is_active_at() {
        let mut g = Gaussian4D::new([0.0, 0.0, 0.0], 0);
        g.time_range = [1.0, 5.0];
        assert!(!g.is_active_at(0.5));
        assert!(g.is_active_at(1.0));
        assert!(g.is_active_at(3.0));
        assert!(g.is_active_at(5.0));
        assert!(!g.is_active_at(5.5));
    }

    #[test]
    fn test_project_inactive_returns_none() {
        let mut g = Gaussian4D::new([0.0, 0.0, -5.0], 0);
        g.time_range = [0.0, 1.0];
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        assert!(g.project(&identity, 2.0).is_none());
    }

    #[test]
    fn test_project_identity_matrix() {
        let g = Gaussian4D::new([0.5, 0.5, 0.5], 7);
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let sg = g.project(&identity, 0.0);
        assert!(sg.is_some());
        let sg = sg.unwrap();
        assert_eq!(sg.id, 7);
        assert!((sg.center_screen[0] - 0.5).abs() < 1e-3);
        assert!((sg.center_screen[1] - 0.5).abs() < 1e-3);
    }
}
