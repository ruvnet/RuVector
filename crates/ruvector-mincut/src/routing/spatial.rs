use super::{RoutingError, MAX_NODES};
const EARTH_M: f64 = 6_371_008.8;
#[derive(Debug, Clone, Copy)]
pub struct Snap {
    pub node: u32,
    pub distance_m: f64,
}
#[derive(Clone, Copy)]
struct Point {
    xyz: [f64; 3],
    id: u32,
}
/// Balanced 3D k-d tree on unit-sphere coordinates. Handles poles/date line.
/// Returns the nearest vertex, not the nearest point on a road segment.
pub struct MapIndex {
    points: Vec<Point>,
}
fn xyz(lat: f64, lon: f64) -> Result<[f64; 3], RoutingError> {
    if !lat.is_finite()
        || !lon.is_finite()
        || !(-90.0..=90.0).contains(&lat)
        || !(-180.0..=180.0).contains(&lon)
    {
        return Err(RoutingError::InvalidInput("latitude or longitude"));
    }
    let (a, b) = (lat.to_radians(), lon.to_radians());
    Ok([a.cos() * b.cos(), a.cos() * b.sin(), a.sin()])
}
impl MapIndex {
    pub fn new(lat_lon: &[f64]) -> Result<Self, RoutingError> {
        if lat_lon.len() % 2 != 0 || lat_lon.len() / 2 > MAX_NODES {
            return Err(RoutingError::InvalidInput("coordinate count"));
        }
        let mut points = Vec::with_capacity(lat_lon.len() / 2);
        for (id, c) in lat_lon.chunks_exact(2).enumerate() {
            points.push(Point {
                xyz: xyz(c[0], c[1])?,
                id: id as u32,
            });
        }
        fn build(p: &mut [Point], axis: usize) {
            if p.is_empty() {
                return;
            }
            let mid = p.len() / 2;
            p.select_nth_unstable_by(mid, |a, b| {
                a.xyz[axis].total_cmp(&b.xyz[axis]).then(a.id.cmp(&b.id))
            });
            let (l, r) = p.split_at_mut(mid);
            build(l, (axis + 1) % 3);
            build(&mut r[1..], (axis + 1) % 3);
        }
        build(&mut points, 0);
        Ok(Self { points })
    }
    pub fn nearest(&self, lat: f64, lon: f64, radius_m: f64) -> Result<Option<Snap>, RoutingError> {
        let q = xyz(lat, lon)?;
        if !radius_m.is_finite() || radius_m < 0.0 {
            return Err(RoutingError::InvalidInput("radius"));
        }
        let angle = (radius_m / EARTH_M).min(std::f64::consts::PI);
        let mut best = (4.0 * (angle / 2.0).sin().powi(2), u32::MAX);
        fn visit(p: &[Point], axis: usize, q: &[f64; 3], best: &mut (f64, u32)) {
            if p.is_empty() {
                return;
            }
            let mid = p.len() / 2;
            let point = p[mid];
            let d = (0..3).map(|i| (q[i] - point.xyz[i]).powi(2)).sum::<f64>();
            if d < best.0 || (d == best.0 && point.id < best.1) {
                *best = (d, point.id);
            }
            let delta = q[axis] - point.xyz[axis];
            let (near, far) = if delta <= 0.0 {
                (&p[..mid], &p[mid + 1..])
            } else {
                (&p[mid + 1..], &p[..mid])
            };
            visit(near, (axis + 1) % 3, q, best);
            if delta * delta <= best.0 {
                visit(far, (axis + 1) % 3, q, best);
            }
        }
        visit(&self.points, 0, &q, &mut best);
        Ok((best.1 != u32::MAX).then(|| Snap {
            node: best.1,
            distance_m: 2.0 * EARTH_M * (best.0.sqrt() / 2.0).min(1.0).asin(),
        }))
    }
}
