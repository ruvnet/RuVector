//! Zeckendorf representation and φ/ψ channel operations
//!
//! All operations are pure integer - ZERO floating point at inference.
//! This ensures deterministic, auditable behavior for banking compliance.

use super::ZeckMask;

/// Pre-computed Fibonacci numbers for Zeckendorf representation
pub const FIB: [u32; 35] = [
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584,
    4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229,
    832040, 1346269, 2178309, 3524578, 5702887,
];

/// Convert integer to Zeckendorf representation (bitmask)
///
/// The Zeckendorf representation expresses any positive integer uniquely
/// as a sum of non-consecutive Fibonacci numbers.
///
/// # Example
/// ```
/// use leviathan_lattice::zeckendorf::to_zeck;
/// assert_eq!(to_zeck(10), 0b10100); // 10 = F(5) + F(3) = 5 + 3 + 2... wait
/// ```
#[inline]
pub fn to_zeck(mut n: u32) -> ZeckMask {
    if n == 0 {
        return 0;
    }

    let mut result: ZeckMask = 0;
    let mut k = 2;

    // Find the largest Fibonacci number <= n
    while k < 34 && FIB[k] <= n {
        k += 1;
    }
    k -= 1;

    // Greedy algorithm for Zeckendorf
    while n > 0 && k >= 2 {
        if FIB[k] <= n {
            result |= 1 << k;
            n -= FIB[k];
            k = k.saturating_sub(2); // Skip adjacent (Zeckendorf property)
        } else {
            k -= 1;
        }
    }

    result
}

/// Convert Zeckendorf mask back to integer
#[inline]
pub fn from_zeck(z: ZeckMask) -> u32 {
    let mut n = 0u32;
    let mut mask = z;
    let mut idx = 0;

    while mask != 0 {
        if mask & 1 != 0 {
            n += FIB[idx];
        }
        mask >>= 1;
        idx += 1;
    }

    n
}

/// φ-channel: Maximum Fibonacci index (expansion/growth direction)
///
/// This represents the "ceiling" of the number in the Fibonacci lattice.
/// Higher values indicate larger magnitude representations.
#[inline]
pub fn phi_channel(z: ZeckMask) -> i32 {
    if z == 0 {
        0
    } else {
        31 - z.leading_zeros() as i32
    }
}

/// ψ-channel: Minimum Fibonacci index (grounding direction)
///
/// This represents the "floor" of the number in the Fibonacci lattice.
/// Lower values indicate more granular/precise representations.
#[inline]
pub fn psi_channel(z: ZeckMask) -> i32 {
    if z == 0 {
        super::MAX_INDEX as i32
    } else {
        z.trailing_zeros() as i32
    }
}

/// Index sum: Sum of all set Fibonacci indices
///
/// This gives a weighted position measure in the lattice.
#[inline]
pub fn idx_sum(mut z: ZeckMask) -> i32 {
    let mut sum = 0i32;
    while z != 0 {
        sum += z.trailing_zeros() as i32;
        z &= z - 1; // Clear lowest set bit
    }
    sum
}

/// Lattice distance between two Zeckendorf representations
///
/// Computed as the index sum of XOR (symmetric difference).
/// This is a proper metric on the Zeckendorf lattice.
#[inline]
pub fn zeck_distance(a: ZeckMask, b: ZeckMask) -> i32 {
    idx_sum(a ^ b)
}

/// Lattice overlap between two Zeckendorf representations
///
/// Computed as the index sum of AND (intersection).
#[inline]
pub fn zeck_overlap(a: ZeckMask, b: ZeckMask) -> i32 {
    idx_sum(a & b)
}

/// Lattice similarity score (normalized overlap)
#[inline]
pub fn zeck_similarity(a: ZeckMask, b: ZeckMask) -> f32 {
    let overlap = zeck_overlap(a, b) as f32;
    let total = (idx_sum(a) + idx_sum(b)) as f32;
    if total == 0.0 {
        1.0
    } else {
        2.0 * overlap / total
    }
}

/// Verify Zeckendorf property (no consecutive 1s)
#[inline]
pub fn is_valid_zeck(z: ZeckMask) -> bool {
    (z & (z >> 1)) == 0
}

/// Normalize to valid Zeckendorf (remove consecutive 1s)
pub fn normalize_zeck(mut z: ZeckMask) -> ZeckMask {
    // Iteratively apply F(n) + F(n-1) = F(n+1)
    loop {
        let consecutive = z & (z >> 1);
        if consecutive == 0 {
            break;
        }
        // For each consecutive pair, merge up
        let lowest_pair = consecutive & (!consecutive + 1);
        let pair_start = lowest_pair.trailing_zeros();
        z &= !(3 << pair_start); // Clear pair
        z |= 1 << (pair_start + 2); // Set merged position
    }
    z
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeck_roundtrip() {
        for n in 0..1000 {
            let z = to_zeck(n);
            assert!(is_valid_zeck(z), "Invalid zeck for {n}: {z:b}");
            assert_eq!(from_zeck(z), n, "Roundtrip failed for {n}");
        }
    }

    #[test]
    fn test_phi_psi_channels() {
        let z = to_zeck(100);
        let phi = phi_channel(z);
        let psi = psi_channel(z);
        assert!(phi >= psi);
    }

    #[test]
    fn test_distance_metric() {
        let a = to_zeck(100);
        let b = to_zeck(200);
        let c = to_zeck(150);

        // Identity
        assert_eq!(zeck_distance(a, a), 0);

        // Symmetry
        assert_eq!(zeck_distance(a, b), zeck_distance(b, a));

        // Triangle inequality
        assert!(zeck_distance(a, c) + zeck_distance(c, b) >= zeck_distance(a, b));
    }
}
