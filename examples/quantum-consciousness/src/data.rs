//! Quantum circuit data: generate measurement statistics as TPMs.
//!
//! For each quantum circuit, we compute the output state vector, then
//! construct a TPM where TPM[i][j] = P(measure outcome j | input basis state i).
//! For unitary circuits, this is |<j|U|i>|^2.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// A quantum circuit with its measurement TPM.
pub struct QuantumCircuit {
    pub name: String,
    pub n_qubits: usize,
    pub description: String,
    /// Row-major TPM: P(outcome_j | input_i), dimension 2^n x 2^n.
    pub tpm: Vec<f64>,
    /// The unitary matrix (row-major, complex: stored as separate re/im vecs).
    #[allow(dead_code)]
    pub unitary_re: Vec<f64>,
    #[allow(dead_code)]
    pub unitary_im: Vec<f64>,
}

impl QuantumCircuit {
    pub fn tpm_size(&self) -> usize {
        1 << self.n_qubits
    }
}

/// Generate all quantum circuits for analysis.
pub fn generate_all_circuits(random_depth: usize) -> Vec<QuantumCircuit> {
    vec![
        generate_bell_state(),
        generate_ghz_state(),
        generate_product_state(),
        generate_w_state(),
        generate_random_circuit(random_depth),
    ]
}

// -------------------------------------------------------------------
// Unitary matrix helpers (2^n x 2^n complex matrices stored as re/im)
// -------------------------------------------------------------------

/// Identity matrix for n qubits.
fn identity(n_qubits: usize) -> (Vec<f64>, Vec<f64>) {
    let dim = 1 << n_qubits;
    let mut re = vec![0.0; dim * dim];
    let im = vec![0.0; dim * dim];
    for i in 0..dim {
        re[i * dim + i] = 1.0;
    }
    (re, im)
}

/// Multiply two complex matrices C = A * B (dim x dim).
fn matmul(
    a_re: &[f64],
    a_im: &[f64],
    b_re: &[f64],
    b_im: &[f64],
    dim: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut c_re = vec![0.0; dim * dim];
    let mut c_im = vec![0.0; dim * dim];
    for i in 0..dim {
        for k in 0..dim {
            let ar = a_re[i * dim + k];
            let ai = a_im[i * dim + k];
            if ar.abs() < 1e-15 && ai.abs() < 1e-15 {
                continue;
            }
            for j in 0..dim {
                let br = b_re[k * dim + j];
                let bi = b_im[k * dim + j];
                c_re[i * dim + j] += ar * br - ai * bi;
                c_im[i * dim + j] += ar * bi + ai * br;
            }
        }
    }
    (c_re, c_im)
}

/// Apply a 2-qubit gate (4x4 unitary) to qubits (q0, q1) in an n-qubit system.
/// q0 is the control (higher-order bit in the 2-qubit subspace), q1 is the target.
fn apply_two_qubit_gate(
    u_re: &mut Vec<f64>,
    u_im: &mut Vec<f64>,
    gate_re: &[f64; 16],
    gate_im: &[f64; 16],
    n_qubits: usize,
    q0: usize,
    q1: usize,
) {
    let dim = 1 << n_qubits;
    // Build the full-size gate by tensoring with identities
    let mut full_re = vec![0.0; dim * dim];
    let mut full_im = vec![0.0; dim * dim];

    for row in 0..dim {
        for col in 0..dim {
            // Extract the 2-bit index for (q0, q1)
            let r0 = (row >> (n_qubits - 1 - q0)) & 1;
            let r1 = (row >> (n_qubits - 1 - q1)) & 1;
            let c0 = (col >> (n_qubits - 1 - q0)) & 1;
            let c1 = (col >> (n_qubits - 1 - q1)) & 1;

            // Check that all other qubits match
            let mut other_match = true;
            for q in 0..n_qubits {
                if q != q0 && q != q1 {
                    if ((row >> (n_qubits - 1 - q)) & 1) != ((col >> (n_qubits - 1 - q)) & 1) {
                        other_match = false;
                        break;
                    }
                }
            }

            if other_match {
                let gi = (r0 * 2 + r1) * 4 + (c0 * 2 + c1);
                full_re[row * dim + col] = gate_re[gi];
                full_im[row * dim + col] = gate_im[gi];
            }
        }
    }

    let (new_re, new_im) = matmul(&full_re, &full_im, u_re, u_im, dim);
    *u_re = new_re;
    *u_im = new_im;
}

/// Apply a single-qubit gate (2x2 unitary) to qubit q in an n-qubit system.
fn apply_single_qubit_gate(
    u_re: &mut Vec<f64>,
    u_im: &mut Vec<f64>,
    gate_re: &[f64; 4],
    gate_im: &[f64; 4],
    n_qubits: usize,
    q: usize,
) {
    let dim = 1 << n_qubits;
    let mut full_re = vec![0.0; dim * dim];
    let mut full_im = vec![0.0; dim * dim];

    for row in 0..dim {
        for col in 0..dim {
            let rq = (row >> (n_qubits - 1 - q)) & 1;
            let cq = (col >> (n_qubits - 1 - q)) & 1;

            // All other qubits must match (identity)
            let mut other_match = true;
            for qq in 0..n_qubits {
                if qq != q {
                    if ((row >> (n_qubits - 1 - qq)) & 1) != ((col >> (n_qubits - 1 - qq)) & 1) {
                        other_match = false;
                        break;
                    }
                }
            }

            if other_match {
                let gi = rq * 2 + cq;
                full_re[row * dim + col] = gate_re[gi];
                full_im[row * dim + col] = gate_im[gi];
            }
        }
    }

    let (new_re, new_im) = matmul(&full_re, &full_im, u_re, u_im, dim);
    *u_re = new_re;
    *u_im = new_im;
}

/// Convert a unitary matrix to a TPM: TPM[i][j] = |U[j][i]|^2 = |<j|U|i>|^2
/// Note: U|i> gives column i of U, so P(j|i) = |U[j,i]|^2.
fn unitary_to_tpm(u_re: &[f64], u_im: &[f64], dim: usize) -> Vec<f64> {
    let mut tpm = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let re = u_re[j * dim + i];
            let im = u_im[j * dim + i];
            tpm[i * dim + j] = re * re + im * im;
        }
    }
    // Row-normalize to correct for floating point
    for i in 0..dim {
        let sum: f64 = (0..dim).map(|j| tpm[i * dim + j]).sum();
        if sum > 1e-30 {
            for j in 0..dim {
                tpm[i * dim + j] /= sum;
            }
        }
    }
    tpm
}

// -------------------------------------------------------------------
// Standard gates
// -------------------------------------------------------------------

/// Hadamard gate
const H_RE: [f64; 4] = [
    std::f64::consts::FRAC_1_SQRT_2,
    std::f64::consts::FRAC_1_SQRT_2,
    std::f64::consts::FRAC_1_SQRT_2,
    -std::f64::consts::FRAC_1_SQRT_2,
];
const H_IM: [f64; 4] = [0.0, 0.0, 0.0, 0.0];

/// CNOT gate (control=0, target=1 in 2-qubit basis |00>, |01>, |10>, |11>)
const CNOT_RE: [f64; 16] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
];
const CNOT_IM: [f64; 16] = [0.0; 16];

/// X (NOT) gate
#[allow(dead_code)]
const X_RE: [f64; 4] = [0.0, 1.0, 1.0, 0.0];
#[allow(dead_code)]
const X_IM: [f64; 4] = [0.0; 4];

// -------------------------------------------------------------------
// Circuit generators
// -------------------------------------------------------------------

/// Bell state: H(0) then CNOT(0,1) on |00>
/// Creates (|00> + |11>) / sqrt(2) -- maximally entangled 2-qubit state.
fn generate_bell_state() -> QuantumCircuit {
    let n = 2;
    let dim = 1 << n;
    let (mut u_re, mut u_im) = identity(n);

    // H on qubit 0
    apply_single_qubit_gate(&mut u_re, &mut u_im, &H_RE, &H_IM, n, 0);
    // CNOT(0, 1)
    apply_two_qubit_gate(&mut u_re, &mut u_im, &CNOT_RE, &CNOT_IM, n, 0, 1);

    let tpm = unitary_to_tpm(&u_re, &u_im, dim);

    QuantumCircuit {
        name: "Bell State".to_string(),
        n_qubits: n,
        description: "|00> + |11> / sqrt(2) -- maximally entangled pair".to_string(),
        tpm,
        unitary_re: u_re,
        unitary_im: u_im,
    }
}

/// GHZ state: H(0) then CNOT(0,1) then CNOT(0,2) on |000>
/// Creates (|000> + |111>) / sqrt(2) -- genuine 3-party entanglement.
fn generate_ghz_state() -> QuantumCircuit {
    let n = 3;
    let dim = 1 << n;
    let (mut u_re, mut u_im) = identity(n);

    apply_single_qubit_gate(&mut u_re, &mut u_im, &H_RE, &H_IM, n, 0);
    apply_two_qubit_gate(&mut u_re, &mut u_im, &CNOT_RE, &CNOT_IM, n, 0, 1);
    apply_two_qubit_gate(&mut u_re, &mut u_im, &CNOT_RE, &CNOT_IM, n, 0, 2);

    let tpm = unitary_to_tpm(&u_re, &u_im, dim);

    QuantumCircuit {
        name: "GHZ State".to_string(),
        n_qubits: n,
        description: "|000> + |111> / sqrt(2) -- genuinely multipartite entangled".to_string(),
        tpm,
        unitary_re: u_re,
        unitary_im: u_im,
    }
}

/// Product state: Identity on |000> -- no entanglement at all.
fn generate_product_state() -> QuantumCircuit {
    let n = 3;
    let dim = 1 << n;
    let (u_re, u_im) = identity(n);
    let tpm = unitary_to_tpm(&u_re, &u_im, dim);

    QuantumCircuit {
        name: "Product State".to_string(),
        n_qubits: n,
        description: "|0> x |0> x |0> -- completely separable, no entanglement".to_string(),
        tpm,
        unitary_re: u_re,
        unitary_im: u_im,
    }
}

/// W state: (|001> + |010> + |100>) / sqrt(3)
/// Constructed via specific rotation sequence.
fn generate_w_state() -> QuantumCircuit {
    let n = 3;
    let dim = 1 << n;

    // Build the W-state preparation circuit:
    // 1. Ry(arccos(1/sqrt(3))) on qubit 0
    // 2. Controlled-Ry(arccos(1/sqrt(2))) on qubit 1 conditioned on qubit 0 = |1>
    // 3. CNOT(1, 2)
    // 4. X on qubit 0 (flip to get the right phase)
    //
    // Simpler approach: directly construct the unitary that maps |000> to W state
    // and extend to a full unitary.

    // Direct construction: we define U such that U|000> = W state.
    // Build a unitary whose first column is W = [0, 1/sqrt(3), 1/sqrt(3), 0, 1/sqrt(3), 0, 0, 0]
    // Use Gram-Schmidt to complete it.
    let s3 = 1.0 / 3.0f64.sqrt();
    let mut u_re = vec![0.0; dim * dim];
    let u_im = vec![0.0; dim * dim];

    // First column: W state
    // |001>=1, |010>=2, |100>=4
    u_re[1 * dim + 0] = s3;
    u_re[2 * dim + 0] = s3;
    u_re[4 * dim + 0] = s3;

    // Complete the unitary with Gram-Schmidt
    gram_schmidt_complete(&mut u_re, dim);

    let tpm = unitary_to_tpm(&u_re, &u_im, dim);

    QuantumCircuit {
        name: "W State".to_string(),
        n_qubits: n,
        description: "|001> + |010> + |100> / sqrt(3) -- bipartite entanglement".to_string(),
        tpm,
        unitary_re: u_re,
        unitary_im: u_im,
    }
}

/// Complete a partially filled unitary matrix using Gram-Schmidt.
/// Assumes column 0 is already set; fills columns 1..dim.
fn gram_schmidt_complete(u: &mut [f64], dim: usize) {
    // Start with the standard basis vectors and orthogonalize against existing columns
    let mut cols_done = 1; // column 0 is already set

    for candidate_col in 0..dim {
        if cols_done >= dim {
            break;
        }

        // Start with e_{candidate_col}
        let mut v = vec![0.0; dim];
        v[candidate_col] = 1.0;

        // Subtract projections onto existing columns
        for c in 0..cols_done {
            let mut dot = 0.0;
            for r in 0..dim {
                dot += u[r * dim + c] * v[r];
            }
            for r in 0..dim {
                v[r] -= dot * u[r * dim + c];
            }
        }

        // Check if remaining vector is non-zero
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-10 {
            continue;
        }

        // Normalize and store
        for r in 0..dim {
            u[r * dim + cols_done] = v[r] / norm;
        }
        cols_done += 1;
    }
}

/// Random circuit: depth layers of random single-qubit rotations + CNOT gates.
fn generate_random_circuit(depth: usize) -> QuantumCircuit {
    let n = 3;
    let dim = 1 << n;
    let (mut u_re, mut u_im) = identity(n);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    for _layer in 0..depth {
        // Random single-qubit rotations on each qubit
        for q in 0..n {
            let theta = rng.gen::<f64>() * std::f64::consts::PI;
            let phi = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
            let (ct, st) = (theta.cos(), theta.sin());
            let (cp, sp) = (phi.cos(), phi.sin());

            // General rotation: Rz(phi) * Ry(theta)
            let gate_re = [ct, -st, st * cp, ct * cp];
            let gate_im = [0.0, 0.0, st * sp, ct * sp];
            // Correct: U = [[cos(t), -sin(t)], [sin(t)*e^(i*p), cos(t)*e^(i*p)]]
            // But we just need a unitary rotation, exact form is less important
            // for generating random entanglement patterns.
            apply_single_qubit_gate(&mut u_re, &mut u_im, &gate_re, &gate_im, n, q);
        }

        // CNOT between adjacent qubits
        let q0 = rng.gen_range(0..n);
        let q1 = (q0 + 1) % n;
        apply_two_qubit_gate(&mut u_re, &mut u_im, &CNOT_RE, &CNOT_IM, n, q0, q1);
    }

    let tpm = unitary_to_tpm(&u_re, &u_im, dim);

    QuantumCircuit {
        name: "Random Circuit".to_string(),
        n_qubits: n,
        description: format!("3 qubits, depth {} -- random rotations + CNOT", depth),
        tpm,
        unitary_re: u_re,
        unitary_im: u_im,
    }
}
