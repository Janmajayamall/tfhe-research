use ndarray::{Array1, Axis, Array2};
use ops::dot_product;
use rand::{CryptoRng, RngCore};
use utils::{sample_gaussian, sample_uniform_vec};

mod ops;
mod utils;
struct LweMessage {}

#[derive(Debug, Clone)]
struct LweParams {
    n: usize,
    /// q in Z/2^qZ
    log_q: usize,
    /// p in Z/2^pZ
    log_p: usize,
    mean: f64,
    std_dev: f64,
}

/// Binary secret key
pub struct LweSecretKey {
    data: Array1<u32>,
    params: LweParams,
}

/// LweCiphertext
/// data is an array of (a_0, a_1, ..., a_{n-1}, b)
pub struct LweCiphertext {
    data: Array1<u32>,
    params: LweParams,
}

pub fn encrypt_lwe_zero<R: CryptoRng + RngCore>(sk: &LweSecretKey, rng: &mut R) -> LweCiphertext {
    let error = sample_gaussian(sk.params.mean, sk.params.std_dev, rng);
    let mut a_samples = Array1::from_vec(sample_uniform_vec(rng));

    let mut a_s = sk.data.dot(&a_samples);
    a_s += error;

    let mut data = a_samples.to_vec();
    data.push(a_s);

    LweCiphertext {
        data: Array1::from_vec(data),
        params: sk.params.clone(),
    }
}

#[allow(non_snake_case)]
struct GlweParams {
    k: usize,
    N: usize,
    /// q in Z/2^qZ
    log_q: usize,
    /// p in Z/2^pZ
    log_p: usize,
    mean: f64,
    std_dev: f64,
}

/// RLWE
struct GlweCiphertext {
    data: Array2<u32>,
}

///
struct GsweCipertext {}

// Implement GLWE
// Implement GSWE
