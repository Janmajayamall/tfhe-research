use std::ops::{Add, AddAssign};

use decomposer::DecomposerParams;
use ndarray::{concatenate, s, Array1, Array2, Axis};
use ops::dot_product;
use rand::{thread_rng, CryptoRng, RngCore};
use utils::{poly_dot_product, sample_binary_array, sample_gaussian, sample_uniform_array};

mod bootstrapping;
mod decomposer;
mod ggsw;
mod glwe;
mod lwe;
mod ops;
mod test_vector;
mod utils;

struct TfheParams {
    k: usize,
    N: usize,
    n: usize,
    log_p: usize,
    log_q: usize,
    decomposer: DecomposerParams,
}

impl Default for TfheParams {
    fn default() -> Self {
        TfheParams {
            k: 2,
            N: 512,
            n: 512,
            log_p: 8,
            log_q: 32,
            decomposer: DecomposerParams {
                log_base: 4,
                levels: 8,
                log_q: 32,
            },
        }
    }
}
