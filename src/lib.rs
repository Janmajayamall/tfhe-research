use std::ops::{Add, AddAssign};

use decomposer::DecomposerParams;
use ggsw::GgswParams;
use glwe::GlweParams;
use lwe::LweParams;
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

pub struct TfheParams {
    k: usize,
    log_degree: usize,
    n: usize,
    padding_bits: usize,
    log_p: usize,
    log_q: usize,
    decomposer: DecomposerParams,
    mean: f64,
    std_dev: f64,
}

impl TfheParams {
    pub fn glwe_params(&self) -> GlweParams {
        GlweParams {
            k: self.k,
            log_degree: self.log_degree,
            log_q: self.log_q,
            log_p: self.log_p,
            padding_bits: self.padding_bits,
            mean: self.mean,
            std_dev: self.std_dev,
        }
    }

    pub fn lwe_params(&self) -> LweParams {
        LweParams {
            n: self.n,
            log_q: self.log_q,
            log_p: self.log_p,
            padding_bits: self.padding_bits,
            mean: self.mean,
            std_dev: self.std_dev,
        }
    }

    pub fn lwe_params_post_pbs(&self) -> LweParams {
        LweParams {
            n: (1 << self.log_degree) * self.k,
            log_q: self.log_q,
            log_p: self.log_p,
            padding_bits: self.padding_bits,
            mean: self.mean,
            std_dev: self.std_dev,
        }
    }

    pub fn ggsw_params(&self) -> GgswParams {
        GgswParams {
            glwe_params: self.glwe_params(),
            decomposer_params: self.decomposer.clone(),
        }
    }
}

impl Default for TfheParams {
    fn default() -> Self {
        TfheParams {
            k: 1,
            log_degree: 9,
            n: 2,
            log_p: 2,
            log_q: 32,
            decomposer: DecomposerParams {
                log_base: 4,
                levels: 8,
                log_q: 32,
            },
            mean: 0.0,
            std_dev: 0.0,
            padding_bits: 1,
        }
    }
}
