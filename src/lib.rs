use std::ops::{Add, AddAssign};

use decomposer::DecomposerParams;
use ggsw::GgswParams;
use glwe::GlweParams;
use lwe::LweParams;
use ndarray::{concatenate, s, Array1, Array2, Axis};
use ops::dot_product;
use rand::{thread_rng, CryptoRng, RngCore};
use utils::{poly_dot_product, sample_binary_array, sample_uniform_array};

mod boolean;
mod bootstrapping;
mod decomposer;
mod ggsw;
mod glwe;
mod key_switching;
mod lwe;
mod ops;
mod test_vector;
mod utils;

pub struct TfheParams {
    glwe_dimension: usize,
    glwe_poly_degree: usize,
    lwe_dimension: usize,
    padding_bits: usize,
    log_p: usize,
    log_q: usize,
    ks_decomposer: DecomposerParams,
    pbs_decomposer: DecomposerParams,
    lwe_std_dev: f64,
    glwe_std_dev: f64,
}

impl TfheParams {
    pub fn glwe_params(&self) -> GlweParams {
        GlweParams {
            glwe_dimension: self.glwe_dimension,
            log_degree: self.glwe_poly_degree,
            log_q: self.log_q,
            log_p: self.log_p,
            padding_bits: self.padding_bits,
            std_dev: self.glwe_std_dev,
        }
    }

    pub fn lwe_params(&self) -> LweParams {
        LweParams {
            lwe_dimension: self.lwe_dimension,
            log_q: self.log_q,
            log_p: self.log_p,
            padding_bits: self.padding_bits,
            std_dev: self.lwe_std_dev,
        }
    }

    pub fn lwe_params_post_pbs(&self) -> LweParams {
        LweParams {
            lwe_dimension: (1 << self.glwe_poly_degree) * self.glwe_dimension,
            log_q: self.log_q,
            log_p: self.log_p,
            padding_bits: self.padding_bits,
            std_dev: self.lwe_std_dev,
        }
    }

    pub fn ggsw_params(&self) -> GgswParams {
        GgswParams {
            glwe_params: self.glwe_params(),
            decomposer_params: self.pbs_decomposer.clone(),
        }
    }
}

impl Default for TfheParams {
    #[cfg(test)]
    fn default() -> Self {
        TfheParams {
            glwe_dimension: 2,
            glwe_poly_degree: 9,
            lwe_dimension: 4,
            log_p: 2,
            log_q: 32,
            ks_decomposer: DecomposerParams {
                log_base: 4,
                levels: 5,
                log_q: 32,
            },
            pbs_decomposer: DecomposerParams {
                log_base: 4,
                levels: 6,
                log_q: 32,
            },
            padding_bits: 1,
            lwe_std_dev: 0.000013071021089943935,
            glwe_std_dev: 0.00000004990272175010415,
        }
    }

    #[cfg(not(test))]
    fn default() -> Self {
        TfheParams {
            glwe_dimension: 2,
            glwe_poly_degree: 9,
            lwe_dimension: 722,
            log_p: 2,
            log_q: 32,
            ks_decomposer: DecomposerParams {
                log_base: 4,
                levels: 5,
                log_q: 32,
            },
            pbs_decomposer: DecomposerParams {
                log_base: 4,
                levels: 6,
                log_q: 32,
            },
            padding_bits: 1,
            lwe_std_dev: 0.000013071021089943935,
            glwe_std_dev: 0.00000004990272175010415,
        }
    }
}
