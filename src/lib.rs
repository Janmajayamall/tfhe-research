use std::ops::{Add, AddAssign};

use ndarray::{concatenate, s, Array1, Array2, Axis};
use ops::dot_product;
use rand::{thread_rng, CryptoRng, RngCore};
use utils::{poly_mul_add_list, sample_binary_array, sample_gaussian, sample_uniform_array};

mod ggsw;
mod glwe;
mod lwe;
mod ops;
mod utils;


// Implement GLWE
// Implement GSWE
