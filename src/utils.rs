use rand::{CryptoRng, RngCore};

pub fn sample_gaussian<R: CryptoRng + RngCore>(mean: f64, std_dev: f64, rng: &mut R) -> u32 {
    return 0;
}

pub fn sample_uniform_vec<R: CryptoRng + RngCore>(rng: &mut R) -> Vec<u32> {
    todo!()
}
