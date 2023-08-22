use crate::utils::{poly_dot_product, sample_binary_array, sample_gaussian, sample_uniform_array};
use ndarray::{concatenate, s, Array1, Array2, Axis};
use rand::{thread_rng, CryptoRng, RngCore};
use std::ops::{Add, AddAssign};

#[derive(Debug, Clone)]
pub struct LweParams {
    n: usize,
    /// q in Z/2^qZ
    log_q: usize,
    /// p in Z/2^pZ
    log_p: usize,
    mean: f64,
    std_dev: f64,
}

impl Default for LweParams {
    fn default() -> Self {
        LweParams {
            n: 512,
            log_q: 32,
            log_p: 8,
            mean: 0.0,
            std_dev: 0.121312,
        }
    }
}

#[derive(Debug, Clone)]
/// Binary secret key
pub struct LweSecretKey {
    data: Array1<u32>,
    params: LweParams,
}

impl LweSecretKey {
    pub fn random<R: CryptoRng + RngCore>(lwe_params: &LweParams, rng: &mut R) -> LweSecretKey {
        LweSecretKey {
            data: sample_binary_array(rng, (lwe_params.n)),
            params: lwe_params.clone(),
        }
    }
}

/// Contains message in clear text (ie without encoding)
#[derive(Debug, Clone)]
pub struct LweCleartext {
    message: u32,
}

impl LweCleartext {
    /// We limit q and p to powers of 2
    pub fn encode_message(m: u32, lwe_params: &LweParams) -> LwePlaintext {
        assert!(m < 1u32 << lwe_params.log_p);
        LwePlaintext {
            data: m << (lwe_params.log_q - lwe_params.log_p),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LwePlaintext {
    data: u32,
}

/// Contains plaintext for LWE (ie encoded LweClearText)
impl LwePlaintext {
    pub fn decode(&self, lwe_params: &LweParams) -> LweCleartext {
        LweCleartext {
            message: self.data >> (lwe_params.log_q - lwe_params.log_p),
        }
    }
}

#[derive(Debug, Clone)]
/// LweCiphertext
/// data is an array of (a_0, a_1, ..., a_{n-1}, b)
pub struct LweCiphertext {
    data: Array1<u32>,
    params: LweParams,
}

pub fn encrypt_lwe_zero<R: CryptoRng + RngCore>(sk: &LweSecretKey, rng: &mut R) -> LweCiphertext {
    let error = sample_gaussian(sk.params.mean, sk.params.std_dev, rng);
    let mut a_samples: Array1<u32> = sample_uniform_array(rng, (sk.params.n));

    let mut a_s = sk.data.dot(&a_samples);
    a_s += error;

    let mut data = a_samples.to_vec();
    data.push(a_s);

    LweCiphertext {
        data: Array1::from_vec(data),
        params: sk.params.clone(),
    }
}

pub fn encrypt_lwe_plaintext<R: CryptoRng + RngCore>(
    sk: &LweSecretKey,
    lwe_plaintext: &LwePlaintext,
    rng: &mut R,
) -> LweCiphertext {
    let error = sample_gaussian(sk.params.mean, sk.params.std_dev, rng);
    let mut a_samples: Array1<u32> = sample_uniform_array(rng, (sk.params.n));

    let mut a_s = sk.data.dot(&a_samples);
    a_s = a_s.wrapping_add(error);
    a_s = a_s.wrapping_add(lwe_plaintext.data);

    let mut data = a_samples.to_vec();
    data.push(a_s);

    LweCiphertext {
        data: Array1::from_vec(data),
        params: sk.params.clone(),
    }
}

pub fn decrypt_lwe(sk: &LweSecretKey, ct: &LweCiphertext) -> LwePlaintext {
    let lwe_params = &sk.params;
    let a_samples = ct.data.slice(s![..-1]);

    // a*s
    let a_s = sk.data.dot(&a_samples);
    // last eleement in ciphertext data is b
    // b - a*s
    let mut plaintext = ct.data.as_slice().unwrap()[lwe_params.n];
    plaintext = plaintext.wrapping_sub(a_s);

    LwePlaintext { data: plaintext }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn encrypt_and_decrypt_lwe() {
        let mut rng = thread_rng();
        let lwe_params = LweParams::default();
        let message = 4;
        let lwe_plaintext = LweCleartext::encode_message(message, &lwe_params);
        let lwe_sk = LweSecretKey::random(&lwe_params, &mut rng);
        let lwe_ciphertext = encrypt_lwe_plaintext(&lwe_sk, &lwe_plaintext, &mut rng);
        let lwe_plaintext_back = decrypt_lwe(&lwe_sk, &lwe_ciphertext);
        let lwe_cleartext_back = lwe_plaintext_back.decode(&lwe_params);
        assert_eq!(message, lwe_cleartext_back.message);
    }
}
