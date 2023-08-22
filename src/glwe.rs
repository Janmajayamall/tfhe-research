use crate::utils::{
    poly_mul_add_list, sample_binary_array, sample_gaussian, sample_gaussian_array,
    sample_uniform_array,
};
use itertools::{izip, Itertools};
use ndarray::{concatenate, s, Array1, Array2, Axis};
use rand::{thread_rng, CryptoRng, RngCore};
use std::ops::{Add, AddAssign};

#[allow(non_snake_case)]
#[derive(Debug, Clone)]
pub struct GlweParams {
    k: usize,
    N: usize,
    /// q in Z/2^qZ
    log_q: usize,
    /// p in Z/2^pZ
    log_p: usize,
    mean: f64,
    std_dev: f64,
}

impl Default for GlweParams {
    fn default() -> Self {
        GlweParams {
            k: 4,
            N: 512,
            /// q in Z/2^qZ
            log_q: 32,
            /// p in Z/2^pZ
            log_p: 8,
            mean: 0.0,
            std_dev: 0.1231231,
        }
    }
}

pub struct GlweCleartext {
    message: Vec<u32>,
}
impl GlweCleartext {
    pub fn encode_message(message: &[u32], glwe_params: &GlweParams) -> GlwePlaintext {
        let mut data = vec![0u32; glwe_params.N];
        izip!(data.iter_mut(), message.iter()).for_each(|(d, m)| {
            assert!(*m < 1u32 << glwe_params.log_p);
            *d = *m << (glwe_params.log_q - glwe_params.log_p);
        });

        GlwePlaintext {
            data: Array1::from_vec(data),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GlwePlaintext {
    data: Array1<u32>,
}

impl GlwePlaintext {
    pub fn decode(&self, glwe_params: &GlweParams) -> GlweCleartext {
        let message = self
            .data
            .iter()
            .map(|m| *m >> (glwe_params.log_q - glwe_params.log_p))
            .collect_vec();

        GlweCleartext { message }
    }
}

#[derive(Debug, Clone)]
pub struct GlweSecretKey {
    data: Array2<u32>,
}

impl GlweSecretKey {
    pub fn random<R: CryptoRng + RngCore>(glwe_params: &GlweParams, rng: &mut R) -> GlweSecretKey {
        GlweSecretKey {
            data: sample_binary_array(rng, (glwe_params.k, glwe_params.N)),
        }
    }
}

/// RLWE
#[derive(Debug, Clone)]
pub struct GlweCiphertext {
    data: Array2<u32>,
}

pub fn encrypt_glwe_zero<R: CryptoRng + RngCore>(
    glwe_params: &GlweParams,
    sk: &GlweSecretKey,
    rng: &mut R,
) -> GlweCiphertext {
    let a_samples = sample_uniform_array(rng, (glwe_params.k, glwe_params.N));

    let mut a_s = poly_mul_add_list(&a_samples.view(), &sk.data.view());

    // sample error
    let error = sample_gaussian_array(glwe_params.mean, glwe_params.std_dev, rng, (glwe_params.N));

    // \sum a*s + e
    a_s.add_assign(&error);
    let a_s_e = a_s.insert_axis(Axis(0));

    let ct_data = concatenate(Axis(0), &[a_samples.view(), a_s_e.view()]).unwrap();

    GlweCiphertext { data: ct_data }
}

pub fn encrypt_glwe_plaintext<R: CryptoRng + RngCore>(
    glwe_params: &GlweParams,
    glwe_plaintext: &GlwePlaintext,
    sk: &GlweSecretKey,
    rng: &mut R,
) -> GlweCiphertext {
    let mut zero_encryption = encrypt_glwe_zero(glwe_params, sk, rng);

    // add message polynomial to `b` polynomial
    izip!(
        zero_encryption.data.row_mut(glwe_params.k).iter_mut(),
        glwe_plaintext.data.iter()
    )
    .for_each(|(c, m)| *c = c.wrapping_add(*m));

    zero_encryption
}

pub fn decrypt_glwe_ciphertext<R: CryptoRng + RngCore>(
    glwe_params: &GlweParams,
    sk: &GlweSecretKey,
    ct: &GlweCiphertext,
    rng: &mut R,
) -> GlwePlaintext {
    let a_samples = ct.data.slice(s![..glwe_params.k, ..]);
    let mut a_s = poly_mul_add_list(&a_samples, &sk.data.view());

    // b - \sum a*s
    let mut plaintext = Array1::from_vec(ct.data.row(glwe_params.k).to_vec());
    izip!(
        plaintext.as_slice_mut().unwrap().iter_mut(),
        a_s.as_slice_mut().unwrap().iter()
    )
    .for_each(|(p, a)| {
        *p = p.wrapping_sub(*a);
    });

    GlwePlaintext { data: plaintext }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{distributions::Uniform, thread_rng, Rng};

    #[test]
    fn encrypt_and_decrypt_glwe() {
        let mut rng = thread_rng();
        let glwe_params = GlweParams::default();
        let message = rng
            .clone()
            .sample_iter(Uniform::new(0, 1 << glwe_params.log_p))
            .take(glwe_params.N)
            .collect_vec();
        let glwe_plaintext = GlweCleartext::encode_message(&message, &glwe_params);
        let glwe_sk = GlweSecretKey::random(&glwe_params, &mut rng);
        let glwe_ciphertext =
            encrypt_glwe_plaintext(&glwe_params, &glwe_plaintext, &glwe_sk, &mut rng);
        let glwe_plaintext_back =
            decrypt_glwe_ciphertext(&glwe_params, &glwe_sk, &glwe_ciphertext, &mut rng);

        assert_eq!(glwe_plaintext, glwe_plaintext_back);

        let glwe_cleartext_back: GlweCleartext = glwe_plaintext_back.decode(&glwe_params);
        assert_eq!(message, glwe_cleartext_back.message);
    }
}
