use crate::{
    decomposer::SignedDecomposer,
    utils::{
        poly_dot_product, sample_binary_array, sample_gaussian, sample_gaussian_array,
        sample_uniform_array,
    },
};
use itertools::{izip, Itertools};
use ndarray::{concatenate, s, Array1, Array2, ArrayView1, Axis};
use rand::{thread_rng, CryptoRng, RngCore};
use std::ops::{Add, AddAssign};

/// Decomposes the polynomial into level polynomials by decomposing each coefficient into levels.
/// Returns a 2d array for levels x degree u32 values with each row contaning one polynomial.
fn decompose_poly(poly: &ArrayView1<u32>, signed_decomposer: &SignedDecomposer) -> Array2<u32> {
    let mut decomposed_poly_matrix =
        Array2::zeros((signed_decomposer.params.levels, poly.shape()[0]));

    izip!(decomposed_poly_matrix.axis_iter_mut(Axis(1)), poly.iter()).for_each(
        |(mut term_col, term)| {
            let decomposed = signed_decomposer.decompose(*term);
            // can't copy as slice since `decomposed_poly_matrix` is store in row major
            // form and term_col is not stored in contiguous memory (`as_slice` will panic)
            izip!(term_col.iter_mut(), decomposed.iter()).for_each(|(v0, v1)| {
                *v0 = *v1;
            });
        },
    );

    decomposed_poly_matrix
}

/// Recall that GLWE ciphertext is stored as a 2d array with k+1 rows, where each row is a polynomial.
/// After decomposition we transform the ciphertext into (k+1)*l rows where each set of l rows correspond
/// to decomposed polynomials of a single polynmoial.
pub fn decompose_glwe_ciphertext(
    glwe_ciphertext: &GlweCiphertext,
    signed_decomposer: &SignedDecomposer,
) -> Array2<u32> {
    // We handle first row separately to avoid unecessary copies
    let mut decomposed_polys =
        decompose_poly(&glwe_ciphertext.data.row(0).view(), signed_decomposer);
    glwe_ciphertext.data.outer_iter().skip(1).for_each(|poly| {
        decomposed_polys = concatenate(
            Axis(0),
            &vec![
                decomposed_polys.view(),
                decompose_poly(&poly.view(), signed_decomposer).view(),
            ],
        )
        .unwrap();
    });
    decomposed_polys
}

#[allow(non_snake_case)]
#[derive(Debug, Clone)]
pub struct GlweParams {
    pub(crate) k: usize,
    pub(crate) N: usize,
    /// q in Z/2^qZ
    pub(crate) log_q: usize,
    /// p in Z/2^pZ
    pub(crate) log_p: usize,
    pub(crate) mean: f64,
    pub(crate) std_dev: f64,
}

impl Default for GlweParams {
    fn default() -> Self {
        GlweParams {
            k: 2,
            N: 8,
            /// q in Z/2^qZ
            log_q: 32,
            /// p in Z/2^pZ
            log_p: 8,
            mean: 0.0,
            std_dev: 0.1231231,
        }
    }
}

#[derive(Debug)]
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
    pub(crate) data: Array2<u32>,
}

pub fn encrypt_glwe_zero<R: CryptoRng + RngCore>(
    glwe_params: &GlweParams,
    sk: &GlweSecretKey,
    rng: &mut R,
) -> GlweCiphertext {
    let a_samples = sample_uniform_array(rng, (glwe_params.k, glwe_params.N));

    let mut a_s = poly_dot_product(&a_samples.view(), &sk.data.view());

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
    let mut a_s = poly_dot_product(&a_samples, &sk.data.view());

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
    use crate::decomposer::DecomposerParams;

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

    #[test]
    fn decompose_glwe_works() {
        let mut rng = thread_rng();
        let glwe_params = GlweParams::default();
        let glwe_sk = GlweSecretKey::random(&glwe_params, &mut rng);
        let glwe_ciphertext = encrypt_glwe_zero(&glwe_params, &glwe_sk, &mut rng);
        let signed_decomposer =
            SignedDecomposer::new(DecomposerParams::new(4, 8, glwe_params.log_q));
        let decomposed_glwe_ciphertext =
            decompose_glwe_ciphertext(&glwe_ciphertext, &signed_decomposer);
        dbg!(decomposed_glwe_ciphertext);
    }
}
