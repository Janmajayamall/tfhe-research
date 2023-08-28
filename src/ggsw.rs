use ndarray::{concatenate, s, Array, Array3, Axis};
use rand::{CryptoRng, RngCore};

use crate::{
    decomposer::{DecomposerParams, SignedDecomposer},
    glwe::{
        decompose_glwe_ciphertext, encrypt_glwe_plaintext, encrypt_glwe_zero, GlweCiphertext,
        GlweCleartext, GlweParams, GlweSecretKey,
    },
    utils::poly_dot_product,
    TfheParams,
};

/// Although message must be a polynomial in Z_{N}[X] we only require
/// Ggsw ciphertexts to encrypt bits of LWE secret key for PBS.
pub struct GgswPlaintext {
    message: u32,
}

impl GgswPlaintext {
    pub fn new(message: u32) -> GgswPlaintext {
        GgswPlaintext { message }
    }
}

pub struct GgswParams {
    pub(crate) glwe_params: GlweParams,
    pub(crate) decomposer_params: DecomposerParams,
}

impl Default for GgswParams {
    fn default() -> Self {
        TfheParams::default().ggsw_params()
    }
}

pub struct GgswCiphertext {
    /// Stored as 3D array of u32s. First 2 dimension must be viewed as a matrix with polynomial elements. Each row of the matrix contains a single GLWE ciphertext.
    /// You can access polynomial at i^th col and j^th row as data[j][i]
    data: Array3<u32>,
}

/**
To encrypt message m we must calculate
[                            [
    GLWE(0),                    B^{l-1}, 0, 0, ..., 0
                                B^{l-2}
                                .
                                1
                +        m      .
                                .
                                    .
                                        .
                                            .
                                                B^{l-1}
                                                B^{l-2}
                                                .
    GLWE(0)                                     1

]                           ]
Both m and B^{i} are constants.

Expanding into  a single row. We must add glwe encryption of zero
with B^{l-i}
[
    a_0, a_1, a_2, ...,      b
] +
[
    0, 0, ..., B^{l-x}, ..., 0
]
This means we need to add cosntant polynomial B^{l-x} to only one of
the polynomials from {a_0, a_1, a_2, ..., b}. And the one to add to depends on
the row we at. For example, for 1st l rows must add to a_0, then for second l
rows we must add to a_1, and the pattern continues...To be exact, i = row / l.
 */
pub fn encrypt_ggsw_plaintext<R: CryptoRng + RngCore>(
    plaintext: &GgswPlaintext,
    sk: &GlweSecretKey,
    ggsw_params: &GgswParams,
    rng: &mut R,
) -> GgswCiphertext {
    let mut ggsw_matrix = None;
    for i in 0..(ggsw_params.glwe_params.glwe_dimension + 1) {
        // which of the polynomial in zero encryption to add constant value `m*decomposition_factor`
        let add_index = i;

        // levels indicate how many most significant legs of the decomposition by base should be includes. For
        // ex, log_base 4 and levels 6 means we require precision of only 24 most significant bits. This translates to
        // constructing gadget matrix with value {B^{l-1}, ..., B^{l-levels}} where l = log_q / log_base
        let log_q_by_log_base =
            ggsw_params.glwe_params.log_q / ggsw_params.decomposer_params.log_base;
        for level_index in 0..ggsw_params.decomposer_params.levels {
            let mut zero_encryption = encrypt_glwe_zero(&ggsw_params.glwe_params, sk, rng);

            // only add when message is not zero
            if plaintext.message != 0 {
                // m * \beta^{l-(level_index+1)}
                let decomposition_factor = plaintext.message
                    * (1 << (ggsw_params.decomposer_params.log_base
                        * (log_q_by_log_base - (level_index + 1))));
                let value = zero_encryption.data.get_mut((add_index, 0)).unwrap();
                *value = value.wrapping_add(decomposition_factor);
            }

            // add another axis to convert 2d array to 3d array, so that each GLWE ciphertext
            // becomes a row at Axis(0) with polynomial elements at Axis(1). We can access
            // a i^th polynomial of j^th row as array[j][i].
            // zero_encryption.data.row
            let tmp = zero_encryption.clone();
            let data = zero_encryption.data.insert_axis(Axis(0));

            {
                let s = data.slice(s![0, 1, ..]);
                assert_eq!(s, tmp.data.row(1));
            }

            if ggsw_matrix.is_none() {
                ggsw_matrix = Some(data);
            } else {
                ggsw_matrix = Some(
                    concatenate(Axis(0), &vec![ggsw_matrix.unwrap().view(), data.view()]).unwrap(),
                );
            }
        }
    }

    GgswCiphertext {
        data: ggsw_matrix.unwrap(),
    }
}

fn external_product(
    ggsw_params: &GgswParams,
    ggsw_ciphertext: &GgswCiphertext,
    glwe_ciphertext: &GlweCiphertext,
) -> GlweCiphertext {
    let signed_decomposer = SignedDecomposer::new(ggsw_params.decomposer_params.clone());
    let decomposed_glwe_ciphertext = decompose_glwe_ciphertext(glwe_ciphertext, &signed_decomposer);

    // In ggsw each GLWE ciphertext corresponds to a row
    let k = ggsw_params.glwe_params.glwe_dimension;
    let mut glwe_ciphertext = None;

    for ggsw_col in 0..k + 1 {
        let col = ggsw_ciphertext.data.slice(s![.., ggsw_col, ..]);
        let res = poly_dot_product(&decomposed_glwe_ciphertext.view(), &col.view());
        let res = res.insert_axis(Axis(0));

        if glwe_ciphertext.is_none() {
            glwe_ciphertext = Some(res);
        } else {
            glwe_ciphertext = Some(
                concatenate(Axis(0), &vec![glwe_ciphertext.unwrap().view(), res.view()]).unwrap(),
            );
        }
    }

    GlweCiphertext {
        data: glwe_ciphertext.unwrap(),
    }
}

/// CMUX gate that either returns `glwe_ciphertext0` or `glwe_ciphertext1` depending on whether `ggsw_ciphertext` is encryption of 0 or 1, respecitvely.
pub fn cmux(
    ggsw_params: &GgswParams,
    ggsw_ciphertext: &GgswCiphertext,
    glwe_ciphertext0: &GlweCiphertext,
    glwe_ciphertext1: &mut GlweCiphertext,
) -> GlweCiphertext {
    // c1 - c0
    *glwe_ciphertext1 -= glwe_ciphertext0;

    // b(c1-c0) + c0
    let mut res = external_product(ggsw_params, ggsw_ciphertext, &glwe_ciphertext1);
    res += glwe_ciphertext0;

    res
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::glwe::decrypt_glwe_ciphertext;

    use super::*;

    #[test]
    fn ggsw_encryption_works() {
        let mut rng = thread_rng();
        let ggsw_params = GgswParams::default();
        let glwe_sk = GlweSecretKey::random(&ggsw_params.glwe_params, &mut rng);
        let ggsw_ciphertext = encrypt_ggsw_plaintext(
            &GgswPlaintext { message: 3 },
            &glwe_sk,
            &ggsw_params,
            &mut rng,
        );
        // dbg!(ggsw_ciphertext.data.slice(s![0..40, 0, ..]).shape());
    }

    #[test]
    fn external_product_works() {
        let mut rng = thread_rng();

        // GGSW
        let ggsw_params = GgswParams::default();
        let glwe_sk = GlweSecretKey::random(&ggsw_params.glwe_params, &mut rng);
        let ggsw_ciphertext = encrypt_ggsw_plaintext(
            &GgswPlaintext { message: 2 },
            &glwe_sk,
            &ggsw_params,
            &mut rng,
        );

        // GLWE
        let message = vec![3; ggsw_params.glwe_params.degree()];
        let glwe_plaintext = GlweCleartext::encode_message(&message, &ggsw_params.glwe_params);
        let glwe_ciphertext = encrypt_glwe_plaintext(
            &ggsw_params.glwe_params,
            &glwe_plaintext,
            &glwe_sk,
            &mut rng,
        );

        let res_ciphertext = external_product(&ggsw_params, &ggsw_ciphertext, &glwe_ciphertext);

        let res_plaintext = decrypt_glwe_ciphertext(
            &ggsw_params.glwe_params,
            &glwe_sk,
            &res_ciphertext,
            &mut rng,
        );
        let res_message = res_plaintext.decode(&ggsw_params.glwe_params);
        dbg!(res_message);
    }

    #[test]
    fn cmux_works() {
        let mut rng = thread_rng();

        // GGSW
        let ggsw_params = GgswParams::default();
        let glwe_sk = GlweSecretKey::random(&ggsw_params.glwe_params, &mut rng);
        // encrypt indicator bit
        let ggsw_ciphertext = encrypt_ggsw_plaintext(
            &GgswPlaintext { message: 1 },
            &glwe_sk,
            &ggsw_params,
            &mut rng,
        );

        // GLWE
        let message0 = vec![3; ggsw_params.glwe_params.degree()];
        let message1 = vec![2; ggsw_params.glwe_params.degree()];
        let glwe_ciphertext0 = encrypt_glwe_plaintext(
            &ggsw_params.glwe_params,
            &GlweCleartext::encode_message(&message0, &ggsw_params.glwe_params),
            &glwe_sk,
            &mut rng,
        );
        let mut glwe_ciphertext1 = encrypt_glwe_plaintext(
            &ggsw_params.glwe_params,
            &GlweCleartext::encode_message(&message1, &ggsw_params.glwe_params),
            &glwe_sk,
            &mut rng,
        );

        let res = cmux(
            &ggsw_params,
            &ggsw_ciphertext,
            &glwe_ciphertext0,
            &mut glwe_ciphertext1,
        );

        let res_plaintext =
            decrypt_glwe_ciphertext(&ggsw_params.glwe_params, &glwe_sk, &res, &mut rng);
        let res_message = res_plaintext.decode(&ggsw_params.glwe_params);
        dbg!(res_message);
    }
}
