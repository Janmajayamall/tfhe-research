use ndarray::{s, Array};
use rand::{CryptoRng, RngCore};

use crate::{
    decomposer::DecomposerParams,
    glwe::{
        encrypt_glwe_plaintext, encrypt_glwe_zero, GlweCiphertext, GlweCleartext, GlweParams,
        GlweSecretKey,
    },
};

/// Although message must be a polynomial in Z_{N,q}[X] we only require
/// Ggsw ciphertexts to encrypt bits of LWE secret key for PBS.
struct GgswPlaintext {
    message: u32,
}

struct GgswParams {
    glwe_params: GlweParams,
    decomposer_params: DecomposerParams,
}

impl Default for GgswParams {
    fn default() -> Self {
        let glwe_params = GlweParams::default();
        let decomposer_params = DecomposerParams {
            log_base: 2,
            levels: 8,
            log_q: glwe_params.log_q,
        };
        GgswParams {
            glwe_params,
            decomposer_params,
        }
    }
}

struct GgswCiphertext {
    /// Contains (k+1)l glwe ciphertext. Vector of GLWE ciphertexts must be viewed as a matrix with each row
    ///  consisting of polynomials forming single GLWE ciphertext.
    data: Vec<GlweCiphertext>,
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
fn encrypt_ggsw_plaintext<R: CryptoRng + RngCore>(
    plaintext: &GgswPlaintext,
    sk: &GlweSecretKey,
    ggsw_params: &GgswParams,
    rng: &mut R,
) -> GgswCiphertext {
    let mut ggsw_ct = vec![];
    for i in 0..(ggsw_params.glwe_params.k + 1) {
        // which of the polynomial in zero encryption to add constant value `m*decomposition_factor`
        let add_index = i;

        for level in (0..ggsw_params.decomposer_params.levels).rev() {
            let mut zero_encryption = encrypt_glwe_zero(&ggsw_params.glwe_params, sk, rng);

            // only add when message is not zero
            if plaintext.message != 0 {
                // m * \beta^{l-x}
                let decomposition_factor =
                    plaintext.message * (1 << (ggsw_params.decomposer_params.log_base * level));

                let value = zero_encryption.data.get_mut((add_index, 0)).unwrap();
                *value = value.wrapping_add(decomposition_factor);
            }
            ggsw_ct.push(zero_encryption);
        }
    }

    GgswCiphertext { data: ggsw_ct }
}
