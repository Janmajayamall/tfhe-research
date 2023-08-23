use std::ops::{Add, AddAssign};

use ndarray::{s, Array1};

use crate::{
    glwe::{GlweCiphertext, GlweParams},
    lwe::LweCiphertext,
};

fn sample_extract(
    glwe_ciphertext: &GlweCiphertext,
    glwe_params: &GlweParams,
    sample_index: usize,
) -> LweCiphertext {
    assert!(sample_index < glwe_params.N);
    let lwe_b = *glwe_ciphertext
        .data
        .get((glwe_params.k, sample_index))
        .unwrap();

    let mut lwe_as = vec![];
    // iterato through all polynomials a_0, ..., a_{k-1}
    glwe_ciphertext
        .data
        .slice(s![..-1, ..])
        .outer_iter()
        .for_each(|poly| {
            // n, n-1, ..., 0
            for i in (0..(sample_index + 1)).rev() {
                lwe_as.push(poly[i]);
            }

            // -(N-1), ... -(n+2), -(n+1)
            for i in (sample_index + 1..glwe_params.N).rev() {
                lwe_as.push(poly[i].wrapping_neg());
            }
        });

    lwe_as.push(lwe_b);

    LweCiphertext {
        data: Array1::from_vec(lwe_as),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        decomposer::{DecomposerParams, SignedDecomposer},
        glwe::{
            decompose_glwe_ciphertext, encrypt_glwe_plaintext, encrypt_glwe_zero, GlweCleartext,
            GlweSecretKey,
        },
        lwe::{self, decrypt_lwe, LweParams, LweSecretKey},
    };
    use rand::{distributions::Uniform, thread_rng, Rng};

    #[test]
    fn sample_extract_glwe_works() {
        let mut rng = thread_rng();

        let glwe_params = GlweParams::default();
        let glwe_sk = GlweSecretKey::random(&glwe_params, &mut rng);
        let message = vec![1, 2, 3, 4, 5, 6];
        let glwe_plaintext = GlweCleartext::encode_message(&message, &glwe_params);
        let glwe_ciphertext =
            encrypt_glwe_plaintext(&glwe_params, &glwe_plaintext, &glwe_sk, &mut rng);

        let sample_index = 4;
        let lwe_params = LweParams {
            n: glwe_params.N * glwe_params.k,
            log_q: glwe_params.log_q,
            log_p: glwe_params.log_p,
            mean: glwe_params.mean,
            std_dev: glwe_params.std_dev,
        };
        let lwe_ciphertext = sample_extract(&glwe_ciphertext, &glwe_params, sample_index);
        let lwe_secret_key = LweSecretKey::from(&glwe_sk);
        let extracted_sample_pt = decrypt_lwe(&lwe_params, &lwe_secret_key, &lwe_ciphertext);
        let extracted_sample = extracted_sample_pt.decode(&lwe_params);

        assert_eq!(message[sample_index], extracted_sample.message);
    }
}
