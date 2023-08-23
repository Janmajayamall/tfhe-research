use crate::{
    ggsw::{cmux, encrypt_ggsw_plaintext, GgswCiphertext, GgswParams, GgswPlaintext},
    glwe::{
        trivial_encrypt_glwe_plaintext, GlweCiphertext, GlweParams, GlwePlaintext, GlweSecretKey,
    },
    lwe::{LweCiphertext, LweParams, LweSecretKey},
    utils::{poly_mul_monomial, switch_modulus},
    TfheParams,
};
use itertools::Itertools;
use ndarray::{concatenate, s, Array1, Axis};
use rand::{CryptoRng, RngCore};
use std::ops::{Add, AddAssign};

struct BootstrappingKey {
    lwe_sk_ggsw_enc: Vec<GgswCiphertext>,
}

fn bootstrapping_key_gen<R: CryptoRng + RngCore>(
    lwe_secret_key: &LweSecretKey,
    glwe_secret_key: &GlweSecretKey,
    ggsw_params: &GgswParams,
    rng: &mut R,
) -> BootstrappingKey {
    // encrypt each bit of lwe secret key
    let lwe_sk_ggsw_enc = lwe_secret_key
        .data
        .iter()
        .map(|si| {
            encrypt_ggsw_plaintext(&GgswPlaintext::new(*si), glwe_secret_key, &ggsw_params, rng)
        })
        .collect_vec();

    BootstrappingKey { lwe_sk_ggsw_enc }
}

fn monomial_index_and_term(v: usize, degree: usize) -> (u32, usize) {
    if v >= degree {
        ((u32::MAX - 1) as u32, v as usize % degree)
    } else {
        (1, v as usize)
    }
}

fn bootstrap(
    tfhe_params: &TfheParams,
    lwe_ciphertext: &LweCiphertext,
    lwe_secret_key: &LweSecretKey,
    bootstrapping_key: &BootstrappingKey,
    test_vector_poly: Array1<u32>,
) -> LweCiphertext {
    // switch modulus of LWE from q to 2N
    let approximate_lwe = switch_modulus(
        lwe_ciphertext.data.as_slice().unwrap(),
        tfhe_params.log_q,
        tfhe_params.log_degree + 1,
    );

    let mod2n = (1 << (tfhe_params.log_degree + 1)) as u32;

    // X^-b
    let neg_b = mod2n - approximate_lwe[tfhe_params.n];
    let (b_term, b_index) = monomial_index_and_term(neg_b as usize, (1 << tfhe_params.log_degree));
    // let mut x_b = vec![0; tfhe_params];
    // x_b[neg_b as usize] = 1;
    dbg!((b_term, b_index));

    let glwe_params = tfhe_params.glwe_params();

    // X^-b * v[x]
    let acc = poly_mul_monomial(test_vector_poly.view(), b_term, b_index);
    let mut acc = trivial_encrypt_glwe_plaintext(&glwe_params, &GlwePlaintext { data: acc });

    let ggsw_params = tfhe_params.ggsw_params();
    let mut tmp = neg_b;
    for i in 0..tfhe_params.n {
        // acc * X^{a_i}
        // X^{a_i} is scalar by which we must multiply acc GLWE ciphertext. GLWE ciphertext scalar multiplication requires to multiply each polynomial in ciphertext
        // with the scalar polynomial
        let mut c1_data = None;

        tmp = (tmp + ((approximate_lwe[i] * lwe_secret_key.data[i]) % mod2n)) % mod2n;

        let (a_term, a_index) =
            monomial_index_and_term(approximate_lwe[i] as usize, 1 << tfhe_params.log_degree);
        acc.data
            .outer_iter()
            .map(|p| {
                // dbg!((a_term, a_index));
                let mut res = poly_mul_monomial(p.view(), a_term, a_index);
                let res = res.insert_axis(Axis(0));
                if c1_data.is_none() {
                    c1_data = Some(res);
                } else {
                    c1_data = Some(
                        concatenate(Axis(0), &vec![c1_data.as_ref().unwrap().view(), res.view()])
                            .unwrap(),
                    );
                }
            })
            .collect_vec();
        let mut c1 = GlweCiphertext {
            data: c1_data.unwrap(),
        };

        acc = cmux(
            &ggsw_params,
            &bootstrapping_key.lwe_sk_ggsw_enc[i],
            &acc,
            &mut c1,
        );
    }

    dbg!(tmp);

    // extract the constant term in acc GLWE ciphertext
    let bootstrapped_lwe = sample_extract(&acc, &glwe_params, 0);
    bootstrapped_lwe
}

fn sample_extract(
    glwe_ciphertext: &GlweCiphertext,
    glwe_params: &GlweParams,
    sample_index: usize,
) -> LweCiphertext {
    assert!(sample_index < glwe_params.degree());
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
            for i in (sample_index + 1..glwe_params.degree()).rev() {
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
        lwe::{self, decrypt_lwe, encrypt_lwe_plaintext, LweCleartext, LweParams, LweSecretKey},
        test_vector::construct_test_vector,
    };
    use rand::{distributions::Uniform, thread_rng, Rng};

    #[test]
    fn sample_extract_glwe_works() {
        let mut rng = thread_rng();
        let tfhe_params = TfheParams::default();

        let glwe_params = tfhe_params.glwe_params();
        let glwe_sk = GlweSecretKey::random(&glwe_params, &mut rng);
        let message = vec![1, 2, 3, 4, 5, 6];
        let glwe_plaintext = GlweCleartext::encode_message(&message, &glwe_params);
        let glwe_ciphertext =
            encrypt_glwe_plaintext(&glwe_params, &glwe_plaintext, &glwe_sk, &mut rng);

        let sample_index = 0;
        let lwe_params = tfhe_params.lwe_params_post_pbs();
        let lwe_ciphertext = sample_extract(&glwe_ciphertext, &glwe_params, sample_index);
        let lwe_secret_key = LweSecretKey::from(&glwe_sk);
        let extracted_sample_pt = decrypt_lwe(&lwe_params, &lwe_secret_key, &lwe_ciphertext);
        let extracted_sample = extracted_sample_pt.decode(&lwe_params);

        dbg!(message[sample_index], extracted_sample.message);
    }

    #[test]
    fn bootstrapping_without_key_switch_works() {
        let mut rng = thread_rng();

        let tfhe_params = TfheParams::default();

        let lwe_params = tfhe_params.lwe_params();
        let lwe_secret_key = LweSecretKey::random(&lwe_params, &mut rng);

        let glwe_params = tfhe_params.glwe_params();
        let glwe_secret_key = GlweSecretKey::random(&glwe_params, &mut rng);

        let ggsw_params = tfhe_params.ggsw_params();
        let bootstrapping_key =
            bootstrapping_key_gen(&lwe_secret_key, &glwe_secret_key, &ggsw_params, &mut rng);

        // encrypt LWE
        let lwe_plaintext = LweCleartext::encode_message(1, &lwe_params);
        let lwe_ciphertext =
            encrypt_lwe_plaintext(&lwe_params, &lwe_secret_key, &lwe_plaintext, &mut rng);

        let test_vector_poly = construct_test_vector(&tfhe_params);

        let bootstrapped_lwe_ct = bootstrap(
            &tfhe_params,
            &lwe_ciphertext,
            &lwe_secret_key,
            &bootstrapping_key,
            test_vector_poly,
        );

        // dbg!(&bootstrapped_lwe_ct);

        // LWE post PBS (without key switch)
        let lwe_params_post_pbs = tfhe_params.lwe_params_post_pbs();
        let lwe_secret_post_pbs = LweSecretKey::from(&glwe_secret_key);
        let lwe_plaintext_post_pbs = decrypt_lwe(
            &lwe_params_post_pbs,
            &lwe_secret_post_pbs,
            &bootstrapped_lwe_ct,
        );
        // dbg!(&lwe_plaintext_post_pbs);
        let lwe_cleartext_post_pbs = lwe_plaintext_post_pbs.decode(&lwe_params_post_pbs);
        dbg!(lwe_cleartext_post_pbs);
    }
}
