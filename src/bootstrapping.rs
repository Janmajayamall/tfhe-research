use crate::{
    decomposer::SignedDecomposer,
    ggsw::{cmux, encrypt_ggsw_plaintext, GgswCiphertext, GgswParams, GgswPlaintext},
    glwe::{
        decrypt_glwe_ciphertext, trivial_encrypt_glwe_plaintext, GlweCiphertext, GlweCleartext,
        GlweParams, GlwePlaintext, GlweSecretKey, Monomial,
    },
    key_switching::{key_switch_lwe, KeySwitchingKey},
    lwe::{LweCiphertext, LweParams, LweSecretKey},
    utils::{poly_mul_monomial, poly_mul_monomial_custom_mod, switch_modulus},
    TfheParams,
};
use itertools::Itertools;
use ndarray::{concatenate, s, Array1, Axis};
use rand::{thread_rng, CryptoRng, RngCore};
use std::ops::{Add, AddAssign};

pub struct BootstrappingKey {
    lwe_sk_ggsw_enc: Vec<GgswCiphertext>,
    ksk: KeySwitchingKey,
}

pub fn bootstrapping_key_gen<R: CryptoRng + RngCore>(
    tfhe_params: &TfheParams,
    lwe_secret_key: &LweSecretKey,
    glwe_secret_key: &GlweSecretKey,
    rng: &mut R,
) -> BootstrappingKey {
    let ggsw_params = tfhe_params.ggsw_params();

    // encrypt each bit of lwe secret key
    let lwe_sk_ggsw_enc = lwe_secret_key
        .data
        .iter()
        .map(|si| {
            encrypt_ggsw_plaintext(&GgswPlaintext::new(*si), glwe_secret_key, &ggsw_params, rng)
        })
        .collect_vec();

    let signed_decomposer = SignedDecomposer::new(tfhe_params.decomposer.clone());

    // Generate key switching key from GLWE to LWE
    let glwe_as_lwe_sk = LweSecretKey::from(glwe_secret_key);
    let lwe_params_post_pbs = tfhe_params.lwe_params_post_pbs();
    let lwe_params = tfhe_params.lwe_params();
    let ksk = KeySwitchingKey::generate_ksk(
        &glwe_as_lwe_sk,
        lwe_secret_key,
        &lwe_params_post_pbs,
        &lwe_params,
        &signed_decomposer,
        rng,
    );
    BootstrappingKey {
        lwe_sk_ggsw_enc,
        ksk,
    }
}

pub fn bootstrap(
    tfhe_params: &TfheParams,
    lwe_ciphertext: &LweCiphertext,
    lwe_secret_key: &LweSecretKey,
    glwe_secret_key: &GlweSecretKey,
    bootstrapping_key: &BootstrappingKey,
    test_vector_poly: &Array1<u32>,
) -> LweCiphertext {
    // switch modulus of LWE from q to 2N
    let approximate_lwe = switch_modulus(
        lwe_ciphertext.data.as_slice().unwrap(),
        tfhe_params.log_q,
        tfhe_params.log_degree + 1,
    );

    let glwe_params = tfhe_params.glwe_params();

    // let mod_2n = 1u32 << (tfhe_params.log_degree + 1);
    // let mut tmp = mod_2n - approximate_lwe[tfhe_params.n];

    // X^-b
    let b_approx = Monomial {
        index: -(approximate_lwe[tfhe_params.n] as isize),
    };
    let v_x = trivial_encrypt_glwe_plaintext(
        &glwe_params,
        &GlweCleartext::encode_message(test_vector_poly.as_slice().unwrap(), &glwe_params),
    );
    let mut acc = &v_x * &b_approx;

    let ggsw_params = tfhe_params.ggsw_params();

    for i in 0..tfhe_params.n {
        // tmp = (tmp + (approximate_lwe[i])) % mod_2n;

        // acc * X^{a_i}
        let mut c1 = &acc
            * &Monomial {
                index: approximate_lwe[i] as isize,
            };

        acc = cmux(
            &ggsw_params,
            &bootstrapping_key.lwe_sk_ggsw_enc[i],
            &acc,
            &mut c1,
        );
    }

    // extract the constant term in acc GLWE ciphertext
    let bootstrapped_lwe = sample_extract(&acc, &glwe_params, 0);

    let signed_decomposer = SignedDecomposer::new(tfhe_params.decomposer.clone());

    // key switch bootstrapped LWE
    let key_switched_lwe = key_switch_lwe(
        &bootstrapped_lwe,
        &tfhe_params.lwe_params_post_pbs(),
        &tfhe_params.lwe_params(),
        &signed_decomposer,
        &bootstrapping_key.ksk,
    );

    key_switched_lwe
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
    fn bootstrapping_works() {
        let mut rng = thread_rng();

        let tfhe_params = TfheParams::default();

        let lwe_params = tfhe_params.lwe_params();
        let lwe_secret_key = LweSecretKey::random(&lwe_params, &mut rng);

        let glwe_params = tfhe_params.glwe_params();
        let glwe_secret_key = GlweSecretKey::random(&glwe_params, &mut rng);

        let bootstrapping_key =
            bootstrapping_key_gen(&tfhe_params, &lwe_secret_key, &glwe_secret_key, &mut rng);

        // encrypt LWE
        let lwe_cleartext = LweCleartext { message: 1 };
        let lwe_plaintext = lwe_cleartext.encode(&lwe_params);
        let lwe_ciphertext =
            encrypt_lwe_plaintext(&lwe_params, &lwe_secret_key, &lwe_plaintext, &mut rng);

        let test_vector_poly = construct_test_vector(&tfhe_params, |lhs, rhs| lhs & rhs);

        let bootstrapped_lwe_ct = bootstrap(
            &tfhe_params,
            &lwe_ciphertext,
            &lwe_secret_key,
            &glwe_secret_key,
            &bootstrapping_key,
            &test_vector_poly,
        );

        let lwe_plaintext_post_pbs =
            decrypt_lwe(&lwe_params, &lwe_secret_key, &bootstrapped_lwe_ct);
        let lwe_cleartext_post_pbs = lwe_plaintext_post_pbs.decode(&lwe_params);

        assert_eq!(lwe_cleartext, lwe_cleartext_post_pbs);
    }

    #[test]
    fn blind_rotate_in_clear() {
        let mut rng = thread_rng();

        let tfhe_params = TfheParams::default();

        let lwe_params = tfhe_params.lwe_params();
        let lwe_secret_key = LweSecretKey::random(&lwe_params, &mut rng);
        // encrypt LWE
        let lwe_plaintext = LweCleartext::encode_message(1, &lwe_params);
        let lwe_ciphertext =
            encrypt_lwe_plaintext(&lwe_params, &lwe_secret_key, &lwe_plaintext, &mut rng);

        let test_vector_poly = construct_test_vector(&tfhe_params, |lhs, rhs| lhs & rhs);
        let test_vector_poly = GlweCleartext::encode_message(
            test_vector_poly.as_slice().unwrap(),
            &tfhe_params.glwe_params(),
        )
        .data;

        let approximate_lwe = switch_modulus(
            lwe_ciphertext.data.as_slice().unwrap(),
            tfhe_params.log_q,
            tfhe_params.log_degree + 1,
        );

        let mod_2n = 1u32 << (tfhe_params.log_degree + 1);
        let mut tmp = mod_2n - approximate_lwe[tfhe_params.n];

        let b_approx = approximate_lwe[tfhe_params.n];
        let mut acc = poly_mul_monomial_custom_mod(
            test_vector_poly.view(),
            -(b_approx as isize),
            tfhe_params.log_q,
        );

        for i in 0..tfhe_params.n {
            if lwe_secret_key.data[i] != 0 {
                tmp = (tmp + (approximate_lwe[i])) % mod_2n;

                acc = poly_mul_monomial_custom_mod(
                    acc.view(),
                    ((approximate_lwe[i]) as isize),
                    tfhe_params.log_q,
                );
            }
        }

        dbg!(tmp);
        dbg!(acc);
        dbg!(poly_mul_monomial_custom_mod(
            test_vector_poly.view(),
            (tmp as isize),
            tfhe_params.log_q
        ));
    }
}
