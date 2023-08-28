use crate::{
    bootstrapping::{bootstrap, BootstrappingKey},
    lwe::LweCiphertext,
    test_vector::construct_test_vector_boolean,
    utils::random_keys,
    TfheParams,
};

fn and(
    tfhe_params: &TfheParams,
    ct0: &LweCiphertext,
    ct1: &LweCiphertext,
    bk: &BootstrappingKey,
) -> LweCiphertext {
    let test_vector_poly = construct_test_vector_boolean(tfhe_params, |lhs, rhs| lhs & rhs);
    let (lwe_secret_key, glwe_secret_key) = random_keys(tfhe_params);

    let ct_in = &(ct1 * 2u32) + ct0;

    let ct_out = bootstrap(
        tfhe_params,
        &ct_in,
        &lwe_secret_key,
        &glwe_secret_key,
        bk,
        &test_vector_poly,
    );

    ct_out
}

fn or(
    tfhe_params: &TfheParams,
    ct0: &LweCiphertext,
    ct1: &LweCiphertext,
    bk: &BootstrappingKey,
) -> LweCiphertext {
    let test_vector_poly = construct_test_vector_boolean(tfhe_params, |lhs, rhs| lhs | rhs);
    let (lwe_secret_key, glwe_secret_key) = random_keys(tfhe_params);

    let ct_in = &(ct1 * 2u32) + ct0;

    let ct_out = bootstrap(
        tfhe_params,
        &ct_in,
        &lwe_secret_key,
        &glwe_secret_key,
        bk,
        &test_vector_poly,
    );

    ct_out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bootstrapping::bootstrapping_key_gen,
        glwe::GlweSecretKey,
        lwe::{decrypt_lwe, encrypt_lwe_plaintext, LweCleartext, LweSecretKey},
        TfheParams,
    };
    use rand::thread_rng;

    #[test]
    fn boolean_gates_work() {
        let mut rng = thread_rng();
        let tfhe_params = TfheParams::default();
        let lwe_params = tfhe_params.lwe_params();
        let glwe_params = tfhe_params.glwe_params();

        let lwe_secret_key = LweSecretKey::random(&lwe_params, &mut rng);
        let glwe_secret_key = GlweSecretKey::random(&glwe_params, &mut rng);
        let bk = bootstrapping_key_gen(&tfhe_params, &lwe_secret_key, &glwe_secret_key, &mut rng);
        for i in 0..(1 << tfhe_params.log_p) {
            let lhs = (i >> 1) & 1;
            let rhs = i & 1;

            let ct1 = encrypt_lwe_plaintext(
                &lwe_params,
                &lwe_secret_key,
                &LweCleartext::encode_message(lhs, &lwe_params),
                &mut rng,
            );
            let ct0 = encrypt_lwe_plaintext(
                &lwe_params,
                &lwe_secret_key,
                &LweCleartext::encode_message(rhs, &lwe_params),
                &mut rng,
            );

            let c1_and_c0 = and(&tfhe_params, &ct0, &ct1, &bk);

            let pt = decrypt_lwe(&lwe_params, &lwe_secret_key, &c1_and_c0);
            let res = pt.decode(&lwe_params);

            assert_eq!(lhs & rhs, res.message);
            println!("Works for {i} ");
        }
    }
}
