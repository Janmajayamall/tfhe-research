use crate::{
    decomposer::{self, SignedDecomposer},
    lwe::{encrypt_lwe_zero, LweCiphertext, LweParams, LweSecretKey},
};
use itertools::{izip, Itertools};
use ndarray::{Array1, Array2};
use rand::{CryptoRng, RngCore};
use std::ops::Neg;

/// Key switching key is stored as 2d array with each row consisting of a single LWE ciphertext.
/// First `levels` rows are recomposition factor LWE ciphertexts corresponding to secret bit at index
/// 0, and so on and so forth.
pub struct KeySwitchingKey {
    pub(crate) data: Array2<u32>,
}

impl KeySwitchingKey {
    /// Generates key switching key using which one can switching LWE ciphertext
    /// encrypted under `from_lwe_sk` to `to_lwe_sk`
    pub fn generate_ksk<R: CryptoRng + RngCore>(
        from_lwe_sk: &LweSecretKey,
        to_lwe_sk: &LweSecretKey,
        from_lwe_params: &LweParams,
        to_lwe_params: &LweParams,
        signed_decomposer: &SignedDecomposer,
        rng: &mut R,
    ) -> KeySwitchingKey {
        // (from_n*levels) x to_n
        let mut ksk = Array2::zeros((
            from_lwe_params.lwe_dimension * signed_decomposer.params.levels,
            to_lwe_params.lwe_dimension + 1,
        ));
        from_lwe_sk
            .data
            .iter()
            .enumerate()
            .for_each(|(s_index, s_bit)| {
                // encrypt secret bit with recomposition factor $\beta^{l-(level_index+1)}$ for all levels
                let l = signed_decomposer.params.log_q / signed_decomposer.params.log_base;
                for level_index in 0..signed_decomposer.params.levels {
                    let mut factor =
                        1 << (signed_decomposer.params.log_base * (l - (level_index + 1)));
                    factor *= *s_bit;

                    // encrypt s_bit * factor
                    let mut zero_encryption = encrypt_lwe_zero(to_lwe_params, to_lwe_sk, rng);
                    zero_encryption.data[to_lwe_params.lwe_dimension] =
                        zero_encryption.data[to_lwe_params.lwe_dimension].wrapping_add(factor);

                    // row in ksk
                    let row_ksk = s_index * signed_decomposer.params.levels + level_index;
                    ksk.row_mut(row_ksk)
                        .as_slice_mut()
                        .unwrap()
                        .copy_from_slice(zero_encryption.data.as_slice().unwrap());
                }
            });

        KeySwitchingKey { data: ksk }
    }
}

pub fn key_switch_lwe(
    lwe_ciphertext: &LweCiphertext,
    from_lwe_params: &LweParams,
    to_lwe_params: &LweParams,
    signed_decomposer: &SignedDecomposer,
    key_switching_key: &KeySwitchingKey,
) -> LweCiphertext {
    // decompose a_i's in lwe_ciphertext
    let a_decomposed_vector = lwe_ciphertext
        .data
        .as_slice()
        .unwrap()
        .iter()
        .take(from_lwe_params.lwe_dimension)
        .flat_map(|a_i| signed_decomposer.decompose(*a_i))
        .collect_vec();

    // \sum (\sum a_ij * LWE(s_ij))
    let mut sum = Array1::zeros(to_lwe_params.lwe_dimension + 1);
    izip!(
        a_decomposed_vector.iter(),
        key_switching_key.data.outer_iter()
    )
    .for_each(|(a_ij, s_ij_lwe)| {
        // multiply a_ij with LWE(s_ij) and add to sum
        sum.scaled_add(*a_ij, &s_ij_lwe);
    });

    // b - \sum (\sum a_ij * LWE(s_ij))
    let b = lwe_ciphertext
        .data
        .get(from_lwe_params.lwe_dimension)
        .unwrap();
    // negate sum
    sum.iter_mut().for_each(|v| *v = v.wrapping_neg());
    // add b part of `from LWE` to b part of `to LWE`
    let sum_b = sum.get_mut(to_lwe_params.lwe_dimension).unwrap();
    *sum_b = sum_b.wrapping_add(*b);

    LweCiphertext { data: sum }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::{
        decomposer::SignedDecomposer,
        lwe::{decrypt_lwe, encrypt_lwe_plaintext, LweCleartext, LweSecretKey},
        TfheParams,
    };

    use super::{key_switch_lwe, KeySwitchingKey};

    #[test]
    fn key_switching_works() {
        let mut rng = thread_rng();

        let tfhe_params = TfheParams::default();

        let lwe_cleartext = LweCleartext { message: 1 };

        let from_params = tfhe_params.lwe_params_post_pbs();
        let from_sk = LweSecretKey::random(&from_params, &mut rng);
        let from_lwe_ciphertext = encrypt_lwe_plaintext(
            &from_params,
            &from_sk,
            &lwe_cleartext.encode(&from_params),
            &mut rng,
        );

        let to_params = tfhe_params.lwe_params();
        let to_sk = LweSecretKey::random(&to_params, &mut rng);

        let signed_decomposer = SignedDecomposer::new(tfhe_params.ks_decomposer);

        // gen ksk
        let ksk = KeySwitchingKey::generate_ksk(
            &from_sk,
            &to_sk,
            &from_params,
            &to_params,
            &signed_decomposer,
            &mut rng,
        );

        let switched_lwe = key_switch_lwe(
            &from_lwe_ciphertext,
            &from_params,
            &to_params,
            &signed_decomposer,
            &ksk,
        );
        let switched_plaintext = decrypt_lwe(&to_params, &to_sk, &switched_lwe);

        assert_eq!(switched_plaintext.decode(&to_params), lwe_cleartext);
    }
}
