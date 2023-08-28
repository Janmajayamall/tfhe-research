use ndarray::Array1;

use crate::{glwe::GlweParams, lwe::LweParams, TfheParams};

pub fn construct_test_vector_boolean(
    tfhe_params: &TfheParams,
    f: fn(u32, u32) -> u32,
) -> Array1<u32> {
    let plaintext_modulus = 1u32 << tfhe_params.log_p;

    // iterate over possible inputs
    // For now we limit to identity function
    let mut lookup_table = vec![];
    for i in 0..plaintext_modulus {
        // extracts 0^th and 1^st bit as inputs to right and left gate
        lookup_table.push(f((i >> 1) & 1, i & 1));
    }

    construct_test_from_lut(tfhe_params, &lookup_table)
}

/// Constructs test vector that evaluates Identity function in PBS
pub fn construct_identity_test_vector(tfhe_params: &TfheParams) -> Array1<u32> {
    let plaintext_modulus = 1u32 << tfhe_params.log_p;

    // iterate over possible inputs
    // For now we limit to identity function
    let mut lookup_table = vec![];
    for i in 0..plaintext_modulus {
        // extracts 0^th and 1^st bit as inputs to right and left gate
        lookup_table.push(i);
    }

    construct_test_from_lut(tfhe_params, &lookup_table)
}

/// Given a LUT for each input in plaintext space, constructs test vector
pub fn construct_test_from_lut(tfhe_params: &TfheParams, lut: &[u32]) -> Array1<u32> {
    // look up table must contain outputs for each possible input
    let plaintext_modulus = 1u32 << tfhe_params.log_p;
    assert!(lut.len() == plaintext_modulus as usize);

    // construct test vector
    let mut test_vector = vec![];
    // We must assure that 2^{log_p}| degree
    let repetition = (1 << tfhe_params.glwe_poly_degree) / (1 << tfhe_params.log_p);

    // repeat each look up value `repetition` times
    lut.iter().for_each(|v| {
        for _ in 0..repetition {
            test_vector.push(*v);
        }
    });
    // negate first repetition/2 values and rotate test_vector to left by repetition/2. Due to error if original plaintext is 0, b - \sum{a_is_i} can be negative and must be rounded
    // to 0. During bootstrapping, this translates to $\mu$ being negative and X^{-\mu}*v(x) resulting into some top coefficient. Thus we need to map top coefficients
    // (exactly reptition/2) with value of 0. However, calculating X^{-\mu}*v(x) if \mu were negative will cause top coefficients to reach lowest coefficints and due to negacylic
    // property of polynomial the coefficients will be negated. Thus we must negate values stored in top coefficients that might map to 0.
    for i in 0..(repetition / 2) {
        // negate the values
        if test_vector[i] != 0 {
            test_vector[i] = plaintext_modulus - test_vector[i];
        }
    }
    test_vector.rotate_left(repetition / 2);

    Array1::from_vec(test_vector)
}

#[cfg(test)]
mod tests {
    use crate::{glwe::GlweParams, lwe::LweParams, TfheParams};

    use super::construct_test_vector_boolean;

    #[test]
    fn test_vector_works() {
        let tfhe_params = TfheParams::default();
        let test_vector = construct_test_vector_boolean(&tfhe_params, |lhs, rhs| lhs & rhs);
        dbg!(test_vector);
    }
}
