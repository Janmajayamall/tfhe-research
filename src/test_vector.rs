use ndarray::Array1;

use crate::{glwe::GlweParams, lwe::LweParams, TfheParams};

pub fn construct_test_vector(tfhe_params: &TfheParams) -> Array1<u32> {
    let plaintext_modulus = 1u32 << tfhe_params.log_p;

    // iterate over possible inputs
    // For now we limit to identity function
    let mut lookup_table = vec![];
    for i in 0..plaintext_modulus {
        lookup_table.push(i);
    }

    // construct test vector
    let mut test_vector = vec![];
    // We must assure that 2^{log_p}| degree
    let repetition = (1 << tfhe_params.log_degree) / (1 << tfhe_params.log_p);

    // repeat each look up value `repetition` times
    lookup_table.iter().for_each(|v| {
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

    use super::construct_test_vector;

    #[test]
    fn test_vector_works() {
        let tfhe_params = TfheParams::default();
        let test_vector = construct_test_vector(&tfhe_params);
        dbg!(test_vector);
    }
}
