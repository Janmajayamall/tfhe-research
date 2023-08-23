use ndarray::Array1;

use crate::{glwe::GlweParams, lwe::LweParams, TfheParams};

pub fn construct_test_vector(tfhe_params: &TfheParams) -> Array1<u32> {
    // iterate over possible inputs
    // For now we limit to identity function
    let mut lookup_table = vec![];
    for i in 0..(1 << tfhe_params.log_p) {
        lookup_table.push(i as u32);
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
    // negate first repetition/2 values and rotate test_vector to left by repetition/2. Due to error, b - \sum{a_is_i} can be negative and must be rounded
    // to 0. During bootstrapping, this translates to $\mu$ being negative and X^{-\mu}*v(x) resulting into some top coefficient. Thus we need to map top coefficients
    // (exactly reptition/2) with value of 0. However, calculating X^{-\mu}*v(x) if \mu were negative will cause top coefficients to reach lowest coefficints and due to negacylic
    // property the coefficients will be negated. Thus we must negate values stored in top coefficients that must map to 0.
    for i in 0..(repetition / 2) {
        test_vector[i] = test_vector[i].wrapping_neg();
    }
    test_vector.rotate_left(repetition / 2);

    // scale test vector
    test_vector.iter_mut().for_each(|v| {
        *v = *v << (tfhe_params.log_q - tfhe_params.log_p);
    });

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