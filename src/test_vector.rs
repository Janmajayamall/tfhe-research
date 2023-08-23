use crate::{glwe::GlweParams, lwe::LweParams};

pub fn construct_test_vector(glwe_params: &GlweParams, lwe_params: &LweParams) -> Vec<u32> {
    // iterate over possible inputs
    // For now we limit to identity function
    let mut lookup_table = vec![];
    for i in 0..(1 << lwe_params.log_p) {
        lookup_table.push(i as u32);
    }

    // construct test vector
    let mut test_vector = vec![];
    // We must assure that 2^{log_p}| glwe_params.N
    let repetition = glwe_params.N / (1 << lwe_params.log_p);
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

    test_vector
}

#[cfg(test)]
mod tests {
    use crate::{glwe::GlweParams, lwe::LweParams};

    use super::construct_test_vector;

    #[test]
    fn test_vector_works() {
        let glwe_params = GlweParams {
            k: 2,
            N: 512,
            log_q: 32,
            log_p: 8,
            mean: 0.0,
            std_dev: 0.0,
        };
        let lwe_params = LweParams {
            n: 512,
            log_q: 32,
            log_p: 8,
            mean: 0.0,
            std_dev: 0.0,
        };
        let test_vector = construct_test_vector(&glwe_params, &lwe_params);
        dbg!(test_vector);
    }
}
