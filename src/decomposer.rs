pub struct DecomposerParams {
    pub(crate) log_base: usize,
    pub(crate) levels: usize,
    pub(crate) log_q: usize,
}

struct SignedDecomposer {
    params: DecomposerParams,
}

impl SignedDecomposer {
    fn new(params: DecomposerParams) -> SignedDecomposer {
        SignedDecomposer { params }
    }

    fn round_value(&self, value: u32) -> u32 {
        let ignored_bits = self.params.log_q - self.params.log_base * self.params.levels;

        if ignored_bits == 0 {
            return value;
        }

        // round value
        let ignored_mask = (1 << ignored_bits) - 1;
        let ignored_value = value & ignored_mask;
        let ignored_msb = ignored_value >> (ignored_bits - 1);

        ((value >> ignored_bits) + ignored_msb) << ignored_bits
    }

    fn decompose(&self, value: u32) -> Vec<u32> {
        let mut decomposition = vec![];

        // round the value to precision
        let value = self.round_value(value);

        let log_base = self.params.log_base;
        let base_mask = (1 << log_base) - 1;
        // to check whether value >=B/2
        let base_by_2_mask = 1 << (log_base - 1);
        let mut carry = 0;
        for l in 0..(self.params.log_q / self.params.log_base) {
            let mut res = ((value >> (log_base * l)) & base_mask) + carry;

            // will carry mask will equal base_by_2_mask only when res >= B/2, otherwise 0
            let carry_mask = res & base_by_2_mask;

            // subtract B from ` only when res >= B/2
            res = (res as i32 - (carry_mask << 1) as i32) as u32;

            // set carry
            carry = carry_mask >> (log_base - 1);

            decomposition.push(res);
        }

        // Switch to big endian
        decomposition.reverse();
        // dbg!(&decomposition);
        // truncate upto precision
        let mut curr = decomposition.len();

        while curr != self.params.levels {
            decomposition.remove(curr - 1);
            curr -= 1;
        }

        decomposition
    }

    // MSB -> LSB
    fn recompose(&self, legs: &[u32]) -> u32 {
        let levels = self.params.levels;
        let log_base = self.params.log_base;

        let mut value = 0u32;
        legs.iter().enumerate().for_each(|(index, leg)| {
            let leg_shifted = leg << (log_base * (levels - 1 - index));
            value = value.wrapping_add(leg_shifted);
        });

        let ignored_bits = self.params.log_q - (self.params.log_base * self.params.levels);
        value << ignored_bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decomposition() {
        let signed_decomposer = SignedDecomposer::new(DecomposerParams {
            log_base: 4,
            levels: 7,
            log_q: 32,
        });

        for i in 0..100_000_000 {
            let decomposed = signed_decomposer.decompose(i);
            let recomposed_value = signed_decomposer.recompose(&decomposed);
            assert_eq!(recomposed_value, signed_decomposer.round_value(i));
        }
    }
}
