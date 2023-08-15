use itertools::izip;

pub fn dot_product(a: &[u32], b: &[u32]) -> u32 {
    let mut d_product = 0;
    izip!(a.iter(), b.iter()).for_each(|(a0, b0)| {
        d_product += a0 * b0;
    });
    d_product
}
