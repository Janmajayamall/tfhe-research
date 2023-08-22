use std::ops::{AddAssign, Mul, MulAssign};

use itertools::izip;
use ndarray::{
    concatenate, Array, Array1, Array2, ArrayView, ArrayView1, ArrayView2, Axis, Dimension, Shape,
    ShapeBuilder,
};
use rand::{distributions::Standard, prelude::Distribution, CryptoRng, Rng, RngCore};

pub fn sample_gaussian<R: CryptoRng + RngCore>(mean: f64, std_dev: f64, rng: &mut R) -> u32 {
    return 0;
}

pub fn sample_gaussian_array<S: ShapeBuilder, R: CryptoRng + RngCore>(
    mean: f64,
    std_dev: f64,
    rng: &mut R,
    shape: S,
) -> Array<u32, S::Dim> {
    let mut res = Array::zeros(shape);
    let fill = res.as_slice_mut().unwrap();
    fill.iter_mut().for_each(|v| {
        *v = sample_gaussian(mean, std_dev, rng);
    });
    res
}

pub fn sample_binary_array<
    A: Clone + num_traits::Zero + From<u8>,
    S: ShapeBuilder,
    R: CryptoRng + RngCore,
>(
    rng: &mut R,
    shape: S,
) -> Array<A, S::Dim>
where
    Standard: Distribution<A>,
{
    let mut res = Array::zeros(shape);
    let fill = res.as_slice_mut().unwrap();

    let mut curr_byte: u8 = rng.gen::<u8>();
    let mut bit_index = 0;
    fill.iter_mut().for_each(|v| {
        *v = ((curr_byte >> bit_index) & 1).into();
        bit_index += 1;

        if bit_index == 8 {
            curr_byte = rng.gen::<u8>();
            bit_index = 0;
        }
    });
    res
}

pub fn sample_uniform_array<A: Clone + num_traits::Zero, S: ShapeBuilder, R: CryptoRng + RngCore>(
    rng: &mut R,
    shape: S,
) -> Array<A, S::Dim>
where
    Standard: Distribution<A>,
{
    let mut res = Array::zeros(shape);
    let fill = res.as_slice_mut().unwrap();
    fill.iter_mut().for_each(|v| {
        *v = rng.gen();
    });
    res
}

// pub fn sign_matrix(rows: usize, cols: usize) -> Array2 {}

pub fn teoplitz(p: ArrayView1<u32>) -> Array2<u32> {
    // generate sign matrix
    // g = [
    //     1, -1, -1, -1 -1
    //     1,  1, -1, -1 -1
    //     1,  1,  1, -1 -1
    //     1,  1,  1,  1 -1
    //     1,  1,  1,  1  1
    // ]

    // generate circulant matrix for p
    // p_c = [
    //     0 4 3 2 1
    //     1 0 4 3 2 <= rotate right by 1
    //     2 1 0 4 3
    //     3 2 1 0 4
    //     4 3 2 1 0
    // ]

    // element wise mutliplication of p_c and g yields
    // p_c * g = [
    //     0 -4 -3 -2 -1
    //     1  0 -4 -3 -2 <= rotate right by 1
    //     2  1  0 -4 -3
    //     3  2  1  0 -4
    //     4  3  2  1  0

    let n = p.shape()[0];
    let original = p.as_slice().unwrap();
    let mut matrix = vec![];
    for i in 0..n {
        for j in (0..i + 1).rev() {
            matrix.push(original[j]);
        }
        for j in (i + 1..n).rev() {
            matrix.push(original[j].wrapping_neg());
        }
    }
    Array2::from_shape_vec((n, n), matrix).unwrap()
}

pub fn poly_mul(p0: ArrayView1<u32>, p1: ArrayView1<u32>) -> Array1<u32> {
    // convert p0 to teoplitz matrix
    let p0_teoplitz = teoplitz(p0.view());
    let mut res = p0_teoplitz.dot(&p1);
    res
}

/// multiply the polynomials and add them
pub fn poly_mul_add_list(p0: &ArrayView2<u32>, p1: &ArrayView2<u32>) -> Array1<u32> {
    let mut res = poly_mul(p0.row(0), p1.row(0));
    izip!(p0.outer_iter(), p1.outer_iter())
        .skip(1)
        .for_each(|(r0, r1)| {
            let r = poly_mul(r0, r1);
            res.add_assign(&r);
        });

    res
}

pub fn school_book_negacylic_mul(p0: ArrayView1<u32>, p1: ArrayView1<u32>) -> Array1<u32> {
    let n = p0.shape()[0];
    let p0 = p0.as_slice().unwrap();
    let p1 = p1.as_slice().unwrap();
    let mut res = vec![0u32; n];
    for i in 0..n {
        for j in (0..i + 1) {
            res[i] = res[i].wrapping_add(p0[j].wrapping_mul(p1[i - j]));
        }
        for j in (i + 1..n) {
            res[i] = res[i].wrapping_sub(p0[j].wrapping_mul(p1[n - (j - i)]));
        }
    }

    Array1::from_vec(res)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use rand::thread_rng;

    #[test]
    fn teoplitz_works() {
        let v = Array1::from_vec(vec![0u32, 1, 2, 3, 4]);
        let res = teoplitz(v.view());
        dbg!(res);
    }

    #[test]
    fn poly_mul_works() {
        let v0 = Array1::from_vec(vec![12, 4, 123, 43, 3, 2, 3]);
        let v1 = Array1::from_vec(vec![12, 232, 5, 3, 2, 4, 2]);
        let teoplitz_res = poly_mul(v0.view(), v1.view());
        let school_book_res = school_book_negacylic_mul(v0.view(), v1.view());

        assert_eq!(teoplitz_res, school_book_res);
    }

    #[test]
    fn generate_random() {
        let mut rng = thread_rng();
        let f: Array2<u32> = sample_uniform_array(&mut rng, (10, 12));
        dbg!(f);
    }
}
