use nalgebra::{
    base::allocator::Allocator, default_allocator::DefaultAllocator, storage::Storage, *,
};
use num_traits::{One, Zero};

pub fn weighted_sum<'m, 'w, Vs, Ws, N, D, S>(
    vectors: Vs,
    weights: Ws,
) -> Vector<N, D, <DefaultAllocator as Allocator<N, D>>::Buffer>
where
    N: Scalar + Zero + One + ClosedAdd + ClosedMul + Copy,
    D: Dim,
    DefaultAllocator: Allocator<N, D>,
    S: 'm,
    S: Storage<N, D>,

    Vs: IntoIterator<Item = &'m Vector<N, D, S>>,
    Ws: IntoIterator<Item = &'w N>,
{
    let mut zip_iter = vectors.into_iter().zip(weights);
    let mut acc = match zip_iter.next() {
        Some((m, &w)) => m * w,
        None => panic!("Cannot compute the weighted sum in any of the iterators is empty."),
    };
    for (y, &w) in zip_iter {
        acc.axpy(w, y, N::one());
    }
    acc
}
