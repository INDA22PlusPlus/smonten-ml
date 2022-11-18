use nalgebra::SMatrix;

pub struct Layer{
    this_size: usize,
    prev_size: usize,
    w: SMatrix<T, R, C>
}