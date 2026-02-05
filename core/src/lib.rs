//! Minimal core ndarray prototype for AeroNum.
//!
//! This is intentionally tiny and exists to make the v0 spec testable via `cargo test`.

#[derive(Clone, Debug, PartialEq)]
pub struct NdArray {
    data: Vec<f32>,
    shape: Vec<usize>,
    /// Strides in elements (not bytes).
    strides: Vec<isize>,
    /// Element offset into `data` for views.
    offset: isize,
}

impl NdArray {
    pub fn zeros(shape: &[usize]) -> Self {
        let size = shape.iter().product::<usize>();
        let data = vec![0.0; size];
        Self::from_data_shape(data, shape)
    }

    pub fn ones(shape: &[usize]) -> Self {
        let size = shape.iter().product::<usize>();
        let data = vec![1.0; size];
        Self::from_data_shape(data, shape)
    }

    pub fn arange(n: usize) -> Self {
        let data = (0..n).map(|i| i as f32).collect();
        Self::from_data_shape(data, &[n])
    }

    pub fn from_list(data: Vec<f32>, shape: Option<&[usize]>) -> Self {
        let n = data.len();
        match shape {
            None => Self::from_data_shape(data, &[n]),
            Some(shape) => {
                assert_eq!(shape.iter().product::<usize>(), n);
                Self::from_data_shape(data, shape)
            }
        }
    }

    pub fn view(&self, shape: &[usize], strides: &[isize], offset: isize) -> Self {
        assert_eq!(shape.len(), strides.len());
        // No bounds checking beyond basic sanity; tests cover invariants.
        Self {
            data: self.data.clone(),
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            offset: self.offset + offset,
        }
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Option<Self> {
        if !self.is_contiguous() {
            return None;
        }
        if new_shape.iter().product::<usize>() != self.len() {
            return None;
        }
        Some(Self::from_data_shape(self.to_vec(), new_shape))
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    pub fn offset(&self) -> isize {
        self.offset
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product::<usize>()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides == c_strides(&self.shape)
    }

    pub fn get(&self, idx: &[usize]) -> Option<f32> {
        let li = self.linear_index(idx)?;
        self.data.get(li).copied()
    }

    pub fn set(&mut self, idx: &[usize], value: f32) -> bool {
        let Some(li) = self.linear_index(idx) else {
            return false;
        };
        if let Some(x) = self.data.get_mut(li) {
            *x = value;
            true
        } else {
            false
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        let (out_shape, left, right) = broadcast_pair(self, other);
        let mut out = NdArray::zeros(&out_shape);
        for out_i in 0..out.len() {
            let mi = unravel_index(out_i, &out_shape);
            let a = left.get_broadcasted(&mi);
            let b = right.get_broadcasted(&mi);
            out.data[out_i] = a + b;
        }
        out
    }

    pub fn sub(&self, other: &Self) -> Self {
        let (out_shape, left, right) = broadcast_pair(self, other);
        let mut out = NdArray::zeros(&out_shape);
        for out_i in 0..out.len() {
            let mi = unravel_index(out_i, &out_shape);
            out.data[out_i] = left.get_broadcasted(&mi) - right.get_broadcasted(&mi);
        }
        out
    }

    pub fn mul(&self, other: &Self) -> Self {
        let (out_shape, left, right) = broadcast_pair(self, other);
        let mut out = NdArray::zeros(&out_shape);
        for out_i in 0..out.len() {
            let mi = unravel_index(out_i, &out_shape);
            out.data[out_i] = left.get_broadcasted(&mi) * right.get_broadcasted(&mi);
        }
        out
    }

    pub fn div(&self, other: &Self) -> Self {
        let (out_shape, left, right) = broadcast_pair(self, other);
        let mut out = NdArray::zeros(&out_shape);
        for out_i in 0..out.len() {
            let mi = unravel_index(out_i, &out_shape);
            out.data[out_i] = left.get_broadcasted(&mi) / right.get_broadcasted(&mi);
        }
        out
    }

    pub fn sum(&self, axis: Option<usize>, keepdims: bool) -> Self {
        match axis {
            None => {
                let s: f32 = (0..self.len())
                    .map(|i| {
                        let mi = unravel_index(i, &self.shape);
                        self.get(&mi).unwrap()
                    })
                    .sum();
                if keepdims {
                    NdArray::from_list(vec![s], Some(&vec![1; self.shape.len()]))
                } else {
                    NdArray::from_list(vec![s], Some(&[]))
                }
            }
            Some(ax) => {
                assert!(ax < self.shape.len());
                let mut out_shape = self.shape.clone();
                if keepdims {
                    out_shape[ax] = 1;
                } else {
                    out_shape.remove(ax);
                }
                let mut out = NdArray::zeros(&out_shape);

                for in_i in 0..self.len() {
                    let mi = unravel_index(in_i, &self.shape);
                    let mut oi = mi.clone();
                    if keepdims {
                        oi[ax] = 0;
                    } else {
                        oi.remove(ax);
                    }
                    let out_li = out.linear_index_usize(&oi);
                    out.data[out_li] += self.get(&mi).unwrap();
                }
                out
            }
        }
    }

    /// 2D matmul only.
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 2);
        let (m, k1) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        assert_eq!(k1, k2);
        let mut out = NdArray::zeros(&[m, n]);
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k1 {
                    acc += self.get(&[i, kk]).unwrap() * other.get(&[kk, j]).unwrap();
                }
                out.set(&[i, j], acc);
            }
        }
        out
    }

    pub fn to_vec(&self) -> Vec<f32> {
        (0..self.len())
            .map(|i| {
                let mi = unravel_index(i, &self.shape);
                self.get(&mi).unwrap()
            })
            .collect()
    }

    fn from_data_shape(data: Vec<f32>, shape: &[usize]) -> Self {
        let strides = c_strides(shape);
        Self {
            data,
            shape: shape.to_vec(),
            strides,
            offset: 0,
        }
    }

    fn linear_index(&self, idx: &[usize]) -> Option<usize> {
        if idx.len() != self.shape.len() {
            return None;
        }
        let mut li: isize = self.offset;
        for ((&i, &dim), &st) in idx.iter().zip(self.shape.iter()).zip(self.strides.iter()) {
            if i >= dim {
                return None;
            }
            li += (i as isize) * st;
        }
        if li < 0 {
            return None;
        }
        Some(li as usize)
    }

    fn linear_index_usize(&self, idx: &[usize]) -> usize {
        self.linear_index(idx).unwrap()
    }

    fn get_broadcasted(&self, out_multi_idx: &[usize]) -> f32 {
        // out_multi_idx is in output shape coordinates.
        // Align trailing dimensions.
        let out_nd = out_multi_idx.len();
        let in_nd = self.shape.len();
        let mut in_idx = vec![0usize; in_nd];
        for (a, in_i) in in_idx.iter_mut().enumerate() {
            let out_axis = out_nd - in_nd + a;
            *in_i = if self.shape[a] == 1 {
                0
            } else {
                out_multi_idx[out_axis]
            };
        }
        self.get(&in_idx).unwrap()
    }
}

fn c_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; shape.len()];
    let mut st: isize = 1;
    for (axis, &dim) in shape.iter().enumerate().rev() {
        strides[axis] = st;
        st *= dim as isize;
    }
    strides
}

fn unravel_index(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut idx = vec![0usize; shape.len()];
    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        idx[axis] = linear % dim;
        linear /= dim;
    }
    idx
}

fn broadcast_pair<'a>(a: &'a NdArray, b: &'a NdArray) -> (Vec<usize>, &'a NdArray, &'a NdArray) {
    let out = broadcast_shape(a.shape(), b.shape())
        .unwrap_or_else(|| panic!("incompatible shapes {:?} and {:?}", a.shape(), b.shape()));
    (out, a, b)
}

fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let na = a.len();
    let nb = b.len();
    let nd = na.max(nb);
    let mut out = vec![0usize; nd];
    for i in 0..nd {
        let da = if i < nd - na { 1 } else { a[i - (nd - na)] };
        let db = if i < nd - nb { 1 } else { b[i - (nd - nb)] };
        if da == db || da == 1 || db == 1 {
            out[i] = da.max(db);
        } else {
            return None;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn shape_stride_invariants_c_contiguous() {
        let a = NdArray::zeros(&[3, 4]);
        assert_eq!(a.shape(), &[3, 4]);
        assert_eq!(a.strides(), &[4, 1]);
        assert_eq!(a.offset(), 0);
        assert!(a.is_contiguous());
    }

    #[test]
    fn indexing_get_set() {
        let mut a = NdArray::zeros(&[2, 3]);
        assert!(a.set(&[1, 2], 7.5));
        assert_eq!(a.get(&[1, 2]), Some(7.5));
        assert_eq!(a.get(&[2, 0]), None);
    }

    #[test]
    fn view_uses_strides_and_offset() {
        // Base 1D of 0..10.
        let base = NdArray::arange(10);
        // Take elements 2..8 step 2 => [2,4,6]
        let v = base.view(&[3], &[2], 2);
        assert_eq!(v.shape(), &[3]);
        assert_eq!(v.strides(), &[2]);
        assert_eq!(v.offset(), 2);
        assert_eq!(v.to_vec(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn broadcasting_matrix_3x1_plus_1x4() {
        let a = NdArray::from_list(vec![1., 2., 3.], Some(&[3, 1]));
        let b = NdArray::from_list(vec![10., 20., 30., 40.], Some(&[1, 4]));
        let c = a.add(&b);
        assert_eq!(c.shape(), &[3, 4]);
        assert_eq!(
            c.to_vec(),
            vec![11., 21., 31., 41., 12., 22., 32., 42., 13., 23., 33., 43.,]
        );
    }

    #[test]
    fn broadcasting_vector_plus_matrix() {
        let v = NdArray::from_list(vec![1., 2., 3., 4.], Some(&[4]));
        let m = NdArray::from_list((0..12).map(|x| x as f32).collect(), Some(&[3, 4]));
        let out = v.add(&m);
        assert_eq!(out.shape(), &[3, 4]);
        assert_eq!(
            out.to_vec(),
            vec![1., 3., 5., 7., 5., 7., 9., 11., 9., 11., 13., 15.,]
        );
    }

    #[test]
    fn matmul_2d() {
        let a = NdArray::from_list(vec![1., 2., 3., 4., 5., 6.], Some(&[2, 3]));
        let b = NdArray::from_list(vec![7., 8., 9., 10., 11., 12.], Some(&[3, 2]));
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.to_vec(), vec![58., 64., 139., 154.,]);
    }

    #[test]
    fn sum_axis_none_scalar() {
        let a = NdArray::from_list(vec![1., 2., 3., 4.], Some(&[2, 2]));
        let s = a.sum(None, false);
        assert_eq!(s.shape(), &[]);
        assert_eq!(s.to_vec(), vec![10.]);
    }

    #[test]
    fn sum_axis_0_and_1_shapes_and_values() {
        let a = NdArray::from_list(vec![1., 2., 3., 4., 5., 6.], Some(&[2, 3]));
        // axis=0 sums rows => shape (3,)
        let s0 = a.sum(Some(0), false);
        assert_eq!(s0.shape(), &[3]);
        assert_eq!(s0.to_vec(), vec![5., 7., 9.]);

        // axis=1 sums cols => shape (2,)
        let s1 = a.sum(Some(1), false);
        assert_eq!(s1.shape(), &[2]);
        assert_eq!(s1.to_vec(), vec![6., 15.]);

        // keepdims
        let s1k = a.sum(Some(1), true);
        assert_eq!(s1k.shape(), &[2, 1]);
        assert_eq!(s1k.to_vec(), vec![6., 15.]);

        // smoke test approximate equality for float behavior
        assert_relative_eq!(s0.to_vec()[0], 5.0);
    }

    #[test]
    fn reshape_contiguous_preserves_row_major_order() {
        let a = NdArray::from_list((0..12).map(|x| x as f32).collect(), Some(&[3, 4]));
        let b = a
            .reshape(&[2, 2, 3])
            .expect("contiguous reshape should succeed");
        assert_eq!(b.shape(), &[2, 2, 3]);
        // Reshape should be a view-like reinterpretation (row-major), i.e. preserve linear order.
        assert_eq!(b.to_vec(), (0..12).map(|x| x as f32).collect::<Vec<_>>());
        assert!(b.is_contiguous());
    }

    #[test]
    fn reshape_size_mismatch_returns_none() {
        let a = NdArray::zeros(&[2, 3]);
        assert!(a.reshape(&[4]).is_none());
        assert!(a.reshape(&[2, 2, 2]).is_none());
    }

    #[test]
    fn reshape_non_contiguous_returns_none() {
        let base = NdArray::arange(10);
        // Non-contiguous: stride 2 with shape 3.
        let v = base.view(&[3], &[2], 2);
        assert!(!v.is_contiguous());
        assert!(v.reshape(&[1, 3]).is_none());
    }

    #[test]
    fn reshape_contiguous_view_with_offset_allowed() {
        // Contiguous view with nonzero offset should still be reshape-able.
        let base = NdArray::arange(6);
        let v = base.view(&[3], &[1], 2); // [2,3,4]
        assert!(v.is_contiguous());
        let r = v.reshape(&[1, 3]).unwrap();
        assert_eq!(r.shape(), &[1, 3]);
        assert_eq!(r.to_vec(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn broadcast_scalar_plus_matrix() {
        let s = NdArray::from_list(vec![2.0], Some(&[]));
        let m = NdArray::from_list((0..6).map(|x| x as f32).collect(), Some(&[2, 3]));
        let out = s.add(&m);
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.to_vec(), vec![2., 3., 4., 5., 6., 7.]);

        // commutative check (same broadcasting rules either side)
        let out2 = m.add(&s);
        assert_eq!(out2.to_vec(), out.to_vec());
    }

    #[test]
    fn broadcasting_higher_rank_edge_case() {
        // (2,1,3) + (1,4,1) -> (2,4,3)
        let a = NdArray::from_list((0..6).map(|x| x as f32).collect(), Some(&[2, 1, 3]));
        let b = NdArray::from_list((0..4).map(|x| (100 + x) as f32).collect(), Some(&[1, 4, 1]));
        let out = a.add(&b);
        assert_eq!(out.shape(), &[2, 4, 3]);
        // spot checks
        assert_eq!(out.get(&[0, 0, 0]), Some(0.0 + 100.0));
        assert_eq!(out.get(&[0, 3, 2]), Some(2.0 + 103.0));
        assert_eq!(out.get(&[1, 1, 1]), Some(4.0 + 101.0));
    }

    #[test]
    #[should_panic(expected = "incompatible shapes")]
    fn broadcasting_incompatible_shapes_panics() {
        let a = NdArray::zeros(&[2, 3]);
        let b = NdArray::zeros(&[2, 2]);
        let _ = a.add(&b);
    }
}
