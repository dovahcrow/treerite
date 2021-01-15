use crate::errors::TreeRiteError;
use crate::sys::DMatrixHandle;
use crate::sys::{treelite_dmatrix_create_from_array, treelite_dmatrix_create_from_slice, treelite_dmatrix_free, treelite_dmatrix_get_dimension, FloatInfo};
use fehler::throws;
use ndarray::{AsArray, Ix2};
use num_traits::Float;
use std::convert::TryInto;
use std::marker::PhantomData;

/// Data matrix used in Treerite.
pub struct DMatrix<F> {
    pub(crate) handle: DMatrixHandle,
    _phantom: PhantomData<F>,
}

unsafe impl<F> Sync for DMatrix<F> where F: Sync {}
unsafe impl<F> Send for DMatrix<F> where F: Send {}

impl<F> DMatrix<F>
where
    F: Float + FloatInfo,
{
    /// Create a DMatrix from any type that can be converted to a 2d ndarray::ArrayView. This function is zero copy.
    #[throws(TreeRiteError)]
    pub fn from_2darray<'a, A>(array: A) -> DMatrix<F>
    where
        A: AsArray<'a, F, Ix2>,
        F: 'a,
    {
        let handle = treelite_dmatrix_create_from_array(array.into())?;
        DMatrix { handle, _phantom: PhantomData }
    }

    /// Create a single row DMatrix from a slice of floats. Useful for prediction for a single instance.
    /// This function is zero copy.
    #[throws(TreeRiteError)]
    pub fn from_slice<'a>(array: &'a [F]) -> DMatrix<F> {
        array.try_into()?
    }
}

impl<F> DMatrix<F> {
    /// Number of rows in this DMatrix
    #[throws(TreeRiteError)]
    pub fn nrows(&self) -> u64 {
        treelite_dmatrix_get_dimension(self.handle)?.0
    }

    /// Number of columns in this DMatrix
    #[throws(TreeRiteError)]
    pub fn ncols(&self) -> u64 {
        treelite_dmatrix_get_dimension(self.handle)?.1
    }
}

impl<'a, F: Float + FloatInfo> TryInto<DMatrix<F>> for &'a [F] {
    type Error = TreeRiteError;

    fn try_into(self) -> Result<DMatrix<F>, Self::Error> {
        let handle = treelite_dmatrix_create_from_slice(self)?;
        Ok(DMatrix { handle, _phantom: PhantomData })
    }
}

impl<F> Drop for DMatrix<F> {
    fn drop(&mut self) {
        match treelite_dmatrix_free(self.handle) {
            Ok(()) => {}
            Err(e) => {
                if cfg!(feature = "free_panic") {
                    panic!("cannot free dmatrix: {}", e)
                }
            }
        }
    }
}
