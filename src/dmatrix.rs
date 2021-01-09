use crate::bindings::DMatrixHandle;
use crate::errors::TreeRiteError;
use crate::sys::{dmatrix_create_from_array, dmatrix_create_from_slice, dmatrix_free, dmatrix_get_dimension, FloatInfo};
use fehler::throws;
use ndarray::{AsArray, Ix2};
use num_traits::Float;
use std::convert::TryInto;
use std::marker::PhantomData;

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
    #[throws(TreeRiteError)]
    pub fn from_2darray<'a, A>(array: A) -> DMatrix<F>
    where
        A: AsArray<'a, F, Ix2>,
        F: 'a,
    {
        let handle = dmatrix_create_from_array(array.into())?;
        DMatrix { handle, _phantom: PhantomData }
    }

    #[throws(TreeRiteError)]
    pub fn from_slice<'a>(array: &'a [F]) -> DMatrix<F> {
        array.try_into()?
    }
}

impl<F> DMatrix<F> {
    #[throws(TreeRiteError)]
    pub fn nrows(&self) -> u64 {
        dmatrix_get_dimension(self.handle)?.0
    }

    #[throws(TreeRiteError)]
    pub fn ncols(&self) -> u64 {
        dmatrix_get_dimension(self.handle)?.1
    }
}

impl<'a, F: Float + FloatInfo> TryInto<DMatrix<F>> for &'a [F] {
    type Error = TreeRiteError;

    fn try_into(self) -> Result<DMatrix<F>, Self::Error> {
        let handle = dmatrix_create_from_slice(self)?;
        Ok(DMatrix { handle, _phantom: PhantomData })
    }
}

impl<F> Drop for DMatrix<F> {
    fn drop(&mut self) {
        match dmatrix_free(self.handle) {
            Ok(()) => {}
            Err(e) => {
                if cfg!(feature = "free_panic") {
                    panic!("cannot free dmatrix: {}", e)
                }
            }
        }
    }
}
