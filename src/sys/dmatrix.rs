use super::{get_last_error, DataType};
use crate::bindings::{DMatrixHandle, TreeliteDMatrixCreateFromMat, TreeliteDMatrixFree, TreeliteDMatrixGetDimension};
use crate::errors::TreeRiteError;
use fehler::{throw, throws};
use libc::c_void;
use ndarray::{ArrayView, Ix2};
use num_traits::Float;
use std::ffi::CStr;
use std::ptr::null_mut;
use std::{f32, f64};

#[throws(TreeRiteError)]
pub fn dmatrix_free(handle: DMatrixHandle) {
    let retcode = unsafe { TreeliteDMatrixFree(handle) };
    if retcode != 0 {
        throw!(get_last_error())
    }
}

pub trait FloatInfo {
    const DATA_TYPE: DataType;
    const MISSING: Self;
}

impl FloatInfo for f64 {
    const DATA_TYPE: DataType = DataType::Float64;
    const MISSING: Self = f64::NAN;
}

impl FloatInfo for f32 {
    const DATA_TYPE: DataType = DataType::Float32;
    const MISSING: Self = f32::NAN;
}

#[throws(TreeRiteError)]
pub fn dmatrix_create_from_array<'a, F: Float + FloatInfo>(data: ArrayView<'a, F, Ix2>) -> DMatrixHandle {
    if !data.is_standard_layout() {
        throw!(TreeRiteError::DataNotCContiguous);
    }
    let mut out = null_mut();
    let retcode = unsafe {
        TreeliteDMatrixCreateFromMat(
            data.as_ptr() as *const c_void,
            Into::<&'static CStr>::into(F::DATA_TYPE).as_ptr(),
            data.nrows() as u64,
            data.ncols() as u64,
            &F::MISSING as *const F as *const c_void,
            &mut out,
        )
    };
    if retcode != 0 {
        throw!(get_last_error())
    }
    out
}

#[throws(TreeRiteError)]
pub fn dmatrix_create_from_slice<'a, T: Float + FloatInfo>(data: &'a [T]) -> DMatrixHandle {
    let mut out = null_mut();
    let retcode = unsafe {
        TreeliteDMatrixCreateFromMat(
            data.as_ptr() as *const c_void,
            Into::<&'static CStr>::into(T::DATA_TYPE).as_ptr(),
            1,
            data.len() as u64,
            &T::MISSING as *const T as *const c_void,
            &mut out,
        )
    };
    if retcode != 0 {
        throw!(get_last_error())
    }
    out
}

#[throws(TreeRiteError)]
pub fn dmatrix_get_dimension(handle: DMatrixHandle) -> (u64, u64, u64) {
    let (mut nrow, mut ncol, mut nelem) = (0, 0, 0);

    let retcode = unsafe { TreeliteDMatrixGetDimension(handle, &mut nrow, &mut ncol, &mut nelem) };
    if retcode != 0 {
        throw!(get_last_error())
    }
    (nrow, ncol, nelem)
}
