use super::bindings::size_t;
use super::bindings::{
    DMatrixHandle, TreeliteDMatrixCreateFromCSR, TreeliteDMatrixCreateFromMat, TreeliteDMatrixFree,
    TreeliteDMatrixGetDimension,
};
use super::{DataType, RetCodeCheck};
use crate::errors::TreeRiteError;
use fehler::{throw, throws};
use libc::c_void;
use ndarray::{ArrayView, Ix2};
use num_traits::Float;
use std::ffi::CStr;
use std::ptr::null_mut;
use std::{f32, f64};

#[throws(TreeRiteError)]
pub fn treelite_dmatrix_free(handle: DMatrixHandle) {
    unsafe { TreeliteDMatrixFree(handle) }.check()?;
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
pub fn treelite_dmatrix_create_from_array<'a, F: Float + FloatInfo>(
    data: ArrayView<'a, F, Ix2>,
) -> DMatrixHandle {
    if !data.is_standard_layout() {
        throw!(TreeRiteError::DataNotCContiguous);
    }
    let mut out = null_mut();
    unsafe {
        TreeliteDMatrixCreateFromMat(
            data.as_ptr() as *const c_void,
            Into::<&'static CStr>::into(F::DATA_TYPE).as_ptr(),
            data.nrows() as u64 as size_t,
            data.ncols() as u64 as size_t,
            &F::MISSING as *const F as *const c_void,
            &mut out,
        )
    }
    .check()?;
    out
}

#[throws(TreeRiteError)]
pub fn treelite_dmatrix_create_from_slice<'a, T: Float + FloatInfo>(
    data: &'a [T],
) -> DMatrixHandle {
    let mut out = null_mut();
    unsafe {
        TreeliteDMatrixCreateFromMat(
            data.as_ptr() as *const c_void,
            Into::<&'static CStr>::into(T::DATA_TYPE).as_ptr(),
            1,
            data.len() as u64 as size_t,
            &T::MISSING as *const T as *const c_void,
            &mut out,
        )
    }
    .check()?;
    out
}

#[throws(TreeRiteError)]
pub fn treelite_dmatrix_create_from_csr_format<'a, T: Float + FloatInfo>(
    headers: &'a [u64],
    indices: &'a [u32],
    data: &'a [T],
    num_row: u64,
    num_col: u64,
) -> DMatrixHandle {
    let mut out = null_mut();
    unsafe {
        TreeliteDMatrixCreateFromCSR(
            data.as_ptr() as *const c_void,
            Into::<&'static CStr>::into(T::DATA_TYPE).as_ptr(),
            indices.as_ptr() as *const u32,
            headers.as_ptr() as *const size_t,
            num_row as size_t,
            num_col as size_t,
            &mut out,
        )
    }
    .check()?;
    out
}

// #[throws(TreeRiteError)]
// pub fn treelite_dmatrix_create_from_file(path: &Path, format: Option<String>, data_type: DataType, nthread: usize, verbose: bool) -> DMatrixHandle {
//     let format = format.unwrap_or_else(|| "libsvm".to_string());
//     let format = CString::new(format)?;
//     let path = CString::new(path.to_string_lossy().to_owned().to_string())?;
//     let verbose = if verbose { 1 } else { 0 };
//     let mut out = null_mut();
//     let retcode = unsafe {
//         TreeliteDMatrixCreateFromFile(
//             path.as_ptr(),
//             format.as_ptr(),
//             Into::<&'static CStr>::into(data_type).as_ptr(),
//             nthread as i32,
//             verbose,
//             &mut out,
//         )
//     };
//     if retcode != 0 {
//         throw!(get_last_error())
//     }
//     out
// }

#[throws(TreeRiteError)]
pub fn treelite_dmatrix_get_dimension(handle: DMatrixHandle) -> (u64, u64, u64) {
    let (mut nrow, mut ncol, mut nelem) = (0, 0, 0);

    unsafe { TreeliteDMatrixGetDimension(handle, &mut nrow, &mut ncol, &mut nelem) }.check()?;

    (nrow as u64, ncol as u64, nelem as u64)
}
