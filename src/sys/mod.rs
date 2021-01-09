mod dmatrix;
mod predictor;

pub use self::dmatrix::{dmatrix_create_from_array, dmatrix_create_from_slice, dmatrix_free, dmatrix_get_dimension, FloatInfo};
pub use self::predictor::{
    create_predictor_output_vector, delete_predictor_output_vector, predictor_free, predictor_load, predictor_predict_batch, predictor_query_leaf_output_type,
    predictor_query_num_class, predictor_query_result_size,
};
use crate::bindings::TreeliteGetLastError;
use crate::errors::TreeRiteError;
use fehler::throw;
use std::convert::TryInto;
use std::ffi::CStr;
// https://stackoverflow.com/questions/58349489/const-static-cstr
#[allow(unconditional_panic)]
const fn illegal_null_in_string() {
    [][0]
}

#[doc(hidden)]
pub const fn validate_cstr_contents(bytes: &[u8]) {
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\0' {
            illegal_null_in_string();
        }
        i += 1;
    }
}

macro_rules! cstr {
    ( $s:literal ) => {{
        $crate::sys::validate_cstr_contents($s.as_bytes());
        unsafe { std::mem::transmute::<_, &std::ffi::CStr>(concat!($s, "\0")) }
    }};
}

const DTYPE_FLOAT32: &std::ffi::CStr = cstr!("float32");
const DTYPE_FLOAT64: &std::ffi::CStr = cstr!("float64");
const DTYPE_UINT32: &std::ffi::CStr = cstr!("uint32");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataType {
    Float32,
    Float64,
    UInt32,
}

impl DataType {
    pub fn as_display(&self) -> String {
        match self {
            DataType::Float32 => "f32".to_string(),
            DataType::Float64 => "f64".to_string(),
            DataType::UInt32 => "u32".to_string(),
        }
    }
}

impl Into<&'static CStr> for DataType {
    fn into(self) -> &'static CStr {
        match self {
            DataType::Float32 => DTYPE_FLOAT32,
            DataType::Float64 => DTYPE_FLOAT64,
            DataType::UInt32 => DTYPE_UINT32,
        }
    }
}

impl TryInto<DataType> for &'static CStr {
    type Error = TreeRiteError;

    fn try_into(self) -> Result<DataType, Self::Error> {
        if self == DTYPE_FLOAT32 {
            Ok(DataType::Float32)
        } else if self == DTYPE_FLOAT64 {
            Ok(DataType::Float64)
        } else if self == DTYPE_UINT32 {
            Ok(DataType::UInt32)
        } else {
            throw!(TreeRiteError::UnknownDataTypeString(self.to_string_lossy().to_owned().to_string()))
        }
    }
}

pub fn get_last_error() -> TreeRiteError {
    let cs = unsafe { CStr::from_ptr(TreeliteGetLastError()) };

    TreeRiteError::CError(cs.to_string_lossy().to_owned().to_string())
}
