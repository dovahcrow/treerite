//! This library contains the Rust Binding for Treelite Runtime API.
//!
//! Take a look at the tests to know details.

mod errors;
#[allow(
    dead_code,
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case
)]
mod sys;

pub use crate::errors::TreeRiteError;

use std::cmp::Ordering;
use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::{null, null_mut};
use std::slice;

use culpa::{throw, throws};
use libc::{c_int, c_void};

use ndarray::{Array, ArrayD, ArrayView, Dimension, IxDyn};
use num_traits::Zero;

#[repr(transparent)]
pub struct Model(sys::TreeliteModelBuilderHandle);

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    #[throws(TreeRiteError)]
    pub fn load_lightgbm<P>(filename: P, config: &str) -> Self
    where
        P: AsRef<Path>,
    {
        let filename = CString::new(filename.as_ref().to_string_lossy().into_owned())?;
        let config = CString::new(config)?;
        let mut out = null_mut();
        unsafe {
            sys::TreeliteLoadLightGBMModel(filename.as_ptr(), config.as_ptr(), &mut out).check()?
        };

        Self(out)
    }

    #[throws(TreeRiteError)]
    pub fn load_lightgbm_string(data: &str, config: &str) -> Self {
        let data = CString::new(data)?;
        let config = CString::new(config)?;
        let mut out = null_mut();
        unsafe {
            sys::TreeliteLoadLightGBMModelFromString(data.as_ptr(), config.as_ptr(), &mut out)
                .check()?
        };

        Self(out)
    }

    #[throws(TreeRiteError)]
    pub fn load_xgboost_model_legacy_binary<P>(filename: P, config: &str) -> Self
    where
        P: AsRef<Path>,
    {
        let filename = CString::new(filename.as_ref().to_string_lossy().into_owned())?;
        let config = CString::new(config)?;
        let mut out = null_mut();
        unsafe {
            sys::TreeliteLoadXGBoostModelLegacyBinary(filename.as_ptr(), config.as_ptr(), &mut out)
                .check()?
        };

        Self(out)
    }

    #[throws(TreeRiteError)]
    pub fn load_xgboost_model_legacy_binary_buffer(buf: &[u8], config: &str) -> Self {
        let config = CString::new(config)?;
        let mut out = null_mut();
        unsafe {
            sys::TreeliteLoadXGBoostModelLegacyBinaryFromMemoryBuffer(
                buf.as_ptr() as *const c_void,
                buf.len(),
                config.as_ptr(),
                &mut out,
            )
            .check()?
        };

        Self(out)
    }

    #[throws(TreeRiteError)]
    pub fn load_xgboost_model_json<P>(filename: P, config: &str) -> Self
    where
        P: AsRef<Path>,
    {
        let filename = CString::new(filename.as_ref().to_string_lossy().into_owned())?;
        let config = CString::new(config)?;
        let mut out = null_mut();
        unsafe {
            sys::TreeliteLoadXGBoostModelJSON(filename.as_ptr(), config.as_ptr(), &mut out)
                .check()?
        };

        Self(out)
    }

    #[throws(TreeRiteError)]
    pub fn load_xgboost_model_from_json_string(json: &str, config: &str) -> Self {
        let len = json.len();
        let json = CString::new(json)?;
        let config = CString::new(config)?;
        let mut out = null_mut();
        unsafe {
            sys::TreeliteLoadXGBoostModelFromJSONString(
                json.as_ptr(),
                len,
                config.as_ptr(),
                &mut out,
            )
            .check()?
        };

        Self(out)
    }

    #[throws(TreeRiteError)]
    pub fn load_xgboost_model_from_ubjson<P>(filename: P, config: &str) -> Self
    where
        P: AsRef<Path>,
    {
        let filename = CString::new(filename.as_ref().to_string_lossy().into_owned())?;
        let config = CString::new(config)?;
        let mut out = null_mut();
        unsafe {
            sys::TreeliteLoadXGBoostModelUBJSON(filename.as_ptr(), config.as_ptr(), &mut out)
                .check()?
        };

        Self(out)
    }

    #[throws(TreeRiteError)]
    pub fn load_xgboost_model_from_ubjson_string(ubjson: &str, config: &str) -> Self {
        let len = ubjson.len();
        let ubjson = CString::new(ubjson)?;
        let config = CString::new(config)?;
        let mut out = null_mut();
        unsafe {
            sys::TreeliteLoadXGBoostModelFromUBJSONString(
                ubjson.as_ptr() as *const u8,
                len,
                config.as_ptr(),
                &mut out,
            )
            .check()?
        };

        Self(out)
    }

    #[throws(TreeRiteError)]
    pub fn get_output_shape(&self, config: &GTIConfig, num_rows: usize) -> Vec<usize> {
        let mut output_shape: *const u64 = null();
        let mut output_ndim: u64 = 0;
        unsafe {
            sys::TreeliteGTILGetOutputShape(
                self.0,
                num_rows as u64,
                config.0,
                &mut output_shape,
                &mut output_ndim,
            )
            .check()?
        };

        let ret = unsafe { slice::from_raw_parts(output_shape, output_ndim as usize) };

        ret.into_iter().map(|e| *e as usize).collect()
    }

    #[throws(TreeRiteError)]
    pub fn predict<A, D>(&self, config: &GTIConfig, input: ArrayView<A, D>) -> ArrayD<A>
    where
        A: TypeName + Clone + Zero,
        D: Dimension,
    {
        let nrows = input.shape()[0];
        let dims = self.get_output_shape(config, nrows)?;
        let mut output = Array::zeros(IxDyn(&dims));
        unsafe {
            sys::TreeliteGTILPredict(
                self.0,
                input.as_ptr() as *const c_void,
                A::type_name().as_ptr(),
                nrows as u64,
                output.as_mut_ptr() as *mut c_void,
                config.0,
            )
            .check()?
        };

        output
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // TODO: how can we throw an error here?
        unsafe {
            sys::TreeliteFreeModel(self.0)
                .check()
                .expect("Destructing Model fail")
        };
    }
}

#[repr(transparent)]
pub struct GTIConfig(sys::TreeliteGTILConfigHandle);

unsafe impl Send for GTIConfig {}
unsafe impl Sync for GTIConfig {}

impl GTIConfig {
    #[throws(TreeRiteError)]
    pub fn parse(config: &str) -> Self {
        let mut out = null_mut();
        let config = CString::new(config)?;
        unsafe { sys::TreeliteGTILParseConfig(config.as_ptr(), &mut out).check()? };
        Self(out)
    }
}

impl Drop for GTIConfig {
    fn drop(&mut self) {
        // TODO: how can we throw an error here?
        unsafe {
            sys::TreeliteGTILDeleteConfig(self.0)
                .check()
                .expect("Destructing GTIConfig fail")
        };
    }
}

#[repr(transparent)]
pub struct Builder(sys::TreeliteModelBuilderHandle);

unsafe impl Send for Builder {}
unsafe impl Sync for Builder {}

impl Builder {
    #[throws(TreeRiteError)]
    pub fn new(metadata: &str) -> Self {
        let mut inner = null_mut();
        let metadata = CString::new(metadata)?;
        unsafe {
            sys::TreeliteGetModelBuilder(metadata.as_ptr(), &mut inner).check()?;
        }
        Self(inner)
    }

    #[throws(TreeRiteError)]
    pub fn start_tree(&mut self) {
        unsafe {
            sys::TreeliteModelBuilderStartTree(self.0).check()?;
        }
    }

    #[throws(TreeRiteError)]
    pub fn end_tree(&mut self) {
        unsafe {
            sys::TreeliteModelBuilderEndTree(self.0).check()?;
        }
    }

    #[throws(TreeRiteError)]
    pub fn start_node(&mut self, node_key: i32) {
        unsafe {
            sys::TreeliteModelBuilderStartNode(self.0, node_key).check()?;
        }
    }

    #[throws(TreeRiteError)]
    pub fn end_node(&mut self) {
        unsafe {
            sys::TreeliteModelBuilderEndNode(self.0).check()?;
        }
    }

    #[throws(TreeRiteError)]
    pub fn numerical_test(
        &mut self,
        split_index: i32,
        threshold: f64,
        default_left: i32,
        cmp: Ordering,
        left_child_key: i32,
        right_child_key: i32,
    ) {
        const LE: &'static CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"<\0") };
        const EQ: &'static CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"=\0") };
        const GE: &'static CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b">\0") };

        unsafe {
            sys::TreeliteModelBuilderNumericalTest(
                self.0,
                split_index,
                threshold,
                default_left,
                match cmp {
                    Ordering::Less => LE,
                    Ordering::Equal => EQ,
                    Ordering::Greater => GE,
                }
                .as_ptr(),
                left_child_key,
                right_child_key,
            )
            .check()?;
        }
    }

    #[throws(TreeRiteError)]
    pub fn leaf_scalar(&mut self, leaf_value: f64) {
        unsafe {
            sys::TreeliteModelBuilderLeafScalar(self.0, leaf_value).check()?;
        }
    }

    #[throws(TreeRiteError)]
    pub fn commit_model(&self) -> Model {
        let mut inner = null_mut();
        unsafe {
            sys::TreeliteModelBuilderCommitModel(self.0, &mut inner).check()?;
        }
        Model(inner)
    }
}

impl Drop for Builder {
    fn drop(&mut self) {
        // TODO: how can we throw an error here?
        unsafe {
            sys::TreeliteDeleteModelBuilder(self.0)
                .check()
                .expect("Destructing Builder fail")
        };
    }
}

// // https://stackoverflow.com/questions/58349489/const-static-cstr
// #[allow(unconditional_panic)]
// const fn illegal_null_in_string() {
//     [][0]
// }

// #[doc(hidden)]
// pub const fn validate_cstr_contents(bytes: &[u8]) {
//     let mut i = 0;
//     while i < bytes.len() {
//         if bytes[i] == b'\0' {
//             illegal_null_in_string();
//         }
//         i += 1;
//     }
// }

// macro_rules! cstr {
//     ( $s:literal ) => {{
//         $crate::sys::validate_cstr_contents($s.as_bytes());
//         unsafe { std::mem::transmute::<_, &std::ffi::CStr>(concat!($s, "\0")) }
//     }};
// }

// const DTYPE_FLOAT32: &std::ffi::CStr = cstr!("float32");
// const DTYPE_FLOAT64: &std::ffi::CStr = cstr!("float64");
// const DTYPE_UINT32: &std::ffi::CStr = cstr!("uint32");

// #[derive(Clone, Copy, Debug, PartialEq, Eq)]
// pub enum DataType {
//     Float32,
//     Float64,
//     UInt32,
// }

// impl DataType {
//     pub fn as_display(&self) -> String {
//         match self {
//             DataType::Float32 => "f32".to_string(),
//             DataType::Float64 => "f64".to_string(),
//             DataType::UInt32 => "u32".to_string(),
//         }
//     }
// }

// impl Into<&'static CStr> for DataType {
//     fn into(self) -> &'static CStr {
//         match self {
//             DataType::Float32 => DTYPE_FLOAT32,
//             DataType::Float64 => DTYPE_FLOAT64,
//             DataType::UInt32 => DTYPE_UINT32,
//         }
//     }
// }

// impl TryInto<DataType> for &'static CStr {
//     type Error = TreeRiteError;

//     fn try_into(self) -> Result<DataType, Self::Error> {
//         if self == DTYPE_FLOAT32 {
//             Ok(DataType::Float32)
//         } else if self == DTYPE_FLOAT64 {
//             Ok(DataType::Float64)
//         } else if self == DTYPE_UINT32 {
//             Ok(DataType::UInt32)
//         } else {
//             throw!(TreeRiteError::UnknownDataTypeString(
//                 self.to_string_lossy().to_owned().to_string()
//             ))
//         }
//     }
// }

trait RetCodeCheck {
    fn check(&self) -> Result<(), TreeRiteError>;
}

impl RetCodeCheck for c_int {
    #[throws(TreeRiteError)]
    fn check(&self) {
        if *self != 0 {
            throw!(get_last_error())
        }
    }
}

pub fn get_last_error() -> TreeRiteError {
    let cs = unsafe { CStr::from_ptr(sys::TreeliteGetLastError()) };

    TreeRiteError::CError(cs.to_string_lossy().into_owned())
}

pub trait TypeName {
    fn type_name() -> &'static CStr;
}

impl TypeName for f32 {
    fn type_name() -> &'static CStr {
        CStr::from_bytes_until_nul(b"float32\0").unwrap()
    }
}

impl TypeName for f64 {
    fn type_name() -> &'static CStr {
        CStr::from_bytes_until_nul(b"float64\0").unwrap()
    }
}

// #[throws(TreeRiteError)]
// pub fn treelite_register_log_callback(func: extern "C" fn(*const ::std::os::raw::c_char)) {
//     unsafe { TreeliteRegisterLogCallback(Some(func)) }.check()?;
// }
