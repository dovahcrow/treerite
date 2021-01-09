use super::{get_last_error, DataType};
use crate::bindings::{
    DMatrixHandle, PredictorHandle, PredictorOutputHandle, TreeliteCreatePredictorOutputVector, TreeliteDeletePredictorOutputVector, TreelitePredictorFree, TreelitePredictorLoad,
    TreelitePredictorPredictBatch, TreelitePredictorQueryGlobalBias, TreelitePredictorQueryLeafOutputType, TreelitePredictorQueryNumClass, TreelitePredictorQueryNumFeature,
    TreelitePredictorQueryPredTransform, TreelitePredictorQueryResultSize, TreelitePredictorQuerySigmoidAlpha, TreelitePredictorQueryThresholdType,
};
use crate::errors::TreeRiteError;
use fehler::{throw, throws};
use libc::c_int;
use std::convert::TryInto;
use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::{null, null_mut};

#[throws(TreeRiteError)]
pub fn treelite_predictor_load(library_path: &Path, num_worker_thread: usize) -> PredictorHandle {
    let library_path = CString::new(library_path.to_string_lossy().to_owned().to_string())?;
    let mut out = null_mut();

    let retcode = unsafe { TreelitePredictorLoad(library_path.as_ptr(), num_worker_thread as c_int, &mut out) };
    if retcode != 0 {
        throw!(get_last_error())
    }
    out
}

#[throws(TreeRiteError)]
pub fn treelite_predictor_free(handle: PredictorHandle) {
    let retcode = unsafe { TreelitePredictorFree(handle) };
    if retcode != 0 {
        throw!(get_last_error())
    }
}

#[throws(TreeRiteError)]
pub fn treelite_predictor_predict_batch(handle: PredictorHandle, batch: DMatrixHandle, verbose: bool, pred_margin: bool) -> (PredictorOutputHandle, u64) {
    let verbose = if verbose { 1 } else { 0 };
    let pred_margin = if pred_margin { 1 } else { 0 };
    let output_result = treelite_create_predictor_output_vector(handle, batch)?;
    let mut out_result_size = 0;

    let retcode = unsafe { TreelitePredictorPredictBatch(handle, batch, verbose, pred_margin, output_result, &mut out_result_size) };
    if retcode != 0 {
        throw!(get_last_error())
    }
    (output_result, out_result_size)
}

#[throws(TreeRiteError)]
pub fn treelite_predictor_query_leaf_output_type(handle: PredictorHandle) -> DataType {
    let mut out = null();

    let retcode = unsafe { TreelitePredictorQueryLeafOutputType(handle, &mut out) };
    if retcode != 0 {
        throw!(get_last_error())
    }

    let typestr = unsafe { CStr::from_ptr(out) };
    typestr.try_into()?
}

#[throws(TreeRiteError)]
pub fn treelite_predictor_query_num_class(handle: PredictorHandle) -> u64 {
    let mut out = 0;

    let retcode = unsafe { TreelitePredictorQueryNumClass(handle, &mut out) };
    if retcode != 0 {
        throw!(get_last_error())
    }
    out
}

#[throws(TreeRiteError)]
pub fn treelite_predictor_query_result_size(handle: PredictorHandle, batch: DMatrixHandle) -> u64 {
    let mut out = 0;

    let retcode = unsafe { TreelitePredictorQueryResultSize(handle, batch, &mut out) };
    if retcode != 0 {
        throw!(get_last_error())
    }

    out
}

#[throws(TreeRiteError)]
pub fn treelite_predictor_query_num_feature(handle: PredictorHandle) -> u64 {
    let mut out = 0;

    let retcode = unsafe { TreelitePredictorQueryNumFeature(handle, &mut out) };
    if retcode != 0 {
        throw!(get_last_error())
    }

    out
}

#[throws(TreeRiteError)]
pub fn treelite_predictor_query_global_bias(handle: PredictorHandle) -> f32 {
    let mut out = 0.;

    let retcode = unsafe { TreelitePredictorQueryGlobalBias(handle, &mut out) };
    if retcode != 0 {
        throw!(get_last_error())
    }

    out
}

#[throws(TreeRiteError)]
pub fn treelite_predictor_query_sigmoid_alpha(handle: PredictorHandle) -> f32 {
    let mut out = 0.;

    let retcode = unsafe { TreelitePredictorQuerySigmoidAlpha(handle, &mut out) };
    if retcode != 0 {
        throw!(get_last_error())
    }

    out
}

#[throws(TreeRiteError)]
pub fn treelite_predictor_query_pred_transform(handle: PredictorHandle) -> String {
    let mut out = null();

    let retcode = unsafe { TreelitePredictorQueryPredTransform(handle, &mut out) };
    if retcode != 0 {
        throw!(get_last_error())
    }

    unsafe { CStr::from_ptr(out) }.to_string_lossy().to_owned().to_string()
}

#[throws(TreeRiteError)]
pub fn treelite_predictor_query_threshold_type(handle: PredictorHandle) -> String {
    let mut out = null();

    let retcode = unsafe { TreelitePredictorQueryThresholdType(handle, &mut out) };
    if retcode != 0 {
        throw!(get_last_error())
    }

    unsafe { CStr::from_ptr(out) }.to_string_lossy().to_owned().to_string()
}

#[throws(TreeRiteError)]
pub fn treelite_create_predictor_output_vector(handle: PredictorHandle, batch: DMatrixHandle) -> PredictorOutputHandle {
    let mut out = null_mut();

    let retcode = unsafe { TreeliteCreatePredictorOutputVector(handle, batch, &mut out) };
    if retcode != 0 {
        throw!(get_last_error())
    }
    out
}

#[throws(TreeRiteError)]
pub fn treelite_delete_predictor_output_vector(handle: PredictorHandle, output_vector: PredictorOutputHandle) {
    let retcode = unsafe { TreeliteDeletePredictorOutputVector(handle, output_vector) };
    if retcode != 0 {
        throw!(get_last_error())
    }
}
