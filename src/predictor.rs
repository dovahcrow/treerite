use crate::dmatrix::DMatrix;
use crate::errors::TreeRiteError;
use crate::sys::{
    treelite_delete_predictor_output_vector, treelite_predictor_free, treelite_predictor_load, treelite_predictor_predict_batch, treelite_predictor_query_global_bias,
    treelite_predictor_query_leaf_output_type, treelite_predictor_query_num_class, treelite_predictor_query_num_feature, treelite_predictor_query_pred_transform,
    treelite_predictor_query_result_size, treelite_predictor_query_sigmoid_alpha, treelite_predictor_query_threshold_type, DataType, FloatInfo, PredictorHandle,
};
use fehler::{throw, throws};
use ndarray::{Array, Array2};
use num_traits::Float;
use std::path::Path;
use std::slice::from_raw_parts;

pub struct Predictor {
    handle: PredictorHandle,
}
// predict_batch is not thread safe
// unsafe impl Sync for Predictor {}
unsafe impl Send for Predictor {}

impl Predictor {
    #[throws(TreeRiteError)]
    pub fn load<P>(library_path: P, num_worker_thread: usize) -> Predictor
    where
        P: AsRef<Path>,
    {
        let handle = treelite_predictor_load(library_path.as_ref(), num_worker_thread)?;
        Predictor { handle }
    }

    #[throws(TreeRiteError)]
    pub fn predict_batch<F, O>(&self, dmatrix: &DMatrix<F>, verbose: bool, pred_margin: bool) -> Array2<O>
    where
        O: Float + FloatInfo,
        F: Float + FloatInfo,
    {
        let dtype = self.leaf_output_type()?;
        let nclasses = self.num_class()?;
        let (output, size) = treelite_predictor_predict_batch(self.handle, dmatrix.handle, verbose, pred_margin)?;
        if dtype != O::DATA_TYPE {
            throw!(TreeRiteError::WrongPredictOutputType(O::DATA_TYPE))
        }
        let data = unsafe { from_raw_parts(output as *mut O, size as usize) };
        let ret = Array::from_shape_vec((dmatrix.nrows()? as usize, nclasses as usize), data.to_vec())?;

        treelite_delete_predictor_output_vector(self.handle, output)?;

        ret
    }

    #[throws(TreeRiteError)]
    pub fn leaf_output_type(&self) -> DataType {
        treelite_predictor_query_leaf_output_type(self.handle)?
    }

    #[throws(TreeRiteError)]
    pub fn num_class(&self) -> u64 {
        treelite_predictor_query_num_class(self.handle)?
    }

    #[throws(TreeRiteError)]
    pub fn result_size<F>(&self, batch: &DMatrix<F>) -> u64 {
        treelite_predictor_query_result_size(self.handle, batch.handle)?
    }

    #[throws(TreeRiteError)]
    pub fn num_feature(&self) -> u64 {
        treelite_predictor_query_num_feature(self.handle)?
    }

    #[throws(TreeRiteError)]
    pub fn global_bias(&self) -> f32 {
        treelite_predictor_query_global_bias(self.handle)?
    }

    #[throws(TreeRiteError)]
    pub fn sigmod_alpha(&self) -> f32 {
        treelite_predictor_query_sigmoid_alpha(self.handle)?
    }

    #[throws(TreeRiteError)]
    pub fn pred_transform(&self) -> String {
        treelite_predictor_query_pred_transform(self.handle)?
    }

    #[throws(TreeRiteError)]
    pub fn threshold_type(&self) -> String {
        treelite_predictor_query_threshold_type(self.handle)?
    }
}

impl Drop for Predictor {
    fn drop(&mut self) {
        match treelite_predictor_free(self.handle) {
            Ok(()) => {}
            Err(e) => {
                if cfg!(feature = "free_panic") {
                    panic!("cannot free predictor: {}", e)
                }
            }
        }
    }
}
