use crate::bindings::PredictorHandle;
use crate::dmatrix::DMatrix;
use crate::errors::TreeRiteError;
use crate::sys::{
    delete_predictor_output_vector, predictor_free, predictor_load, predictor_predict_batch, predictor_query_leaf_output_type, predictor_query_num_class,
    predictor_query_result_size, DataType, FloatInfo,
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
        let handle = predictor_load(library_path.as_ref(), num_worker_thread)?;
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
        let (output, size) = predictor_predict_batch(self.handle, dmatrix.handle, verbose, pred_margin)?;
        if dtype != O::DATA_TYPE {
            throw!(TreeRiteError::WrongPredictOutputType(O::DATA_TYPE))
        }
        let data = unsafe { from_raw_parts(output as *mut O, size as usize) };
        let ret = Array::from_shape_vec((dmatrix.nrows()? as usize, nclasses as usize), data.to_vec())?;

        delete_predictor_output_vector(self.handle, output)?;

        ret
    }

    #[throws(TreeRiteError)]
    pub fn leaf_output_type(&self) -> DataType {
        predictor_query_leaf_output_type(self.handle)?
    }

    #[throws(TreeRiteError)]
    pub fn num_class(&self) -> u64 {
        predictor_query_num_class(self.handle)?
    }
    #[throws(TreeRiteError)]
    pub fn result_size<F>(&self, batch: &DMatrix<F>) -> u64 {
        predictor_query_result_size(self.handle, batch.handle)?
    }
}

impl Drop for Predictor {
    fn drop(&mut self) {
        match predictor_free(self.handle) {
            Ok(()) => {}
            Err(e) => {
                if cfg!(feature = "free_panic") {
                    panic!("cannot free predictor: {}", e)
                }
            }
        }
    }
}
