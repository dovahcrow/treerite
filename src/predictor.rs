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

/// Loader for compiled shared libraries.
///
/// There should be at most only one thread calling `predict_batch` at the same time.
/// As the result, Predictor is Send but not Sync.
pub struct Predictor {
    handle: PredictorHandle,
}

// predict_batch is not thread safe
// unsafe impl Sync for Predictor {}
unsafe impl Send for Predictor {}

impl Predictor {
    /// Load the compiled shared libraries from a filesystem location.
    ///
    /// # Parameters
    ///
    /// - library_path: location of dynamic shared library (.dll/.so/.dylib)
    ///
    /// - num_worker_thread: number of worker threads to use; if unspecified, use maximum number of hardware threads.
    ///
    /// # Example
    /// ```
    /// use treerite::Predictor;
    /// let model = Predictor::load("examples/iris.so", 1).unwrap();
    /// ```
    #[throws(TreeRiteError)]
    pub fn load<P>(library_path: P, num_worker_thread: usize) -> Predictor
    where
        P: AsRef<Path>,
    {
        let handle = treelite_predictor_load(library_path.as_ref(), num_worker_thread)?;
        Predictor { handle }
    }

    /// Perform batch prediction with a 2D sparse data matrix.
    /// Worker threads will internally divide up work for batch prediction.
    ///
    /// # Example
    /// ```
    /// use treerite::{Predictor, DMatrix};
    /// use ndarray::Array2;
    /// let model = Predictor::load("examples/iris.so", 1).unwrap();
    /// let feat: Vec<f64> = vec![7.7, 2.8, 6.7, 2.];
    /// let pred: Array2<f64> = model.predict_batch(&DMatrix::from_slice(&feat).unwrap(), false, false).unwrap();
    /// ```
    ///
    /// # Note
    /// The output array should be explicitly typed, i.e. f64 or f32.
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

    /// Get the output type of the leaves.
    #[throws(TreeRiteError)]
    pub fn leaf_output_type(&self) -> DataType {
        treelite_predictor_query_leaf_output_type(self.handle)?
    }

    /// Get how many classes this model output
    #[throws(TreeRiteError)]
    pub fn num_class(&self) -> u64 {
        treelite_predictor_query_num_class(self.handle)?
    }

    /// Get the output result size (num instances x num classes) given a DMatrix
    #[throws(TreeRiteError)]
    pub fn result_size<F>(&self, batch: &DMatrix<F>) -> u64 {
        treelite_predictor_query_result_size(self.handle, batch.handle)?
    }

    /// Get the number of features this model required
    #[throws(TreeRiteError)]
    pub fn num_feature(&self) -> u64 {
        treelite_predictor_query_num_feature(self.handle)?
    }

    /// Get the global bias of the model
    #[throws(TreeRiteError)]
    pub fn global_bias(&self) -> f32 {
        treelite_predictor_query_global_bias(self.handle)?
    }

    /// Get the sigmoid alpha of the model
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
