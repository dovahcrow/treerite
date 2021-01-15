//! This library contains the Rust Binding for Treelite Runtime API.
//!
//! # Simple Example
//!
//! ```
//! use fehler::throws;
//! use ndarray::Array2;
//! use treerite::{DMatrix, Predictor, TreeRiteError};
//!
//! #[throws(TreeRiteError)]
//! fn main() {
//!    let model = Predictor::load("examples/iris.so", 1).unwrap();
//!
//!    let feat: Vec<f64> = vec![7.7, 2.8, 6.7, 2.];
//!
//!    let pred: Array2<f64> = model.predict_batch(&DMatrix::from_slice(&feat)?, false, false)?;
//!    println!("{:?}", pred);
//! }
//! ```
//! The main entry point is the `Predictor` struct and `DMatrix` struct.
//!
//! Take a look at the document on these two structs for more information.
mod dmatrix;
mod errors;
mod predictor;
mod sys;

pub use dmatrix::DMatrix;
pub use errors::TreeRiteError;
pub use predictor::Predictor;
pub use sys::{treelite_register_log_callback, DataType, FloatInfo};
