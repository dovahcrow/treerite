use std::ffi::NulError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TreeRiteError {
    #[error("Error: {0}")]
    CError(String),

    #[error("Unknown Data Type String: {0}")]
    UnknownDataTypeString(String),

    #[error(transparent)]
    NullError(#[from] NulError),

    #[error(transparent)]
    ShapeError(#[from] ndarray::ShapeError),

    #[error("Data is not C contiguous")]
    DataNotCContiguous,
}
