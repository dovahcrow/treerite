use fehler::throws;
use ndarray::Array2;
use treerite::{DMatrix, Predictor, TreeRiteError};

#[throws(TreeRiteError)]
fn main() {
    let model = Predictor::load("examples/iris.so", 1).unwrap();

    let feat: Vec<f64> = vec![7.7, 2.8, 6.7, 2.];

    let pred: Array2<f64> = model.predict_batch(&DMatrix::from_slice(&feat)?, false, false)?;
    println!("{:?}", pred);
}
