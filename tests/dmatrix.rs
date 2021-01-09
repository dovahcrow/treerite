use ndarray::array;
use treerite::DMatrix;

#[test]
fn dmatrix_from_slice() {
    let feat = vec![5.4, 3.7, 1.5, 0.2];
    DMatrix::from_slice(&feat).unwrap();
}

#[test]
fn dmatrix_from_2darray() {
    let feat = array![
        [5.5, 4.2, 1.4, 0.2],
        [4.9, 3.1, 1.5, 0.2],
        [5., 3.2, 1.2, 0.2],
        [5.5, 3.5, 1.3, 0.2],
        [4.9, 3.6, 1.4, 0.1],
        [4.4, 3., 1.3, 0.2],
        [5.1, 3.4, 1.5, 0.2],
        [5., 3.5, 1.3, 0.3],
        [4.5, 2.3, 1.3, 0.3],
        [4.4, 3.2, 1.3, 0.2],
        [5., 3.5, 1.6, 0.6],
        [5.1, 3.8, 1.9, 0.4],
        [4.8, 3., 1.4, 0.3],
        [5.1, 3.8, 1.6, 0.2],
        [4.6, 3.2, 1.4, 0.2],
        [5.3, 3.7, 1.5, 0.2],
        [5., 3.3, 1.4, 0.2],
        [7., 3.2, 4.7, 1.4],
        [6.4, 3.2, 4.5, 1.5],
        [6.9, 3.1, 4.9, 1.5],
        [5.5, 2.3, 4., 1.3],
        [6.5, 2.8, 4.6, 1.5]
    ];

    DMatrix::from_2darray(&feat).unwrap();
}

// #[test]
// fn dmatrix_from_libsvm_file() {
//     DMatrix::<f32>::from_file("glass.scale", Some("libsvm".into()), 1, false).unwrap();
// }
