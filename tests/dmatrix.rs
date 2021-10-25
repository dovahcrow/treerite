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

#[test]
fn dmatrix_from_csr_format(){
    // this sample is from https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)
    let data = vec![10f32, 20f32, 30f32, 40f32, 50f32, 60f32, 70f32, 80f32];
    let indices = vec![0, 1, 1, 3, 2, 3, 4, 5];
    let headers = vec![0, 2, 4, 7, 8];
    let num_row = 3;
    let num_col = 6;
    DMatrix::from_csr_format(&headers, &indices, &data, num_row, num_col).unwrap();
}

// #[test]
// fn dmatrix_from_libsvm_file() {
//     DMatrix::<f32>::from_file("glass.scale", Some("libsvm".into()), 1, false).unwrap();
// }
