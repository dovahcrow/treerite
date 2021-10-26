use ndarray::{array, Array2};
use treerite::{DMatrix, DataType, Predictor};

#[test]
fn load_model() {
    Predictor::load("examples/iris.so", 1).unwrap();
}

#[test]
fn query_num_class() {
    let model = Predictor::load("examples/iris.so", 1).unwrap();
    assert_eq!(model.num_class().unwrap(), 3);
}

#[test]
fn query_result_size() {
    let model = Predictor::load("examples/iris.so", 1).unwrap();
    let feats = array![
        [5.5, 4.2, 1.4, 0.2],
        [4.9, 3.1, 1.5, 0.2],
        [5., 3.2, 1.2, 0.2],
        [5.3, 3.7, 1.5, 0.2],
        [5., 3.3, 1.4, 0.2],
        [5.5, 2.3, 4., 1.3],
        [6.5, 2.8, 4.6, 1.5]
    ];
    assert_eq!(
        model
            .result_size(&DMatrix::from_2darray(&feats).unwrap())
            .unwrap(),
        feats.nrows() as u64 * model.num_class().unwrap()
    );
}

#[test]
fn query_num_features() {
    let model = Predictor::load("examples/iris.so", 1).unwrap();
    assert_eq!(model.num_feature().unwrap(), 4);
}

#[test]
fn query_global_bias() {
    let model = Predictor::load("examples/iris.so", 1).unwrap();
    assert_eq!(model.global_bias().unwrap(), 0.);
}

#[test]
fn query_sigmod_alpha() {
    let model = Predictor::load("examples/iris.so", 1).unwrap();
    assert_eq!(model.sigmod_alpha().unwrap(), 1.);
}

#[test]
fn query_pred_transform() {
    let model = Predictor::load("examples/iris.so", 1).unwrap();
    assert_eq!(model.pred_transform().unwrap(), "softmax");
}

#[test]
fn query_threshold_type() {
    let model = Predictor::load("examples/iris.so", 1).unwrap();
    assert_eq!(model.threshold_type().unwrap(), "float64");
}

#[test]
fn query_leaf_output_type() {
    let model = Predictor::load("examples/iris.so", 1).unwrap();
    assert_eq!(model.leaf_output_type().unwrap(), DataType::Float64);
}

#[test]
fn single_row_predict() {
    let model = Predictor::load("examples/iris.so", 1).unwrap();

    let feat = vec![5.4, 3.7, 1.5, 0.2];

    let pred: Array2<f64> = model
        .predict_batch(&DMatrix::from_slice(&feat).unwrap(), false, false)
        .unwrap();

    assert_eq!(
        pred,
        array![[0.7979739007026813, 0.1002914545900944, 0.10173464470722438]]
    );
}

#[test]
fn multi_row_predict() {
    let model = Predictor::load("examples/iris.so", 1).unwrap();

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

    let pred: Array2<f64> = model
        .predict_batch(&DMatrix::from_2darray(&feat).unwrap(), false, false)
        .unwrap();

    assert_eq!(
        pred,
        array![
            [0.7970173602626079, 0.1013699455121738, 0.1016126942252183],
            [0.7855779193763847, 0.10426475445126739, 0.11015732617234784],
            [0.7884355845836272, 0.10589626990554922, 0.10566814551082357],
            [0.7970173602626079, 0.1013699455121738, 0.1016126942252183],
            [0.798662024938987, 0.09827206006221767, 0.10306591499879524],
            [0.78459897754481, 0.1053809679805116, 0.11002005447467829],
            [0.7979739007026813, 0.1002914545900944, 0.10173464470722438],
            [0.8023194177997669, 0.09872208713207009, 0.09895849506816293],
            [0.783361935314134, 0.10521481850615293, 0.11142324617971305],
            [0.78459897754481, 0.1053809679805116, 0.11002005447467829],
            [0.6923711611763775, 0.22223141901979301, 0.08539741980382948],
            [0.6807851318681493, 0.2324207593877195, 0.08679410874413115],
            [0.78459897754481, 0.1053809679805116, 0.11002005447467829],
            [0.7979739007026813, 0.1002914545900944, 0.10173464470722433],
            [0.78459897754481, 0.1053809679805116, 0.11002005447467829],
            [0.7979739007026813, 0.1002914545900944, 0.10173464470722438],
            [0.7880951088248122, 0.10585053996443955, 0.10605435121074837],
            [0.11503313620087713, 0.7290118961885667, 0.15595496761055624],
            [0.12005764679011498, 0.737553111151496, 0.142389242058389],
            [0.10559415310594689, 0.5434573428426337, 0.35094850405141953],
            [0.10949968249280011, 0.7796727054679523, 0.11082761203924764],
            [0.11893611835237156, 0.7306631977589019, 0.15040068388872643]
        ]
    );
}
