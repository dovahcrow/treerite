/*!
 * This is the rust version of treelite/tests/example_app/example.c
 * Copyright (c) 2023 by Contributors
 * \file example.c
 * \brief Test using Treelite as a C++ library
 */

use std::cmp::Ordering;

use ndarray::array;
use treerite::{Builder, GTIConfig, Model};

fn build_model() -> Model {
    let model_metadata = r#"{
              "threshold_type": "float32",
              "leaf_output_type": "float32",
              "metadata": {
                "num_feature": 2,
                "task_type": "kRegressor",
                "average_tree_output": false,
                "num_target": 1,
                "num_class": [1],
                "leaf_vector_shape": [1, 1]
              },
              "tree_annotation": {
                "num_tree": 1,
                "target_id": [0],
                "class_id": [0]
              },
              "postprocessor": {
                "name": "identity"
              },
              "base_scores": [0.0]
            }"#;
    let mut builder = Builder::new(model_metadata).unwrap();
    builder.start_tree().unwrap();
    builder.start_node(0).unwrap();
    builder
        .numerical_test(0, 0.0, 0, Ordering::Less, 1, 2)
        .unwrap();
    builder.end_node().unwrap();
    builder.start_node(1).unwrap();
    builder.leaf_scalar(-1.0).unwrap();
    builder.end_node().unwrap();
    builder.start_node(2).unwrap();
    builder.leaf_scalar(1.0).unwrap();
    builder.end_node().unwrap();
    builder.end_tree().unwrap();

    let model = builder.commit_model().unwrap();
    model
}

fn main() {
    let model = build_model();
    let input = array![
        [-2.0f64, 0.0f64],
        [-1.0f64, 0.0f64],
        [0.0f64, 0.0f64],
        [1.0f64, 0.0f64],
        [2.0f64, 0.0f64]
    ];

    let config = GTIConfig::parse(
        r#"{
          "predict_type": "default",
          "nthread": 2
        }"#,
    )
    .unwrap();

    let y = model.predict(&config, &input).unwrap();

    println!("{:?}", y);
}
