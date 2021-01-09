# TreeRite: [TreeLite](https://github.com/dmlc/treelite) runtime in Rust ![CI](https://github.com/dovahcrow/treerite/workflows/treerite%20CI/badge.svg)

This binding currently works for treelite 1.0.0rc1

# Usage

* Make sure your machine has cmake, libstdc++ and libgomp installed.
* Add this to your project's `Cargo.toml`.
  ```toml
  treerite = { git = "https://github.com/dovahcrow/treerite" }
  ```

By default, the treerite library is static linked to your binary. If you'd like to use the dynamic lib,
set the dynamic feature of `treerite`.

# Documentation

There's no documentation available yet. But you can take a look at the example folder and the tests folder. 
The usage should be quite straight forward: you first load the `Predictor` from a shared library of the model.
Then, load the data from `Vec` or `ndarray::Array2` into `DMatrix`. Finally, you do prediction using
`Predictor::predict_batch(dmatrix)`.

# TODO

- [ ] TreeliteDMatrixCreateFromCSR
- [ ] TreeliteDMatrixCreateFromFile
