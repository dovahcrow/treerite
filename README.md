# TreeRite: [TreeLite](https://github.com/dmlc/treelite) runtime in Rust ![CI](https://github.com/dovahcrow/treerite/workflows/CI/badge.svg)

This binding currently works for treelite 4.3.0.

# Usage

* Make sure your machine has cmake, libstdc++, rapidjson, nlohmann-json and libgomp installed.
* Add this to your project's `Cargo.toml`.
  ```toml
  treerite = "0.2"
  ```

By default, the treerite library is static linked to your binary. If you'd like to use the dynamic lib,
set the dynamic feature of `treerite`.

The usage of this binding is straight forward: you first load the `Predictor` from the shared library.
Then, load the data from `&[f64]` or `ndarray::Array2` into `DMatrix`. Finally, you do prediction using
`Predictor::predict_batch(dmatrix)`.

# Documentation

The documentation is hosted at [here](http://dovahcrow.github.io/treerite/treerite/). 
You can also take a look at the [tests](https://github.com/dovahcrow/treerite/tests) folder and the [examples](https://github.com/dovahcrow/treerite/examples) folder.

# TODO

- [ ] TreeliteDMatrixCreateFromCSR
- [ ] TreeliteDMatrixCreateFromFile

