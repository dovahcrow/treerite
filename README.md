# TreeRite: [TreeLite](https://github.com/dmlc/treelite) runtime in Rust ![CI](https://github.com/dovahcrow/treerite/workflows/treerite%20CI/badge.svg)

This binding currently works for treelite 1.0.0rc1

The library is currently under heavy development. APIs are subject to change and error-prone. 

# Usage

* Make sure your machine has cmake, libstdc++ and libgomp installed.
* Add this to your project's `Cargo.toml`.
  ```toml
  treerite = { version = "0.0.1", git = "https://github.com/dovahcrow/treerite" }
  ```

By default, the treerite library is static linked to your binary. If you'd like to use the dynamic lib,
set the dynamic feature of `treerite`.

# Documentation

There's no documentation available yet. But you can take a look at the example folder.

# TODO

- [ ] TreeliteRegisterLogCallback
- [ ] TreeliteDMatrixCreateFromFile
- [ ] TreeliteDMatrixCreateFromCSR
- [ ] TreelitePredictorQueryNumFeature
- [ ] TreelitePredictorQueryPredTransform
- [ ] TreelitePredictorQuerySigmoidAlpha
- [ ] TreelitePredictorQueryGlobalBias
- [ ] TreelitePredictorQueryThresholdType