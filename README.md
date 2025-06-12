# TreeRite: [TreeLite](https://github.com/dmlc/treelite) runtime in Rust ![CI](https://github.com/dovahcrow/treerite/workflows/CI/badge.svg)

This binding currently works for treelite 4.3.0.

# Usage

* Make sure your machine has cmake, libstdc++, rapidjson (optional), nlohmann-json (optional) and libgomp installed.
* Add this to your project's `Cargo.toml`.

```toml
treerite  = { git = "https://github.com/dovahcrow/treerite" }
```

The treerite library is static linked to your binary. You don't need to install treelite in your machine.

The library is being revamped. Currently only model loading and prediction is working.

Please see the tests folder and examples folder for detailed usage.
