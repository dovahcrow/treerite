bindgen:
    echo "#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, unused)]" > src/sys/bindings.rs
    bindgen treelite/include/treelite/c_api_runtime.h >> src/sys/bindings.rs
