fn main() {
    add_search_path();
    add_llvm_path();
    build_lib();
}

#[cfg(feature = "static")]
fn build_lib() {
    let dst = cmake::Config::new("treelite")
        .define("BUILD_STATIC_LIBS", "ON")
        .define("CMAKE_INSTALL_LIBDIR", "lib")
        .build();
    println!("cargo:rustc-link-search={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=treelite_runtime_static");
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=stdc++");
    }
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=c++");
    }
}

#[cfg(feature = "dynamic")]
fn build_lib() {
    let dst = cmake::Config::new("treelite")
        .define("CMAKE_INSTALL_LIBDIR", "lib")
        .build();
    println!("cargo:rustc-link-search={}/lib", dst.display());
    println!("cargo:rustc-link-lib=dylib=treelite_runtime");
}

#[cfg(not(target_os = "windows"))]
fn add_search_path() {
    for path in std::env::var("LD_LIBRARY_PATH")
        .unwrap_or_else(|_| "".to_string())
        .split(":")
    {
        if path.trim().len() == 0 {
            continue;
        }
        println!("cargo:rustc-link-search={}", path);
    }
}

#[cfg(target_os = "windows")]
fn add_search_path() {
    for path in std::env::var("PATH")
        .unwrap_or_else(|_| "".to_string())
        .split(";")
    {
        if path.trim().len() == 0 {
            continue;
        }
        println!("cargo:rustc-link-search={}", path);
    }
}

fn add_llvm_path() {
    if let Some(llvm_config_path) = option_env!("LLVM_CONFIG_PATH") {
        println!("cargo:rustc-env=LLVM_CONFIG_PATH={}", llvm_config_path);
    }
}
