use std::path::Path;
fn main() {
    add_search_path();
    add_llvm_path();
    let dst = cmake::Config::new("treelite").build();
    link_lib(&dst);
}

#[cfg(feature = "static")]
fn link_lib(dst: &Path) {
    println!("cargo:rustc-link-search={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=treelite_runtime_static");
    println!("cargo:rustc-link-lib=stdc++");
}

#[cfg(feature = "dynamic")]
fn link_lib(dst: &Path) {
    println!("cargo:rustc-link-search={}/lib", dst.display());
    println!("cargo:rustc-link-lib=dynamic=treelite_runtime");
}

#[cfg(not(target_os = "windows"))]
fn add_search_path() {
    for path in std::env::var("LD_LIBRARY_PATH").unwrap_or_else(|_| "".to_string()).split(":") {
        if path.trim().len() == 0 {
            continue;
        }
        println!("cargo:rustc-link-search={}", path);
    }
}

#[cfg(target_os = "windows")]
fn add_search_path() {
    for path in std::env::var("PATH").unwrap_or_else(|_| "".to_string()).split(";") {
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
